"""
Train a single fidelity emualtor
"""
import logging
import os.path as op
import numpy as np
import argparse
import h5py
import gpflow
from gpflow.models import GPR
from gpflow.kernels import SquaredExponential
from gpflow.optimizers import Scipy
from gpflow.utilities import print_summary
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error

import summary_stats

class SingleFid:

    def __init__(self, X, Y, model_err=None, logging_level='INFO'):
        self.logger = self.configure_logging(logging_level)
        self.model_err = model_err
        self.n_params = Y.shape[1]
        self.n_samples = X.shape[0]
        assert Y.ndim == 2, 'Input data should be 2D'
        assert X.ndim == 2, 'Output data should be 2D'
        self.X = X
        self.Y = Y
        assert np.any(np.isnan(self.Y))==False , f'Y has nans at {np.argwhere(np.isnan(self.Y))}'
        
        # Normalize the input and output data
        (self.X_min,  self.X_max, 
         self.mean_func, self.model_err) = self.normalize(X, Y, model_err)
        
        self .model = None

    
    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('SingleFid')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
    
    def normalize(self, X, Y, model_err):
        """
        Normalize the input data such it is between 0 and 1
        We don't normalize the output data as it the median of the stack
        could be 0.
        Returns:
        --------
        X_normalized: normalized input data between 0 and 1
        X_min: minimum value of the input data
        X_max: maximum value of the input data
        mean_func: the mean of the output to be used as the mean function
        in the GP model
        """
        X_min, X_max = tf.reduce_min(X, axis=0), tf.reduce_max(X, axis=0)

        medind = np.argsort(np.mean(Y, axis=1))[np.shape(Y)[0]//2]
        mean_func = Y[medind,:]
        c = -1
        while np.any(np.isnan(mean_func)):
            medind = (medind + 1) % np.shape(Y)[0]
            mean_func = Y[medind,:]
            c += 1
        model_err = model_err[medind,:]
        assert np.any(np.isnan(model_err)==False)
        self.logger.debug(f'moved {c} steps out of {Y.shape[0]} sims to find a non-nan median')
        return X_min, X_max, mean_func, model_err
    

    def build_model(self, X_train=None, Y_train=None, kernel=None):
        """
        Build the GP model
        """
        if X_train is None:
            X_train_norm = (self.X - self.X_min) / (self.X_max - self.X_min)
            Y_train = self.Y
        else:
            X_train_norm = (X_train - self.X_min) / (self.X_max - self.X_min)
        
        mean_func = self.mean_func
        dtype = self.Y.dtype
        if kernel is None:
            initial_lengthscales = np.ones(X_train_norm.shape[1])
            self.kernel = SquaredExponential(lengthscales=initial_lengthscales)
            self.kernel.lengthscales.trainable = True
            self.kernel.variance.trainable = True
        
        class MeanFunction(gpflow.mean_functions.MeanFunction):
            def __call__(self, X):
                return tf.convert_to_tensor(mean_func, dtype=dtype)
        #kernel.lengthscales = tf.Variable([0.5] * X_train_norm.shape[1], dtype=dtype)
        self.model = GPR(data=(X_train_norm, Y_train), kernel=self.kernel, 
                 mean_function=MeanFunction())

        self.logger.info(print_summary(self.model))
        
        ## I get this error for the noise:
        ## Shape mismatch.The variable shape (), and the assigned value shape (65,)
        #self.model.likelihood.variance.assign(self.model_err)

    def train(self, X_train=None, Y_train=None, max_iter=1000_000_000):
        """
        Train the GP model
        """
        self.build_model(X_train, Y_train)
        optimizer = Scipy()
        optimizer.minimize(self.model.training_loss, 
                        variables=self.model.trainable_variables,
                        options=dict(maxiter=max_iter))
        self.logger.info(f'trained hyperparameters: lengthscales: {self.kernel.lengthscales.numpy()}')
        self.logger.info(f'trained hyperparameters: variance: {self.kernel.variance.numpy()}')
        # Get the training loss
        #training_loss = self.model.training_loss().numpy()
        #self.logger.info(f'Training loss: {training_loss}')

    def predict(self, X):
        """
        Predict the output for the given input
        """
        X_norm = (X - self.X_min) / (self.X_max - self.X_min)
        mean, var = self.model.predict_y(X_norm)
        return mean, var

class EvaluateSingleFid:
    def __init__(self, X, Y, model_err, logging_level='INFO'):
        self.logger = self.configure_logging(logging_level)
        self.X = X
        self.Y = Y
        self.model_err = model_err
        self.n_params = Y.shape[1]
        self.n_samples = X.shape[0]
        assert Y.ndim == 2, 'Input data should be 2D'
        assert X.ndim == 2, 'Output data should be 2D'
        self.logger.info(f'X shape: {X.shape}, Y shape: {Y.shape}, model_err shape: {model_err.shape}')
        self.logger.info(f'Number of parameters: {self.n_params}')
        self.sf = SingleFid(X = X, Y = Y, model_err = model_err, logging_level='DEBUG')
    
    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('EvaluateSingleFid')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
    
    def loo_train_pred(self, rp, savefile=None, labels=None):
        """
        Iterate over the samples to leave one out and train the model
        Parameters:
        -------
        rp: array
            The projected rp radii that used for training
        savefile: 
            The file to save the results to

        Returns:
        --------
        mean_pred: the predictions for the left out samples while trained on the rest
        var_pred: the variance of the predictions
        """
        n_samples = self.Y.shape[0]
        mean_pred = np.zeros((n_samples, self.Y.shape[1]) )
        var_pred = np.zeros((n_samples, self.Y.shape[1]) )
        
        progress = np.arange(0, 1, 0.01)
        print_marks = (progress*n_samples).astype(int)

        for i in range(n_samples):
            if i in print_marks:
                self.logger.info(f'Progress: {i/n_samples*100:.2f}%')
            X_train = np.delete(self.X, i, axis=0)
            Y_train = np.delete(self.Y, i, axis=0)
            X_test = self.X[i][np.newaxis, :]
            Y_test = self.Y[i]
            self.sf.train(X_train, Y_train)
            mean_pred[i], var_pred[i] = self.sf.predict(X_test)
        if savefile is not None:
            with h5py.File(savefile, 'w') as f:
                f.create_dataset('pred', data=mean_pred)
                f.create_dataset('var_pred', data=var_pred)
                f.create_dataset('truth', data=self.Y)
                f.create_dataset('X', data=self.X)
                f.create_dataset('rp', data=rp)
                if labels is not None:
                    f.create_dataset('labels', data=labels)
        else:
            return mean_pred, var_pred
    
    def loo_errors(self):
        """
        Compute the leave one out errors
        """
        mean_pred, var_pred = self.loo_train_pred()
        #loo_errors[i] = (mean_pred/Y_test - 1)**2
        #loo_errors = np.mean(loo_errors, axis=0)
        #loo_errors = np.sqrt(loo_errors)
        # return loo_errors
    

class TestSingleFiled():
    
    def __init__(self):
        self.nsmaples = 200
        self.ndim_input = 10
        self.ndim_output = 18

        # This is similar to log(r_p)
        r_edges = np.logspace(-1.5, np.log10(2), 8)
        r_edges = np.append(r_edges, np.logspace(np.log10(2), np.log10(30), 10)[1:])
        self.r_out = 0.5*(r_edges[1:] + r_edges[:-1])
        self.ndim_output = len(self.r_out)

        self.qoeffs = np.array([1.45, 3.54, 5.43, 0.54, 
                                0.43, 0.32, 0.21, 0.10, 
                                0.05, 0.03])
        assert len(self.qoeffs) == self.ndim_input
        self.X_train = np.random.uniform(0, 1, (self.nsmaples, self.ndim_input))
        self.Y_true = self.get_truth(self.X_train)
    
        # I amy need to scale the noise amplitude
        self.noise_amp = 0.1
        self.Y_train = self.Y_true + np.random.normal(0, self.noise_amp, (self.ndim_output,))


        # Train the model
        model_err = np.ones_like(self.Y_train) * self.noise_amp
        self.eval = EvaluateSingleFid(self.X_train, self.Y_train, model_err)
        # Train with all the training data
        self.eval.sf.train()
    
    def get_truth(self, X):
        """
        I assume the each bin is realted to the cosmolgical parameters
        in a polynomial way
        """
        nsamples, ndim_input = X.shape

        # This is like W_p(r_p)
        r_dependence = -np.log10((0.01*(1+ self.r_out)))
        # Indices for exponentiation
        indices = np.arange(ndim_input)
        Y_true = np.zeros((nsamples, self.ndim_output))
        for i in range(X.shape[0]):
            cosmo_dependence = np.dot(self.qoeffs, X[i,:]**indices)
            Y_true[i,:] = r_dependence * cosmo_dependence

        return Y_true
    
    def predict(self, X=None):
        """
        Get the predictionfor this simple test
        """
        if X is None:
            X = self.X_train
        Y_pred, Y_var = self.eval.sf.predict(X)
        return Y_pred, Y_var


    
