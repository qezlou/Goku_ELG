"""
Train a single fidelity emualtor
"""
import logging
import numpy as np
import h5py
import gpflow
from gpflow.models import GPR
from gpflow.kernels import SquaredExponential
from gpflow.optimizers import Scipy
from gpflow.utilities import print_summary
import tensorflow as tf

class SingleFid:

    def __init__(self, X, Y, model_err=None, logging_level='INFO'):
        self.logger = self.configure_logging(logging_level)
        self.model_err = model_err
        self.ndim_input = X.shape[1]
        self.n_samples = X.shape[0]
        assert Y.ndim == 2, 'Ouput data should be 2D'
        assert X.ndim == 2, 'Input data should be 2D'
        self.X = X
        self.Y = Y
        assert np.any(np.isnan(self.Y))==False , f'Y has nans at {np.argwhere(np.isnan(self.Y))}'
        
        # Normalize the input and output data
        (self.X_min,  self.X_max, 
         self.mean_func, self.model_err) = self.normalize(X, Y, model_err)
        
        self.model = None

    
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

        if self.logger.level == logging.DEBUG:
            self.logger.debug(print_summary(self.model))
        
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
        self.logger.debug(f'trained hyperparameters: lengthscales: {self.kernel.lengthscales.numpy()}')
        self.logger.debug(f'trained hyperparameters: variance: {self.kernel.variance.numpy()}')
        # Get the training loss
        #training_loss = self.model.training_loss().numpy()
        #self.logger.info(f'Training loss: {training_loss}')

    def predict(self, X):
        """
        Predict the output for the given input
        """
        X_norm = (X - self.X_min) / (self.X_max - self.X_min)
        assert np.all(X_norm >= -0.1) and np.all(X_norm <= 1.1), f'Input data should be normalized, X_min, X_max = {np.min(X_norm)}, {np.max(X_norm)}' 
        mean, var = self.model.predict_y(X_norm)
        return mean, var

class EvaluateSingleFid:
    def __init__(self, X, Y, model_err, logging_level='INFO'):
        self.logger = self.configure_logging(logging_level)
        self.X = X
        self.Y = Y
        self.model_err = model_err
        self.ndim_input = X.shape[1]
        self.n_samples = X.shape[0]
        assert Y.ndim == 2, 'Output data should be 2D'
        assert X.ndim == 2, 'Input data should be 2D'
        self.logger.debug(f'X shape: {X.shape}, Y shape: {Y.shape}, model_err shape: {model_err.shape}')
        self.sf = SingleFid(X = X, Y = Y, model_err = model_err, logging_level='INFO')
    
    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('EvaluateSingleFid')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
    
    def train(self, X_train=None, Y_train=None):
        """
        Train the model
        """
        if X_train is not None:
            self.sf.train(X_train, Y_train)
        else:
            self.sf.train()

    def predict(self, X):
        """
        Predict the output for the given input
        """
        return self.sf.predict(X)

    def loo_train_pred(self, mbins, savefile=None, labels=None, sub_sample=None):
        """
        Iterate over the samples to leave one out and train the model
        Parameters:
        -------
        mbins: array
            The projected bins radii that used for training
        savefile: 
            The file to save the results to

        Returns:
        --------
        mean_pred: the predictions for the left out samples while trained on the rest
        var_pred: the variance of the predictions
        """
        if sub_sample is None:
            sample = np.arange(self.n_samples)
        else:
            sample = sub_sample
        mean_pred = np.zeros((sample.size, self.Y.shape[1]) )
        var_pred = np.zeros((sample.size, self.Y.shape[1]) )
        
        progress = np.arange(0, 1, 0.01)
        print_marks = (progress * sample.size).astype(int)

        for i, s in enumerate(sample):
            if i in print_marks:
                self.logger.info(f'Progress: {i/sample.size*100:.2f}%')
            X_train = np.delete(self.X, s, axis=0)
            Y_train = np.delete(self.Y, s, axis=0)
            X_test = self.X[s][np.newaxis, :]
            Y_test = self.Y[s]
            self.sf.train(X_train, Y_train)
            mean_pred[i], var_pred[i] = self.sf.predict(X_test)
        if savefile is not None:
            with h5py.File(savefile, 'w') as f:
                f.create_dataset('pred', data=mean_pred)
                f.create_dataset('var_pred', data=var_pred)
                f.create_dataset('truth', data=self.Y[sub_sample])
                f.create_dataset('X', data=self.X[sub_sample])
                f.create_dataset('bins', data=mbins)
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

    def leave_bunch_out(self, n_out=1, sub_sample=None):
        """
        Leave one out cross validation
        """
        if sub_sample is None:
            sample = self.n_samples
        else:
            sample = sub_sample
        indices = np.sort(np.random.choice(sample, n_out, replace=False))

        self.logger.info(f'Leaving out {len(indices)} samples')
        X_train = np.delete(self.X, indices, axis=0)
        Y_train = np.delete(self.Y, indices, axis=0)
        self.logger.info(f'Training on {X_train.shape[0]} samples')
        
        self.sf.train(X_train, Y_train)

        X_test = self.X[indices]
        Y_test = self.Y[indices]
        Y_pred, var_pred = self.sf.predict(X_test)

        return  X_test, Y_test, Y_pred, var_pred
    
    def load_saved_loo(self, savefile):
        """
        Load the saved leave one out predictions
        """
        with h5py.File(savefile, 'r') as f:
            mean_pred = f['pred'][:]
            var_pred = f['var_pred'][:]
            truth = f['truth'][:]
            X = f['X'][:]
            bins = f['bins'][:]
            try:
                labels = f['labels'][:]
            except KeyError:
                labels = None
        return mean_pred, var_pred, truth, X, bins, labels

class EvaluateSingleFidMultiBins(EvaluateSingleFid):
    """Build a single fidelity emulator for each bin of the summary statistics"""
    def __init__(self, X, Y, model_err, logging_level='INFO'):
        """
        """
        self.logger = self.configure_logging(logging_level)
        self.evalutors = []
        self.X = X
        self.Y = Y
        self.model_err = model_err

        self.ndim_input = X.shape[1]
        self.ndim_output = Y.shape[1]
        self.n_samples = X.shape[0]
        
        for i in range(self.ndim_output):
            if i == 0:
                logging_level_to_pass = logging_level
            else:
                logging_level_to_pass = 'ERROR'
            self.evalutors.append(EvaluateSingleFid(X, Y[:,i][:, None], model_err[:,i][:, None], logging_level_to_pass))

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('EvaluateSingleFidMultiBins')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
    
    def train(self, X_train=None, Y_train=None):
        """
        Train the model
        """
        for i in range(self.ndim_output):
            if X_train is not None:
                self.evalutors[i].train(X_train, Y_train[:,i][:, None])
            else:
                self.evalutors[i].train(self.X, self.Y[:,i][:, None])
    
    def predict(self, X):
        """
        Predict the output for the given input
        """
        Y_pred = np.zeros((X.shape[0], self.ndim_output))
        Y_var = np.zeros((X.shape[0], self.ndim_output))
        for i in range(self.ndim_output):
            Y_pred[:,i], Y_var[:,i] = self.evalutors[i].predict(X)
        return Y_pred, Y_var
    
    def loo_train_pred(self, bins, savefile=None, labels=None):
        """
        Iterate over the samples to leave one out and train the model
        Parameters:
        -------
        bins: array
            The projected bins radii that used for training
        savefile:
            The file to save the results to
        """
        mean_pred = np.zeros((self.n_samples, self.ndim_output) )
        var_pred = np.zeros((self.n_samples, self.ndim_output) )
        
        # Just for logging
        progress = np.arange(0, 1, 0.01)
        print_marks = (progress*self.n_samples).astype(int)

        for i in range(self.n_samples):
            if i in print_marks:
                self.logger.info(f'Progress: {i/self.n_samples*100:.2f}%')
            X_train = np.delete(self.X, i, axis=0)
            Y_train = np.delete(self.Y, i, axis=0)
            X_test = self.X[i][np.newaxis, :]
            Y_test = self.Y[i]
            self.logger.debug(f'X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
            self.train(X_train, Y_train)
            mean_pred[i], var_pred[i] = self.predict(X_test)
        if savefile is not None:
            with h5py.File(savefile, 'w') as f:
                f.create_dataset('pred', data=mean_pred)
                f.create_dataset('var_pred', data=var_pred)
                f.create_dataset('truth', data=self.Y)
                f.create_dataset('X', data=self.X)
                f.create_dataset('bins', data=bins)
                if labels is not None:
                    f.create_dataset('labels', data=labels)
        else:
            return mean_pred, var_pred
    
    def leave_bunch_out(self, n_out=1):
        raise NotImplementedError('This is not implemented yet, but it is easy '+
                                  'to copy the code from the EvaluateSingleFid class')

class TestSingleFiled():
    
    def __init__(self, n_samples=200):
        #np.random.seed = 10
        self.n_samples = n_samples
        self.ndim_input = 2
        self.ndim_output = 3

        self.X_train = np.random.uniform(0, 10, (self.n_samples, self.ndim_input))
        self.Y_train = self.get_truth(self.X_train)
    
        # I amy need to scale the noise amplitude
        self.noise_amp = 0.1
        #self.Y_train = self.Y_true + np.random.normal(0, self.noise_amp, (self.ndim_output,))


        # Train the model
        model_err = np.ones_like(self.Y_train) * self.noise_amp
        self.eval = EvaluateSingleFid(self.X_train, self.Y_train, model_err)
        # Train with all the training data
        self.eval.sf.train()
    
    def get_truth(self, X):
        y = np.zeros((X.shape[0], 3))
        y[:,0] = 60 + 0.9 * X[:,0] * np.sin(X[:,0]) + X[:,1] * np.cos(2*X[:,1])
        y[:,1] = 30 + 0.9 * X[:,0]**0.5 * np.cos(X[:,0])  + X[:,1] * np.tan(0.1*X[:,1])
        y[:,2] = 10 + 0.7 * X[:,0]**0.8 * np.sin(X[:,0])  + 2*X[:,1] * np.tan(0.1*X[:,1])
        return y
        
    def predict(self, X=None):
        """
        Get the predictionfor this simple test
        """
        if X is None:
            X = self.X_train
        Y_pred, Y_var = self.eval.sf.predict(X)
        return Y_pred, Y_var
