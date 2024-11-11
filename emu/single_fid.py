"""
Train a single fidelity emualtor
"""
import os.path as op
import numpy as np
import gpflow
from gpflow.models import GPR
from gpflow.kernels import SquaredExponential
from gpflow.optimizers import Scipy
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error

class SingleFid:

    def __init__(self, X, Y, model_err=None):
        self.model_err = model_err
        self.n_params = Y.shape[1]
        self.n_samples = X.shape[0]

        # Normalize the input and output data
        (self.X_normalized, self.Y_normalized, 
         self.model_err_normalized, self.X_min, 
         self.X_max, self.Y_mean, self.Y_std) = self.normalize(X, Y, model_err)
    
    def normalize(self, X, Y, model_err):
        """
        Normalize the input and output data
        Input is normalized between 0 and 1
        Output is normalized with mean 0 and std 1
        """
        X_min, X_max = tf.reduce_min(X, axis=0), tf.reduce_max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min)
        Y_mean, Y_std = tf.reduce_mean(Y, axis=0), tf.math.reduce_std(Y, axis=0)
        Y_normalized = (Y - Y_mean) / Y_std
        if model_err is not None:
            model_err_normalized = (model_err - Y_mean)/Y_std
        else:
            model_err_normalized = None
        return X_normalized, Y_normalized, model_err_normalized, X_min, X_max, Y_mean, Y_std
    
    def denormalize(self, X, Y):
        """
        Denormalize the input and output data
        """
        X_denormalized = X * (self.X_max - self.X_min) + self.X_min
        Y_denormalized = Y * self.Y_std + self.Y_mean
        return X_denormalized, Y_denormalized
    
    def build_model(self, kernel=None):
        """
        Build the GP model
        """
        if kernel is None:
            kernel = SquaredExponential()
        self.model = GPR(data=(self.X_normalized, self.Y_normalized), kernel=kernel)
        self.model.likelihood.variance.assign(self.model_err)