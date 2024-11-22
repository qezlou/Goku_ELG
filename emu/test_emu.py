import unittest
import numpy as np
from single_fid import EvaluateSingleFid

class TestSingleFiled(unittest.TestCase):
    
    def setUp(self):
        self.nsmaples = 200
        self.ndim_input = 10
        self.ndim_output = 30
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

        # This is similar to log(r_p)
        r_out = 0.1+np.arange(self.ndim_output)
        # This is like W_p(r_p)
        r_dependence = 1/r_out
        # Indices for exponentiation
        indices = np.arange(ndim_input)
        Y_true = np.zeros((nsamples, self.ndim_output))
        for i in range(X.shape[0]):
            cosmo_dependence = np.dot(self.qoeffs, X[i,:]**indices)
            Y_true[i,:] = r_dependence * cosmo_dependence

        return Y_true

    def test_prediction(self):
        """
        Test the prediction
        """
        Y_pred, Y_var = self.eval.sf.predict(self.X_train)
        self.assertTrue(np.allclose(Y_pred, self.Y_true, atol=1e-1), f'Prediction is not close to the truth {np.max(Y_pred/ self.Y_true)}')
        self.assertTrue(np.allclose(Y_var, self.noise_amp**2, atol=1e-2))
        
        


if __name__ == '__main__':
    unittest.main()