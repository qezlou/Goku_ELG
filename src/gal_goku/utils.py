import numpy as np
import pymc3 as pm
from scipy.interpolate import PPoly

"""
Common utilities for all emulators
"""

class BayesianPiecewisePolynomialFitter:
    def __init__(self):
        """
        Initialize the fitter for piecewise quadratic polynomials with Bayesian inference.
        """
        self.degree = 2  # Quadratic polynomial

    def fit_piecewise(self, x, y, m):
        """
        Fit piecewise quadratic polynomials to m consecutive bins using Bayesian inference.

        Parameters:
        x (array): 1D array of x-coordinates.
        y (2D array): 2D array of y-coordinates with shape (num_examples, num_dimensions).
        m (int): Number of bins to group for fitting.

        Returns:
        list: A list of lists containing posterior samples for polynomial coefficients. Outer list corresponds to dimensions.
        """
        if len(x) != y.shape[1]:
            raise ValueError("Length of x must match the second dimension of y.")
        num_examples, num_dimensions = y.shape
        all_posteriors = []

        for dim in range(num_dimensions):
            dimension_posteriors = []
            
            for i in range(0, len(x) - m + 1, m - 1):
                x_segment = x[i:i + m]
                y_segment = y[:, dim][:, i:i + m]

                if len(x_segment) < m:
                    break

                with pm.Model() as model:
                    # Prior distributions for coefficients
                    a = pm.Normal("a", mu=0, sigma=10)
                    b = pm.Normal("b", mu=0, sigma=10)
                    c = pm.Normal("c", mu=0, sigma=10) 

                    # Polynomial model
                    mu = a * x_segment**2 + b * x_segment + c

                    # Likelihood
                    sigma = pm.HalfNormal("sigma", sigma=1)
                    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_segment)

                    # Perform MCMC sampling
                    trace = pm.sample(1000, return_inferencedata=False, progressbar=True)

                dimension_posteriors.append(trace)

            all_posteriors.append(dimension_posteriors)

        return all_posteriors
    


    def evaluate_piecewise(self, x, posteriors, x_eval):
        """
        Evaluate the piecewise quadratic model at new points using posterior samples.

        Parameters:
        x (array): Original x-coordinates used in fitting.
        posteriors (list): List of posterior samples for each segment and dimension.
        x_eval (array): Points at which to evaluate the piecewise polynomials.

        Returns:
        np.ndarray: Evaluated values at x_eval.
        """
        results = np.zeros((len(posteriors), len(x_eval)))
        
                segment_mask = (x_eval >= x[segment]) & (x_eval < x[segment + 1])
                x_segment_eval = x_eval[segment_mask]

                if len(x_segment_eval) > 0:
                    for a, b, c in zip(a_samples, b_samples, c_samples):
                        results[dim, segment_mask] += a * x_segment_eval**2 + b * x_segment_eval + c

                    results[dim, segment_mask] /= len(a_samples)

        return results
