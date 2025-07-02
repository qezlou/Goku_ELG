import logging
import numpy as np
#mport pymc3 as pm
from scipy.interpolate import BSpline # Requires scipy>=1.15.0 on python>=3.10
from scipy.optimize import minimize

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
    


class ConstrainedSplineFitter:
    def __init__(self, degree=2, s=None, constraints=True, logging_level='INFO'):
        """
        A Spline fitter which requires the second derivative to be all negative.
        Of particular use in fitting the halo mass function where we expect the slope
        to be decreasing with mass.  THis is similar to what implemented in MiraTitanII arxiv:2003.12116
        Parameters:
        -----------
        degree: int, default=2
            Degree of the spline
        s: int, default=x.size, optional
            Smoothing factor, refer to scipy.interpolate.generate_knots.
            This helps find the internal knots to achieve the desired smoothness.
        constraints: bool, default=True
            If True, enforce the constraints on the second derivative to be negative
        
        """
        self.degree = degree
        self.s = s
        self.constraints = constraints
        self.logger = self.configure_logging(logging_level)

        self.logger.debug(f'Fitting spline with degree={self.degree}, s={self.s}, constraints={self.constraints}')

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('ConstrainedSplineFitter')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def fit_spline(self, x, y, knots, sigma=1):
        """
        Fit a spline with constraints on the coefficients of the quadratic term.

        Parameters:
        x (array): 1D array of x-coordinates.
        y (array): 1D array of y-coordinates.
        knots (array): Internal knot positions for the spline.

        Returns:
        BSpline: A spline object with the specified constraints.
        """
        # Generate internal knots to achieve the desired smoothness
        if self.s is None:
            s = x.size
            if s < 5:
                s = 0
        self.logger.debug(f'Fit for x.shape={x.shape}, y.shape={y.shape}, s={s}')


        # Objective function: Sum of squared errors
        def objective(c):
            return np.sum(( (BSpline(knots, c, self.degree)(x) - y)/sigma )**2)

        # Constraint: For degree=2, all quadratic term coefficients (x^2) must be negative
        if self.degree ==2 and self.constraints:
            constraints = [{'type': 'ineq', 'fun': lambda c: -BSpline(knots, c, self.degree).derivative(2).c}]  # Ensure all coefficients are negative
        else:
            constraints = []
        
        # Optimize the objective function
        result = minimize(
            objective,
            x0=np.zeros(len(knots)),
            constraints=constraints,
            #method='L-BFGS-B',
            options={'disp': False,
            'maxiter': 1000000}
    )

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        # Return the constrained spline
        return BSpline(knots, result.x, self.degree)

