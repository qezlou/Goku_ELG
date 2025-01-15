from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np

"""
Common utilities for all emulators
"""


class PiecewisePoly:
    """
    Fitting Piecewise Polynomial to the summary statistics
    """
    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64], d: int, window: int):
        """
        Initialize the PiecewisePoly object.
        Parameters:
        ------------
        x: np.ndarray, shape (n,)
            The x values of the summary statistics
        y: np.ndarray, shape (m, n)
            The y values of the summary statistics for all the simulations
        d: int
            The degree of the polynomial
        window: int
            Number of consecutive bins of x within which the polynoomial of degree d
        """
        self.x = x
        self.y = y
        self.n = len(x)
        self.m = len(y[0])

    def __call__(self, x):
        """
        Evaluate the piecewise polynomial at x
        """
        if x < self.x[0]:
            return self.y[0]
        if x > self.x[-1]:
            return self.y[-1]
        for i in range(self.n - 1):
            if x < self.x[i + 1]:
                break
        return sum([self.y[i][j] * (x - self.x[i]) ** j for j in range(self.m)])