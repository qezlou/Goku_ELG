"""
Hold all the emulation part of the code.
"""

import pickle
import logging
import sys
import numpy as np
import gpflow
from mfgpflow import linear_svgp
from os import path as op


class BaseEmulator:
    """
    Base emulator class to inherit from.
    """
    def __init__(self, loggin_level='INFO', logger_name='BaseEmulator'):
        """
        Initialize the base emulator.

        Parameters
        ----------
        loggin_level : str, optional
            Logging level. Default is 'INFO'.
        """
        self.logger = self.configure_logging(logger_name=logger_name, logging_level=loggin_level)
        # Min and Max of the cosmological parameters.
        # Scale the desired parameters to the range of the emulator,
        # as the emulator is trained for parameters between 0 and 1.
        self.cosmo_min = np.array([2.20120000e-01, 4.00100000e-02, 6.00106667e-01, 1.00133333e-09,
                                   8.00200000e-01, -1.29896667e+00, -2.99766667e+00, 2.20153333e+00,
                                   -4.99333333e-02, 4.00000000e-04])
        self.cosmo_max = np.array([3.99880000e-01, 5.49900000e-02, 7.59893333e-01, 2.99866667e-09,
                                   1.09980000e+00, 2.46900000e-01, 4.65000000e-01, 4.49846667e+00,
                                   4.99333333e-02, 5.99600000e-01])
    
    def configure_logging(self, logger_name, logging_level='INFO'):
        """
        Set up logging based on the provided logging level in an MPI environment.

        Parameters
        ----------
        logger_name : str
            Name of the logger.
        logging_level : str, optional
            Logging level. Default is 'INFO'.

        Returns
        -------
        logger : logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Logger Name, Timestamp, Level, and Message in format
        formatter = logging.Formatter(
            '%(name)s | %(asctime)s | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger

    def _get_emu(self, model_file):
        """
        Load the trained emulator from a file.

        Parameters
        ----------
        model_file : str
            Path to the model file.

        Returns
        -------
        emu : linear_svgp.LatentMFCoregionalizationSVGP
            Loaded emulator instance.
        """
        self.logger.debug(f'Loading the emulators from {model_file}')
        with open(model_file, 'rb') as f:
            hyper_params = pickle.load(f)
        # Load the model hyperparameters
        num_outputs, num_latents = hyper_params['.kernel.W'].shape
        num_inducing = hyper_params['.inducing_variable.inducing_variable.Z'].shape[0]
        num_params = hyper_params['.inducing_variable.inducing_variable.Z'].shape[1] - 1

        kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(num_params, np.float64), variance=np.float64(1.0))
        kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(num_params, np.float64), variance=np.float64(1.0))
        # The X_train is just a placeholder, dummy array
        X_train = np.random.rand(600, num_params+1)


        emu = linear_svgp.LatentMFCoregionalizationSVGP(
                                X_train, None, kernel_L, 
                                kernel_delta, 
                                num_latents=num_latents, 
                                num_inducing=num_inducing,
                                num_outputs=num_outputs,
                                heterosed=True)

    
        # Set the hyperparameters
        gpflow.utilities.multiple_assign(emu, hyper_params)
        self.logger.debug(f' num_outputs: {num_outputs}, num_latents: {num_latents}, num_inducing: {num_inducing}, num_params: {num_params}')
        return emu

    def _prepare_cosmo(self, cosmo_pars):
        """
        Prepare the cosmological parameters for the emulator.

        Scale to (0,1) as the emulator is trained on and add
        the fidelity level column (1 for high fidelity).

        Parameters
        ----------
        cosmo_pars : list or np.ndarray
            The desired cosmological parameters to predict.

        Returns
        -------
        X : np.ndarray
            The prepared cosmological parameters for the emulator.
        """
        # Scale the cosmological parameters to (0,1) as the emulator is trained on
        X = []
        for i in range(len(cosmo_pars)):
            X.append((cosmo_pars[i] - self.cosmo_min[i]) / (self.cosmo_max[i] - self.cosmo_min[i]))
        X = np.array(X).reshape(1, -1)
        # Add the fidelity index, we want high fidelity level prediction
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X
    
class Hmf(BaseEmulator):
    """
    Trained emulator for the halo mass function (HMF).
    """
    def __init__(self, loggin_level='INFO'):
        """
        Initialize the multifidelity emulator for the HMF.

        Parameters
        ----------
        loggin_level : str, optional
            Logging level. Default is 'INFO'.
        """
        super().__init__(loggin_level=loggin_level)
        self.mbins = np.array([11.15, 11.25, 11.35, 11.45, 11.55, 11.65, 11.75, 11.85, 11.95,
                               12.05, 12.15, 12.25, 12.35, 12.45, 12.55, 12.65, 12.75, 12.85,
                               12.95, 13.05, 13.15, 13.25, 13.35, 13.45])
        self.emu = self.get_emu()

    def get_emu(self):
        """
        Load the trained emulator.

        Returns
        -------
        emu : linear_svgp.LatentMFCoregionalizationSVGP
            Loaded emulator instance.
        """
        # TODO: Replace this with the GP trained on the full simulation suite
        model_file = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/train/hmf_emu_combined_inducing_500_latents_20_leave31.pkl'
        return self._get_emu(model_file=model_file)

    def predict(self, cosmo_pars):
        """
        Predict the halo mass function for a given set of cosmological parameters.

        Parameters
        ----------
        cosmo_pars : list or np.ndarray
            The desired cosmological parameters to predict.

        Returns
        -------
        mu : np.ndarray
            Predicted mean halo mass function.
        std : np.ndarray
            Predicted standard deviation of the halo mass function.
        """
        X = self._prepare_cosmo(cosmo_pars)
        # Predict the halo mass function for the given cosmology
        mu, var = self.emu.predict_f(X)
        mu = 10**mu.numpy()
        var = (10**np.sqrt(var.numpy()))**2
        return mu, np.sqrt(var)

class Xi(BaseEmulator):
    """
    Trained emulator for the xi(r) correlation function.
    """
    def __init__(self, loggin_level='INFO'):
        """
        Initialize the multifidelity emulator for xi(r).

        Parameters
        ----------
        loggin_level : str, optional
            Logging level. Default is 'INFO'.
        """
        super().__init__(loggin_level=loggin_level, logger_name='XiEmulator')
        
        # The mass pairs the emulator was trained on
        #mbins = np.arange(13, 10.9, -0.1)
        mbins = np.arange(12.3, 11, -0.1)
        idx = np.triu_indices(len(mbins), k=0)
        self.mass_pairs = np.column_stack((mbins[idx[0]], mbins[idx[1]]))
        self.mass_pairs = np.round(self.mass_pairs, 1)
        # The final xi(r, m1, m2) will be in increasing order of the mass pairs
        self.mass_bins = np.round(mbins, 1)[::-1] 
        
        # The rbins used
        self.rbins = np.array([0.21268578, 0.26343238, 0.32628708, 0.40413884, 0.50056595, 0.62000047, 
                  0.76793194, 0.9511597, 1.17810543, 1.45920017, 1.80736382, 2.27499553, 
                  2.90060912, 3.69826365, 4.71526961, 6.01194766, 7.66520638, 9.77310384, 
                  12.46066368, 15.88729045, 20.25622426, 25.82659532, 32.9287935, 41.98406441, 
                  53.52949432, 60.52631579])

        self.emu = self.get_emu()
    
    def get_emu(self):
        """
        Load the trained emulator.

        Returns
        -------
        emu : linear_svgp.LatentMFCoregionalizationSVGP
            Loaded emulator instance.
        """
        # TODO: Replace this with the GP trained on the full simulation suite
        emu_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/train_less_massive/'
        model_file = op.join(emu_dir, 'xi_emu_combined_inducing_500_latents_40_leave2.pkl')
        self.logger.info(f'Loading the xi emulator from {model_file}')
        return self._get_emu(model_file=model_file)
    

    def predict(self, cosmo_pars):
        """
        Predict the xi_hh for a given set of cosmological parameters.

        Parameters
        ----------
        cosmo_pars : list or np.ndarray
            The desired cosmological parameters to predict.

        Returns
        -------
        mu : np.ndarray
            Predicted mean 3D correlation function xi(r).
        std : np.ndarray
            Predicted standard deviation of the 3D correlation function.
        """
        X = self._prepare_cosmo(cosmo_pars)
        # Predict the xi_hh for the given cosmology
        mu, var = self.emu.predict_f(X)
        mu = 10**mu.numpy()
        var = (10**np.sqrt(var.numpy()))**2

        mu = mu.reshape(-1, len(self.mass_pairs), len(self.rbins))
        var = var.reshape(-1, len(self.mass_pairs), len(self.rbins))
        mu = mu.squeeze()
        var = var.squeeze()
        mu = self.make_3d_corr(mu, symmetric=True)
        var = self.make_3d_corr(var, symmetric=True)
        return mu, np.sqrt(var)

    def make_3d_corr(self, corr, symmetric=True):
        """
        Convert xi data from 2D mass pairs and rbins to a 3D array.

        Parameters
        ----------
        corr : np.ndarray
            Array of shape (n_mass_pairs, n_rbins) with correlation data.
        symmetric : bool, optional
            Whether to mirror the correlation matrix to be symmetric. Default is True.

        Returns
        -------
        corr_3d : np.ndarray
            3D array of shape (n_mass_bins, n_mass_bins, n_rbins).
        """
        ind_m_pair = np.digitize(self.mass_pairs, self.mass_bins).astype(int) - 1
        corr_3d = np.full((self.mass_bins.size, self.mass_bins.size, corr.shape[-1]), np.nan)
        for (i, j, val) in zip(ind_m_pair[:, 0], ind_m_pair[:, 1], corr):
            corr_3d[i, j] = val
            if symmetric:
                corr_3d[j, i] = val
        return corr_3d