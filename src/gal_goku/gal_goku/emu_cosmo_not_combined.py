import pickle
import logging
import sys
import numpy as np
import gpflow
from mfgpflow import linear_svgp
from os import path as op

class XiEmulator:
    def __init__(self, dat_dir, loggin_level='INFO'):
        """
        Loading them ultifidelity emulator for the xi(r)
        Parameters
        ----------
        emu_dir : str
            Directory where the emulator is stored
        mass_pair : tuple
            Tuple with the mass pair for the emulator
        loggin_level : str
            Logging level. Default is INFO
        
        """

        self.logger = self.configure_logging(logging_level=loggin_level)
        self.emu_dir = op.join(dat_dir, 'train_larger_iters')
        
        # The mass pairs we have trained the emulator on
        mbins = np.arange(13, 10.9,-0.1 )
        idx = np.triu_indices(len(mbins), k=0)
        self.mass_pairs = np.column_stack((mbins[idx[0]], mbins[idx[1]]))
        self.mass_pairs = np.round(self.mass_pairs, 1)
        
        # Min and MAx of the cosmo parameters
        # We need to scale the desired parameters to the range of the emulator
        # As the emualtor is trained for paramters between 0 and 1
        self.cosmo_min = np.array([2.20120000e-01, 4.00100000e-02, 6.00106667e-01, 1.00133333e-09,
                                   8.00200000e-01, -1.29896667e+00, -2.99766667e+00, 2.20153333e+00,
                                   -4.99333333e-02, 4.00000000e-04])
        self.cosmo_max = np.array([3.99880000e-01, 5.49900000e-02, 7.59893333e-01, 2.99866667e-09,
                                   1.09980000e+00, 2.46900000e-01, 4.65000000e-01, 4.49846667e+00,
                                   4.99333333e-02, 5.99600000e-01])
        
        # The rbins used:
        self.rbins = np.array([0.21268578, 0.26343238, 0.32628708, 0.40413884, 0.50056595, 0.62000047, 
                  0.76793194, 0.9511597, 1.17810543, 1.45920017, 1.80736382, 2.27499553, 
                  2.90060912, 3.69826365, 4.71526961, 6.01194766, 7.66520638, 9.77310384, 
                  12.46066368, 15.88729045, 20.25622426, 25.82659532, 32.9287935, 41.98406441, 
                  53.52949432, 60.52631579])
    
    def configure_logging(self, logging_level='INFO'):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger('XiEmulator')
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    def get_emu(self, mass_pair):
        """
        load the trained emulator
        """
        # We usually have the mass pair in descending order
        if mass_pair[1] > mass_pair[0]:
            mass_pair = (mass_pair[1], mass_pair[0])
        
        self.logger.debug(f'Loading the emulators from {self.emu_dir}')
        # Load the dicitonary storing the emulator
        self.emus = []
        model_file = op.join(self.emu_dir, f'Xi_Native_emu_mapirs2_spline_{mass_pair[0]}_{mass_pair[1]}_wide_narrow_leave_12.pkl')
        with open(model_file, 'rb') as f:
            hyper_params = pickle.load(f)
        # Load the model hyperparameters
        num_outputs, num_latents = hyper_params['.kernel.W'].shape
        num_inducing = hyper_params['.inducing_variable.inducing_variable.Z'].shape[0]
        num_params = hyper_params['.inducing_variable.inducing_variable.Z'].shape[1] - 1

        kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(num_params), variance=1.0)
        kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(num_params), variance=1.0)
        # The X_train is just a placeholder, dummmy array
        X_train = np.random.rand(200, num_params+1)


        emu = linear_svgp.LatentMFCoregionalizationSVGP(
                                X_train, None, kernel_L, 
                                kernel_delta, 
                                num_latents=num_latents, 
                                num_inducing=num_inducing,
                                num_outputs=num_outputs)

    
        # Set the hyperparameters
        gpflow.utilities.multiple_assign(emu, hyper_params)
        self.logger.debug(f' num_outputs: {num_outputs}, num_latents: {num_latents}, num_inducing: {num_inducing}, num_params: {num_params}')
        return emu
    
    def predict_xi(self, mthresh1, mthresh2, cosmo):
        """
        Predict the xi_hh for a given set of cosmological parameters and halo masses
        Parameters
        ----------
        mthresh1 : float
            The minimum halo mass for the first halo sample.
        mthresh2 : float
            The minimum halo mass for the second halo sample.
        r : array_like
            The distances at which to calculate the correlation function.
        cosmo : list or np.ndarrays
            The desired cosmology to predict the xi_hh for
        Returns
        -------
        xi : array_like
            The 3D correlation function xi(r).
        """
        # Get the emulator for the given mass pair
        emu = self.get_emu((mthresh1, mthresh2))
        # scale the cosmo parameters to (0,1) as the emulator is trained on
        X = []
        for i in range(len(cosmo)):
            X.append((cosmo[i] - self.cosmo_min[i]) / (self.cosmo_max[i] - self.cosmo_min[i]))
        X = np.array(X).reshape(1, -1)
        # Add the fidelity index, we want HF level prediction
        # Add the fidelity level column (1 for high fidelity)
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        # Predict the xi_hh for the given cosmology
        mu, var = emu.predict_f(X)
        mu = 10**mu.numpy()
        var = (10**np.sqrt(var.numpy()))**2
        return mu.flatten(), var.flatten()
    
    def predict_xi_on_grid(self, cosmo):
        """
        Precit the xi_hh on the predefined (r, Mth1, Mth2) grid
        Parameters
        ----------
        cosmo : list or np.ndarrays
            The desired cosmology to predict the xi_hh for
        """
        X = []
        for i in range(len(cosmo)):
            X.append((cosmo[i] - self.cosmo_min[i]) / (self.cosmo_max[i] - self.cosmo_min[i]))
        X = np.array(X).reshape(1, -1)
        assert np.all(X >= 0) and np.all(X <= 1), f"Cosmo parameters are out of range: {X}"
        # Add the fidelity index, we want HF level prediction
        # Add the fidelity level column (1 for high fidelity)
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        xi_pred = []
        var_pred = []
        for i, mass_pair in enumerate(self.mass_pairs):
            # Get the emulator for the given mass pair
            emu = self.get_emu(mass_pair)
            # Predict the xi_hh for the given cosmology
            mu, var = emu.predict_f(X)
            xi_pred.append(mu.numpy().flatten())
            var_pred.append(var.numpy().flatten())
        return np.array(xi_pred), np.array(var_pred)

        