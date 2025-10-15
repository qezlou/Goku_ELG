"""
To evaluate all the emulators for the Halo Mass Function.
Evaluating multifid emulator, HF and L2 pairs. Using Ming-Feng's thin wrappers of Emukit.
"""

import logging
import pickle
import h5py
import numpy as np
from . import summary_stats
#from . import single_fid
#from . import gpemulator_singlebin as gpemu
import gpflow
import tensorflow as tf
from mfgpflow.linear_svgp import LatentMFCoregionalizationSVGP
import sys
import os.path as op
import copy
from glob import glob

try :
    #raise ImportError
    import mpi4py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
except ImportError:
    MPI = None
    comm = None
    rank = 0
    mpi_size = 1

class BaseStatEmu():

    def __init__(self, X, Y, 
                 logging_level='info', 
                 emu_type={'multi-fid':False, 'single-bin':False, 'linear':True, 'mf-svgp':False},
                 n_optimization_restarts=5, emu_args=None):
        """The base emu to be inherited by single fidelity emulators built on any summary statistics
        This is the interface for all classses above.
        :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
        :param Y_train:  (n_fidelities, n_points, n_nins) list of matter power spectra.
        """
        self.logger = self.configure_logging(logging_level)
        # This if statement is to make sure that the emu_type has all the keys
        if 'mf-svgp' not in emu_type:
            emu_type['mf-svgp'] = False
        self.emu_type = emu_type
        self.n_optimization_restarts = n_optimization_restarts

        ## TO DO: Better layout for the emu_type
        if emu_type['mf-svgp'] and emu_type['multi-fid'] and emu_type['linear']:
            # Class for Linear multi-fidelity emulator using gpflow's SVGP
            # This helps with dimensionality reduction of the output space
            self.emu = LatentMFCoregionalizationSVGP

        elif emu_type['multi-fid'] and emu_type['linear'] and emu_type['single-bin']:
            # Class for Linear multi-fidelity emulators
            self.emu = gpemu.SingleBinLinearGP
        
        elif emu_type['multi-fid'] and not emu_type['linear'] and emu_type['single-bin']:
            # Class for Non-linear multi-fidelity emulators
            self.emu = gpemu.SingleBinNonLinearGP

        elif not emu_type['multi-fid'] and not emu_type['single-bin']:
            # Class for single-fidelity emulators for all bins
            self.emu = gpemu.SingleBinGP

        else:
            raise NotImplementedError("This type of emulator is not implemented")

        self.X = X
        self.Y = Y
        # If multi-fid = True, then X and Y are lists of arrays
        #if emu_type['multi-fid']:
        self.n_fidelities = len(X)
        self.n_points = []
        self.n_dims = []
        self.n_bins = []
        for n in range(self.n_fidelities):
            self.n_points.append(X[n].shape[0])
            self.n_dims.append(X[n].shape[1])
            self.n_bins.append(Y[n].shape[1])
        # If multi-fid = False, then X and Y are arrays
        #else:
        #    self.n_fidelities = 1
        #    self.n_points = X.shape[0]
        #    self.n_dims = X.shape[1]
        #    self.n_bins = Y.shape[1]

        if rank == 0:
            self.logger.info(f'Fidelities: {self.n_fidelities}, Points: {self.n_points}, Dimensions: {self.n_dims}, Bins: {self.n_bins}')

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger('BaseStatEmu')
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | Rank {rank} | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger

        
    def loo_train_pred(self, savefile):
        """
        Get the leave one out predictions
        """
        mean_pred = np.zeros((self.n_points[-1], self.n_bins[-1]))
        var_pred = np.zeros((self.n_points[-1], self.n_bins[-1]))
        if self.emu_type['multi-fid']:
            for i, s in enumerate(self.labels[-1]):
                if  rank ==0:
                    self.logger.info(f'Leaving out {s}, progress {i}/{len(self.labels[-1])}')
                X_train = [self.X[0]]
                X_train.append(np.delete(self.X[-1], i, axis=0))
                Y_train = [self.Y[0]]
                Y_train.append(np.delete(self.Y[-1], i, axis=0))
                X_test = self.X[-1][i][np.newaxis, :]
                Y_test = self.Y[-1][i]
                #self.logger.info(X_train[0].shape, X_train[1].shape, Y_train[0].shape, Y_train[1].shape, X_test.shape, Y_test.shape)
                model = self.emu(copy.deepcopy(X_train), copy.deepcopy(Y_train), n_fidelities=self.n_fidelities, kernel_list=None)
                model.optimize(n_optimization_restarts=self.n_optimization_restarts)
                mean_pred[i], var_pred[i] = model.predict(X_test)
        else:
            for i, s in enumerate(self.labels[0]):
                self.logger.info(f'Leaving out {s}')
                X_train = np.delete(self.X[0], i, axis=0)
                Y_train = np.delete(self.Y[0], i, axis=0)
                X_test = self.X[0][i][np.newaxis, :]
                Y_test = self.Y[0][i]
                model = self.emu(X_train, Y_train, kernel_list=None, single_bin=self.emu_type['single-bin'])
                model.optimize(n_optimization_restarts=self.n_optimization_restarts)
                mean_pred[i], var_pred[i] = model.predict(X_test)
        if MPI is not None:
            comm.Barrier()
        if rank==0:
            with h5py.File(savefile, 'w') as f:
                self.logger.info(f'Writing on {savefile}')
                f.create_dataset('pred', data=mean_pred)
                f.create_dataset('var_pred', data=var_pred)
                f.create_dataset('bins', data=self.mbins)
                if self.emu_type['multi-fid']:
                    f.create_dataset('truth', data=self.Y[-1])
                    f.create_dataset('X', data=self.X[-1])
                    # Writing a string dataset on h5py is a bit tricky
                    labels = np.array(self.labels[-1], dtype='S')
                    # Define an HDF5-compatible string data type
                    string_dtype = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset('labels', data=labels.astype(string_dtype), dtype=string_dtype)
                else:
                    f.create_dataset('truth', data=self.Y)
                    f.create_dataset('X', data=self.X)
        if MPI is not None:
            comm.Barrier()
    
    def train_pred_all_sims(self, savefile=None):
        """
        Train the model on all simulations and comapre with the truth
        """
        if self.emu_type['mf-svgp']:
            # Add the fidelity indocators
            X_l2_aug = np.hstack([self.X[0], np.zeros((self.X[0].shape[0], 1))])
            X_hf_aug = np.hstack([self.X[-1], np.ones((self.X[-1].shape[0], 1))])
            # Stack the L2 and HF data vertically
            self.X = np.vstack([X_l2_aug, X_hf_aug])
            self.Y = np.vstack([self.Y[0], self.Y[-1]])
            # Base kernel of the MF GP
            kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(self.X.shape[1]-1), variance=1.0)
            kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(self.X.shape[1]-1), variance=1.0)
            self.emu = LatentMFCoregionalizationSVGP(self.X, self.Y, kernel_L, kernel_delta, 
                                                     num_latents=5, num_inducing=100,
                                                     num_outputs=self.n_bins[0])
            self.logger.info(f'shapes passed to LMF : {self.X.shape, self.Y.shape}')
            self.emu.optimize(data=(self.X, self.Y))

        
        elif self.emu_type['multi-fid']:
            model = self.emu(copy.deepcopy(self.X), copy.deepcopy(self.Y), n_fidelities=self.n_fidelities, kernel_list=None)
            model.optimize(n_optimization_restarts=self.n_optimization_restarts)
        else:
            model = self.emu(self.X[0], self.Y[0], kernel_list=None, single_bin=self.emu_type['single-bin'])
            model.optimize(n_optimization_restarts=self.n_optimization_restarts)
            #if self.emu_type['multi-fid']:
        mean_pred, var_pred = model.predict(copy.deepcopy(self.X[-1]))
        #else:
        #    mean_pred, var_pred = model.predict(self.X)
        if MPI is not None:
            comm.Barrier()
        if savefile is not None:
            if rank == 0:
                self.logger.info(f'Writing on {savefile}')
                with h5py.File(savefile, 'w') as f:
                    f.create_dataset('pred', data=mean_pred)
                    f.create_dataset('var_pred', data=var_pred)
                    if self.emu_type['multi-fid']:
                        f.create_dataset('truth', data=self.Y[-1])
                        f.create_dataset('X', data=self.X[-1])
                        labels = np.array(self.labels[-1], dtype='S')
                        # Define an HDF5-compatible string data type
                        string_dtype = h5py.string_dtype(encoding='utf-8')
                        f.create_dataset('labels', data=labels.astype(string_dtype), dtype=string_dtype)
                    else:
                        f.create_dataset('truth', data=self.Y)
                        f.create_dataset('X', data=self.X)
                    f.create_dataset('bins', data=self.mbins)

                    
            if MPI is not None:
                comm.Barrier()
    
    def train(self, save_dir=None):
        """
        Train the model and save this in `save_dir`, furthur instruction in 
        `save` routines of `gal_goku.gpemulator_singlebin`
        """
        if self.emu_type['multi-fid']:
            model = self.emu(copy.deepcopy(self.X), copy.deepcopy(self.Y), n_fidelities=self.n_fidelities, kernel_list=None)
            model.optimize(n_optimization_restarts=self.n_optimization_restarts)
        else:
            raise NotImplementedError
        if save_dir is not None:
            if rank ==0:
                model.save(save_dir=save_dir)
            if MPI is not None:
                comm.Barrier()
        else:
            return model


class Hmf(BaseStatEmu):
    def __init__(self, data_dir, fid=['L2'], logging_level='INFO', no_merge=True, emu_type={'multi-fid':False, 'single-bin':False, 'linear':True, 'wide_and_narrow':True}):
        """
        emu_type : dict
            A dictionary with the emulator types. 
            linear : bool
                If True, use the linear emulator.
            multi-fid : bool
                If True, use the multi-fidelity emulator.
            wide_and_narrow : bool
                If True, use both wide and narrow simulations.
            mf-svgp : bool
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.no_merge = no_merge
        self.emu_type = emu_type
        self.X = []
        self.Y = []
        self.labels = []
        # Train on both Goku-wide and goku-narrow sims
        if emu_type['multi-fid']:
            fids = ['L2', 'HF']
        else:
            fids = ['L2']
        for fd in fids:
            # Goku-wide sims
            hmf = summary_stats.HMF(data_dir=data_dir, fid = fd,  narrow=False, no_merge=no_merge, logging_level=logging_level)
            # Trainthe spline coefficients
            Y, self.mbins = hmf.get_coeffs()
            # For now, get rid of the lastbins with 0 value
            Y_wide = Y[:, :-3]
            X_wide = hmf.get_params_array()
            labels_wide = hmf.get_labels()
            # Only use Goku-wide
            if not emu_type['wide_and_narrow']:
                self.Y.append(Y_wide)
                self.X.append(X_wide)
                self.labels.append(labels_wide)
            # Use both Goku-wide and narrow
            else:
                # Goku-narrow sims
                hmf = summary_stats.HMF(data_dir=data_dir, fid = fd,  narrow=True, no_merge=no_merge, logging_level=logging_level)
                # Trainthe spline coefficients
                Y, self.mbins = hmf.get_coeffs()
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y[:, :-3]), axis=0))
                self.X.append(np.concatenate((X_wide, hmf.get_params_array()), axis=0))
                self.labels.append(np.concatenate((labels_wide, hmf.get_labels()), axis=0))
        if rank==0:
            self.logger.info(f'X: {len(self.X), np.array(self.X[0]).shape}, Y: {len(self.Y), np.array(self.Y[0]).shape}')

        super().__init__(X=self.X, Y=self.Y, logging_level=logging_level, emu_type=emu_type, n_optimization_restarts=5)
    

class BaseMFCoregEmu():
    """
    Emulator for the Halo Mass Function using the native bins
    This does the full dimensionality reduction of the output space using
    `LatentMFCoregionalizationSVGP` which allows each output to have a different
    observational (simualtion quality) uncertainty.
    """
    def __init__(self, DataLoader, data_dir, z, num_latents, num_inducing, emu_type={'wide_and_narrow':True}, norm_type='subtract_mean', noise_floor=0.0, logging_level='INFO'):
        """
        Parameters
        ----------
        data_dir : str
            The directory where the data is stored.
        mass_pair : tuple
            The mass pair for which the correlation function is to be emulated.
        interp : str
            If 'spline', interpolate the nan values in the correlation function
            using a spline. Else, remove the sims with even a single nan values.
        emu_type : dict
            wide_and_narrow : bool
                If True, use both wide and narrow simulations.
        norm_type : str
            The type of normalization to be applied to the data.
            'subtract_mean' : subtract the mean of the LF and let
            the MF GP match the HF mean.
            'std_gaussian' : normalize each bin to have mean 0 and std 1. Mean
            and std are calculated based on the LF sims. Both LF and HF sims
            are normalized using the same mean and std.
        logging_level : str
            The logging level. 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.num_latents = num_latents
        self.num_inducing = num_inducing
        self.norm_type = norm_type
        self.noise_floor = noise_floor
        # Load the data
        self.X = []
        self.Y = []
        self.Y_err = []
        self.labels = []
        self.wide_array = np.array([])
        # Keeping the id for good HF sims
        self.good_sim_ids = []
        # Fix a few features of the emulator
        #emu_type.update({'multi-fid':True, 'single-bin':False, 'linear':True, 'mf-svgp':True})
        self.emu_type = emu_type
        fids = ['L2', 'HF']
        for fd in fids:
            # Goku-wide sims
            data_loader = DataLoader(data_dir=data_dir, fid =fd, z=z, narrow=False, no_merge=True, logging_level=logging_level)
            # Load xi((m1, m2), r) for wide
            self.mbins, Y_wide, err_wide, X_wide, labels_wide = data_loader.get_data(noise_floor=noise_floor)
            self.wide_array= np.append(self.wide_array, np.ones(Y_wide.shape[0]))
            self.logger.debug(f'Y_wide: {Y_wide.shape}')
            # Only use Goku-wide
            if not emu_type['wide_and_narrow']:
                self.Y.append(Y_wide)
                self.X.append(X_wide)
                self.Y_err.append(err_wide)
                self.labels.append(labels_wide)
            # Use both Goku-wide and narrow
            else:
                # Goku-narrow sims
                data_loader = DataLoader(data_dir=data_dir, fid = fd, z=z, narrow=True, no_merge=True, logging_level=logging_level)
                # Load xi((m1, m2), r) for wide
                _, Y_narrow, err_narrow, X_narrow, labels_narrow = data_loader.get_data(noise_floor=noise_floor)
                self.wide_array= np.append(self.wide_array, np.zeros(Y_narrow.shape[0]))
                self.logger.debug(f'Y_narrow: {Y_narrow.shape}')
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y_narrow), axis=0))
                self.X.append(np.concatenate((X_wide, X_narrow), axis=0))
                self.Y_err.append(np.concatenate((err_wide, err_narrow), axis=0))
                self.labels.append(np.concatenate((labels_wide, labels_narrow), axis=0))
        # X is normalized between 0 and 1, but for Y only HF fideliy is not normalized
        # the MF GP will match the HF mean
        if norm_type == 'subtract_mean':
            self.logger.info('Normalizing X between 0 and 1, subtracting the mean of LF from LF Y')
            self.X, self.Y, self.X_min, self.X_max, self.lf_mean_func = self.normalize(self.X, self.Y)
        elif norm_type == 'std_gaussian':
            self.logger.info('Normalizing each bin to have mean 0 and std 1')
            self.X, self.Y, self.Y_err, self.X_min, self.X_max, self.mean_Y, self.std_Y = self.normalize_std_gaussian(self.X, self.Y, self.Y_err)
        self.output_dim = self.Y[0].shape[1]
        # Concatenate the errors to Y, so self.Y is a list of fidelities: [array([Y_wide ... err_wide]), array([Y_narrow ... err_narrow])]

        #self.Y[0] = np.concatenate((self.Y[0][:, :], Y_err[0][:,:]), axis=1)
        #self.Y[1] = np.concatenate((self.Y[1][:,:], Y_err[1][:,:]), axis=1)

        self.Y[0] = self.Y[0].astype(np.float64)
        self.Y[1] = self.Y[1].astype(np.float64)
        self.Y_err[0] = self.Y_err[0].astype(np.float64)
        self.Y_err[1] = self.Y_err[1].astype(np.float64)
        self.X[0] = self.X[0].astype(np.float64)
        self.X[1] = self.X[1].astype(np.float64)

        assert not np.isnan(self.X[0]).any(), f'X[0] has nans {np.where(np.isnan(self.X[0]))}'
        assert not np.isnan(self.X[1]).any(), f'X[1] has nans {np.where(np.isnan(self.X[1]))}'
        assert not np.isnan(self.Y[0]).any(), f'Y[0] has nans {np.where(np.isnan(self.Y[0]))}'
        assert not np.isnan(self.Y[1]).any(), f'Y[1] has nans {np.where(np.isnan(self.Y[1]))}'
        self.logger.debug(f'X: ({np.array(self.X[0]).shape}, {np.array(self.X[1]).shape}, Y: ({np.array(self.Y[0]).shape}, {np.array(self.Y[1]).shape}, Y_err: ({np.array(self.Y_err[0]).shape}, {np.array(self.Y_err[1]).shape})')
        self.logger.info(f'norm_type {norm_type}')
        self.logger.info(f'noise_floor {noise_floor}')

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger('HMF-MFCoregEmu')
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | Rank {rank} | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def normalize(self, X, Y):
        """
        Normalize all input, X, such it is between 0 and 1
        Subtract the output, Y, by it's mean, only for LF
        and leave the HF uncrouched

        Returns:
        --------
        X_normalized: normalized input data between 0 and 1
        X_min: minimum value of the input data
        X_max: maximum value of the input data
        mean_func: the mean of the output to be used as the mean function
        in the GP model
        """
        X_min, X_max = np.min(X[0], axis=0), np.max(X[0], axis=0)
        X_normalized = []
        for i in range(len(X)):
            X_normalized.append((X[i]-X_min)/(X_max-X_min))
        Y_normalized = []
        # The zeros row is the LF
        lf_mean_func =  np.mean(Y[0], axis=0)
        Y_normalized.append(Y[0] -lf_mean_func)
        # Don't subtract the mean for the HF, The MF GP will match
        # the HF mean
        Y_normalized.append(Y[1])

        return X_normalized, Y_normalized, X_min, X_max, lf_mean_func
    
    def normalize_std_gaussian(self, X, Y, Y_err):
        """
        Normalize all input, X, such it is between 0 and 1
        Normalize Y at each bin to have mean 0 and std 1 -- should
        help with forcing the GP to spend similar focus on all bins

        """
        X_min, X_max = np.min(X[0], axis=0), np.max(X[0], axis=0)
        X_normalized = []
        for i in range(len(X)):
            X_normalized.append((X[i]-X_min)/(X_max-X_min))
        
        Y_normalized = []
        Y_err_normalized = []
        # We have more LF sims, so normalize based on LF
        mean = np.mean(Y[0], axis=0)
        std = np.std(Y[0], axis=0)
        for i in range(len(Y)):
            Y_normalized.append((Y[i] - mean) / std)
            Y_err_normalized.append(Y_err[i] / std)

        return X_normalized, Y_normalized, Y_err_normalized, X_min, X_max, mean, std

    def train(self, ind_train=None, ind_test=None, model_file='Xi_Native_emu_mapirs2.pkl', opt_params={}, force_train=True, train_subdir = 'train', composite_kernel=None, w_type='diagonal', loss_type='gaussian'):
        """
        Train the model and save this in `model_file`
        Parameters
        ----------
        model_file : str
            The file to save the Emulator. Two files
            will be saved, one with the model and the other
            with the loss history.
        composite_kernel : dict
            If not None, add the specified kernels. e.g {'kernel_L':['matern32', 'matern52'], 'kernel_delta':['rbf']}
        """
        if ind_train is None:
            ind_train = np.arange(self.X[1].shape[0])
        # Add the fidelity indocators, 0 for L2 and 1 for HF
        X_l2_aug = np.hstack([self.X[0], np.zeros((self.X[0].shape[0], 1), dtype=np.float64)])
        X_hf_aug = np.hstack([self.X[1][ind_train], np.ones((ind_train.size, 1), dtype=np.float64)])
        # Stack the L2 and HF data vertically
        X_train = np.vstack([X_l2_aug, X_hf_aug])
        Y_train = np.vstack([self.Y[0], self.Y[1][ind_train]])
        Y_err = np.vstack([self.Y_err[0], self.Y_err[1][ind_train]])
        Y_train = np.concatenate((Y_train, Y_err), axis=1) # shape becomes [N, 2*P]
        self.logger.debug(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')

        X_train = X_train.astype(np.float64)
        Y_train = Y_train.astype(np.float64)

        # Base kernel of the MF GP
        if composite_kernel is None:
            kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0))
            kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0))
        elif composite_kernel == ['matern32', 'matern52', 'matern32', 'matern52']:
            kernel_L = (gpflow.kernels.Matern32(lengthscales=0.5*np.ones(X_train.shape[1]-1, dtype=np.float64), variance=np.float64(0.3)) + \
                          gpflow.kernels.Matern52(lengthscales=np.ones(X_train.shape[1]-1, dtype=np.float64), variance=np.float64(1.0)))
            kernel_delta = (gpflow.kernels.Matern32(lengthscales=0.5*np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(0.3)) + \
                            gpflow.kernels.Matern52(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0)))
        elif composite_kernel == ['matern32', 'matern52', 'rbf']:
            kernel_L = (gpflow.kernels.Matern32(lengthscales=0.5*np.ones(X_train.shape[1]-1, dtype=np.float64), variance=np.float64(0.3)) + \
                          gpflow.kernels.Matern52(lengthscales=np.ones(X_train.shape[1]-1, dtype=np.float64), variance=np.float64(1.0)))
            kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0))
        elif composite_kernel == ['rbf','matern52', 'rbf']:
            kernel_L = (gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1, dtype=np.float64), variance=np.float64(1.0)) + \
                          gpflow.kernels.Matern52(lengthscales=np.ones(X_train.shape[1]-1, dtype=np.float64), variance=np.float64(1.0)))
            kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0))
        else:
            raise ValueError(f"Unknown composite_kernel: {composite_kernel}")
        
        #kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0))
        P = int(Y_train.shape[1]/2) # Number of outputs
        # We only pass the actual Y_train, not the errors to the
        # class constructor
        self.emu = LatentMFCoregionalizationSVGP(
            X_train, Y_train, kernel_L, kernel_delta,
            num_latents=self.num_latents, num_inducing=self.num_inducing,
            num_outputs=self.output_dim, heterosed=True, w_type=w_type, loss_type=loss_type)

        model_file = op.join(self.data_dir, train_subdir, model_file)
        #self.logger.info(f'Will save to {model_file}')
        existing_model_files = glob(model_file.replace('.pkl', '*.pkl'))
        if len(existing_model_files) > 0:
            # extract epoch number from the filenames
            try: 
                if force_train:
                    epochs = [int(op.basename(f).split('_')[-1].replace('.pkl', '')) for f in existing_model_files if '_' in op.basename(f)]
                    latest_epoch = max(epochs)
                    model_file = model_file.replace('.pkl', f'_{latest_epoch}.pkl')
                    self.logger.info(f'Found {len(existing_model_files)} existing model files, will load the latest one: {model_file}')
                else:
                    self.logger.info(f'will NOT train, loading the model from {model_file}')
            except ValueError:
                self.logger.info(f'loading from the only model file found {model_file}')
            with open(model_file, "rb") as f:
                params = pickle.load(f)
                # TODO: Save the model already in float64:
                # Convert all parameters to float64 type
                # This won't be necessary if the saved model is already in float64
                for key, value in params.items():
                    if isinstance(value, dict):
                        for inner_key, inner_value in value.items():
                            if isinstance(inner_value, np.ndarray):
                                params[key][inner_key] = inner_value.astype(np.float64)
                            elif isinstance(inner_value, (int, float)):
                                params[key][inner_key] = np.float64(inner_value)
                    elif isinstance(value, np.ndarray):
                        params[key] = value.astype(np.float64)
                    elif isinstance(value, (int, float)):
                        params[key] = np.float64(value)
                gpflow.utilities.multiple_assign(self.emu, params)
            # load the loss_history:
            try:
                with open(f'{model_file}.attrs', 'rb') as f:
                    attrs = pickle.load(f)
                    # Reload the loss history, so it will be appended
                    # during the new training
                    self.emu.loss_history = attrs['loss_history']
                    self.emu.kl_history = attrs['kl_history']
                    current_iters = len(self.emu.loss_history)
            except:
                current_iters = None
                self.logger.warning(f'No loss history found for {model_file}.attrs, but model exists')
        else:

            current_iters = 0
            model_file = model_file.replace('.pkl', f'_{current_iters}.pkl')
        # Log the model specifications
        self.logger.info(f'Built the model with')
        self.logger.info(f'#num_latents {self.num_latents}')
        self.logger.info(f'output_dim {self.output_dim}')
        self.logger.info(f'num_inducing {self.num_inducing}')
        if loss_type == 'gaussian':
            self.logger.info(f'variance dim {self.emu.likelihood.variance.numpy().shape}')
        self.logger.info(f'composite_kernel {composite_kernel}')
        self.logger.info(f'w_type {w_type}')
        self.logger.info(f'trained epochs {current_iters}')


        if len(list(opt_params)) == 0:
            max_iters = 4_000
            initial_lr = 5e-3
        else:
            iter_save = opt_params['iter_save']
            max_iters = opt_params['max_iters']
            initial_lr = opt_params['initial_lr']
        # It won't train unless instructed
        if force_train:
            self.logger.debug(f'Training. shapes passed to LMF : {X_train.shape, Y_train.shape}')
            if len(self.emu.loss_history) >= max_iters:
                self.logger.info(f'{model_file} already trained for {max_iters} iterations')
                return
            # Do the training in batches of iter_save, so we defenitely save
            # the model every iter_save iterations
            iter_stop_point = np.append(np.arange(current_iters, max_iters, iter_save), max_iters) if max_iters % iter_save != 0 else np.arange(current_iters, max_iters + 1, iter_save)
            iter_stop_point = iter_stop_point[1:]
            for it_stp in iter_stop_point:
                current_iters = len(self.emu.loss_history)
                self.logger.info(f'Continue optimization from {current_iters} to {it_stp}')
                # The decaying learning rate
                start_lr = tf.keras.optimizers.schedules.CosineDecay(initial_lr, max_iters)(current_iters)
                # Both data and uncertainty are passed to the optimizer
                self.emu.optimize(data=(X_train, Y_train), max_iters=it_stp, initial_lr=start_lr, unfix_noise_after=500)
                # We need the udpated current iters to save the model
                current_iters = len(self.emu.loss_history)
                model_file = op.join(op.dirname(model_file), op.basename(model_file).rsplit('_', 1)[0]+f'_{int(current_iters)}.pkl')
                self.emu.save_model(model_file)
                # Save loss_history, ind_train and emu_type
                with open(f'{model_file}.attrs', 'wb') as f:
                    self.logger.debug(f'Writing the model on {model_file}')
                    self.model_attrs = {}
                    self.model_attrs['loss_history'] = self.emu.loss_history
                    self.model_attrs['kl_history'] = self.emu.kl_history
                    #self.model_attrs['ind_train'] = ind_train
                    self.model_attrs['emu_type'] = self.emu_type
                    pickle.dump(self.model_attrs, f)
            self.logger.info(f'done with optimization {max_iters}')

    def predict(self, ind_test, model_file, train_subdir = 'train', composite_kernel=None):
        """
        Posteroir prediction of the emulator
        Parameters
        ----------
        ind_train : array
            The indices of the HF sims to be used for training
        ind_test : array
            The indices of the HF sims to be used for testing
        model_file : str
            The file to save the Emulator. If the file exists, 
            the model is loaded from the file.
        Returns
        -------
        mean_pred, var_pred : (array, array)
            The mean and variance of the predicted 
            log10(xi(r)) for the test sims.
        """
        try:
            with open(op.join(self.data_dir, train_subdir, f'{model_file}.attrs'), 'rb') as f:
                self.model_attrs = pickle.load(f)
        except:
            self.logger.warning(f'No model attributes found for {op.join(self.data_dir, train_subdir, f"{model_file}.attrs")}')
            self.model_attrs = {}
        #ind_train = self.model_attrs['ind_train']
        #self.emu_type = self.model_attrs['emu_type']
        #self.train(ind_train, model_file, force_train=False, train_subdir=train_subdir)
        self.train(model_file=model_file, force_train=False, train_subdir=train_subdir, composite_kernel=composite_kernel)
        
        # Add the fidelity indocators
        X_test = np.hstack([self.X[1][ind_test], np.ones((ind_test.size, 1))]).astype(np.float64)
        mean_pred, var_pred = self.emu.predict_f(X_test)

        if self.norm_type == 'std_gaussian':
            # Add back the mean subtracted during normalization
            mean_pred *= self.std_Y
            mean_pred += self.mean_Y
            var_pred *= self.std_Y**2
        
        return mean_pred, var_pred

class HmfNativeBins(BaseMFCoregEmu):
    """
    Emulator for the Halo Mass Function using the native bins
    This does the full dimensionality reduction of the output space using
    `LatentMFCoregionalizationSVGP` which allows each output to have a different
    observational (simualtion quality) uncertainty.
    """

    def __init__(self, data_dir, z, num_latents, num_inducing, emu_type={ 'wide_and_narrow': True }, norm_type='subtract_mean', noise_floor=0.0, logging_level='INFO'):
        
        DataLoader = summary_stats.HMF
        super().__init__(DataLoader, data_dir, z, num_latents, num_inducing, emu_type, norm_type=norm_type, noise_floor=noise_floor, logging_level=logging_level)



class XiNativeBinsFullDimReduc():
    """
    Emulator for the Correlation Function, xi(r, n1, n2) using the native bins
    This does the full dimensionality reduction of the output space using
    `LatentMFCoregionalizationSVGP` which allows each output to have a different
    observational (simualtion quality) uncertainty.
    """
    def __init__(self, data_dir, num_latents, num_inducing, 
                 use_rho=True, emu_type={'wide_and_narrow':True}, 
                 logging_level='INFO'):
        """
        Parameters
        ----------
        data_dir : str
            The directory where the data is stored.
        mass_pair : tuple
            The mass pair for which the correlation function is to be emulated.
        interp : str
            If 'spline', interpolate the nan values in the correlation function
            using a spline. Else, remove the sims with even a single nan values.
        emu_type : dict
            wide_and_narrow : bool
                If True, use both wide and narrow simulations.
        remove_sims : list
            A list of simulation indices to remove from the training/test set.
        logging_level : str
            The logging level. 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.num_latents = num_latents
        self.num_inducing = num_inducing
        self.use_rho = use_rho
        # Laod the data
        self.X = []
        self.Y = []
        self.Y_err = []
        self.labels = []
        self.wide_array = np.array([])
        # Keeping the id for good HF sims
        self.good_sim_ids = []
        # Fix a few features of the emulator
        emu_type.update({'multi-fid':True, 'single-bin':False, 'linear':True, 'mf-svgp':True})
        self.emu_type = emu_type
        fids = ['L2', 'HF']
        for fd in fids:
            # Goku-wide sims
            xi = summary_stats.Xi(data_dir=data_dir, fid = fd,  narrow=False, MPI=None, logging_level=logging_level)
            # Load xi((m1, m2), r) for wide
            self.mbins, Y_wide, err_wide, X_wide, labels_wide = xi.get_wt_err(rcut=(0.2, 61))
            self.wide_array= np.append(self.wide_array, np.ones(Y_wide.shape[0]))
            self.logger.debug(f'Y_wide: {Y_wide.shape}')
            # Only use Goku-wide
            if not emu_type['wide_and_narrow']:
                self.Y.append(Y_wide)
                self.X.append(X_wide)
                self.Y_err.append(err_wide)
                #self.labels.append(labels_wide)
            # Use both Goku-wide and narrow
            else:
                # Goku-narrow sims
                xi = summary_stats.Xi(data_dir=data_dir, fid = fd,  narrow=True, MPI=None, logging_level=logging_level)
                # Load xi((m1, m2), r) for wide
                _, Y_narrow, err_narrow, X_narrow, labels_narrow = xi.get_wt_err(rcut=(0.2, 61))
                self.wide_array= np.append(self.wide_array, np.zeros(Y_narrow.shape[0]))
                self.logger.debug(f'Y_narrow: {Y_narrow.shape}')
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y_narrow), axis=0))
                self.X.append(np.concatenate((X_wide, X_narrow), axis=0))
                self.Y_err.append(np.concatenate((err_wide, err_narrow), axis=0))
                self.labels.append(np.concatenate((labels_wide, labels_narrow), axis=0))
        # X is normalized between 0 and 1, but for Y only HF fideliy is not normalized
        # the MF GP will match the HF mean
        self.X, self.Y, self.X_min, self.X_max = self.normalize(self.X, self.Y)
        self.output_dim = self.Y[0].shape[1]
        # Concatenate the errors to Y, so self.Y is a list of fidelities: [array([Y_wide ... err_wide]), array([Y_narrow ... err_narrow])]

        #self.Y[0] = np.concatenate((self.Y[0][:, :], Y_err[0][:,:]), axis=1)
        #self.Y[1] = np.concatenate((self.Y[1][:,:], Y_err[1][:,:]), axis=1)

        self.Y[0] = self.Y[0].astype(np.float64)
        self.Y[1] = self.Y[1].astype(np.float64)
        self.Y_err[0] = self.Y_err[0].astype(np.float64)
        self.Y_err[1] = self.Y_err[1].astype(np.float64)
        self.X[0] = self.X[0].astype(np.float64)
        self.X[1] = self.X[1].astype(np.float64)

        assert not np.isnan(self.X[0]).any(), f'X[0] has nans {np.where(np.isnan(self.X[0]))}'
        assert not np.isnan(self.X[1]).any(), f'X[1] has nans {np.where(np.isnan(self.X[1]))}'
        assert not np.isnan(self.Y[0]).any(), f'Y[0] has nans {np.where(np.isnan(self.Y[0]))}'
        assert not np.isnan(self.Y[1]).any(), f'Y[1] has nans {np.where(np.isnan(self.Y[1]))}'

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger('XiNativeBinsFullDimReduc')
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | Rank {rank} | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    def normalize(self, X, Y):
        """
        Normalize all input, X, such it is between 0 and 1
        Subtract the output, Y, by it's mean, only for LF
        and leave the HF uncrouched

        Returns:
        --------
        X_normalized: normalized input data between 0 and 1
        X_min: minimum value of the input data
        X_max: maximum value of the input data
        mean_func: the mean of the output to be used as the mean function
        in the GP model
        """
        X_min, X_max = np.min(X[0], axis=0), np.max(X[0], axis=0)
        X_normalized = []
        for i in range(len(X)):
            X_normalized.append((X[i]-X_min)/(X_max-X_min))
        Y_normalized = []
        # The zeros row is the LF
        lf_median_func =  np.nanmedian(Y[0], axis=0)
        Y_normalized.append(Y[0] - lf_median_func)
        Y_normalized.append(Y[1])

        return X_normalized, Y_normalized, X_min, X_max
    
    def train(self, ind_train=None, model_file='Xi_Native_emu_mapirs2.pkl', opt_params={}, force_train=True, train_subdir = 'train'):
        """
        Train the model and save this in `model_file`
        Parameters
        ----------
        model_file : str
            The file to save the Emulator. Two files
            will be saved, one with the model and the other
            with the loss history.
        """
        if ind_train is None:
            ind_train = np.arange(self.X[1].shape[0])

        # Also subtract median from HF sims, I had noticed the f_HF - f_LF
        # is < 3% for all 36 LF-HF pairs, it gets slightly larger 
        # closer to r =60 CMpc/h, but still similar for all smulations

        # We subtract the median of the HF sims for training if use_rho is False
        if not self.use_rho:
            self.hf_median_func = np.nanmedian(self.Y[1][ind_train], axis=0)

        # Add the fidelity indocators, 0 for L2 and 1 for HF
        X_l2_aug = np.hstack([self.X[0], np.zeros((self.X[0].shape[0], 1), dtype=np.float64)])
        X_hf_aug = np.hstack([self.X[1][ind_train], np.ones((ind_train.size, 1), dtype=np.float64)])
        # Stack the L2 and HF data vertically
        X_train = np.vstack([X_l2_aug, X_hf_aug])
        if self.use_rho:
            Y_train = np.vstack([self.Y[0], self.Y[1][ind_train]])
        else:
            # We subtract the median of the HF sims for training if use_rho is False
            Y_train = np.vstack([self.Y[0], self.Y[1][ind_train] - self.hf_median_func])
        Y_err = np.vstack([self.Y_err[0], self.Y_err[1][ind_train]])
        Y_train = np.concatenate((Y_train, Y_err), axis=1) # shape becomes [N, 2*P]
        self.logger.debug(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')

        X_train = X_train.astype(np.float64)
        Y_train = Y_train.astype(np.float64)

        # Base kernel of the MF GP
        kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0))
        kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float64), variance=np.float64(1.0))
        P = int(Y_train.shape[1]/2) # Number of outputs
        # We only pass the actual Y_train, not the errors to the
        # class constructor
        self.emu = LatentMFCoregionalizationSVGP(
            X_train, Y_train[:,0:P], kernel_L, kernel_delta,
            num_latents=self.num_latents, num_inducing=self.num_inducing,
            num_outputs=self.output_dim, heterosed=True, use_rho=self.use_rho)
        
        model_file = op.join(self.data_dir, train_subdir, model_file)
        if op.exists(model_file):
            self.logger.info(f'Loading model from {model_file}')
            with open(model_file, "rb") as f:
                params = pickle.load(f)
                # TODO: Save the model already in float64:
                # Convert all parameters to float64 type
                # This won't be necessary if the saved model is already in float64
                for key, value in params.items():
                    if isinstance(value, dict):
                        for inner_key, inner_value in value.items():
                            if isinstance(inner_value, np.ndarray):
                                params[key][inner_key] = inner_value.astype(np.float64)
                            elif isinstance(inner_value, (int, float)):
                                params[key][inner_key] = np.float64(inner_value)
                    elif isinstance(value, np.ndarray):
                        params[key] = value.astype(np.float64)
                    elif isinstance(value, (int, float)):
                        params[key] = np.float64(value)
                gpflow.utilities.multiple_assign(self.emu, params)
            # load the loss_history:
            try:
                with open(f'{model_file}.attrs', 'rb') as f:
                    attrs = pickle.load(f)
                    # Reload the loss history, so it will be appended
                    # during the new training
                    self.emu.loss_history = attrs['loss_history']
                    current_iters = len(self.emu.loss_history)
            except:
                current_iters = None
                self.logger.warning(f'No loss history found for {model_file}.attrs, but model exists')
        else:

            current_iters = 0
        # Log the model specifications
        self.logger.info(f'Built the model with')
        self.logger.info(f'#num_latents {self.num_latents}')
        self.logger.info(f'output_dim {self.output_dim}')
        self.logger.info(f'num_inducing {self.num_inducing}')
        self.logger.info(f'varaince dim {self.emu.likelihood.variance.numpy().shape}')
        self.logger.info(f'trained epochs {current_iters}')


        if len(list(opt_params)) == 0:
            max_iters = 4_000
            initial_lr = 5e-3
            kl_multiplier=1.0
        else:
            iter_save = opt_params.get('iter_save', 4000)
            max_iters = opt_params['max_iters']
            initial_lr = opt_params['initial_lr']
            kl_multiplier= opt_params['kl_multiplier']
        self.logger.info(f'opt_params: {opt_params}')
        # It won't train unless instructed
        if force_train:
            self.logger.debug(f'Training. shapes passed to LMF : {X_train.shape, Y_train.shape}')
            if len(self.emu.loss_history) >= max_iters:
                self.logger.info(f'{model_file} already trained for {max_iters} iterations')
                return
            # Do the training in batches of iter_save, so we defenitely save
            # the model every iter_save iterations
            iter_stop_point = np.append(np.arange(current_iters, max_iters, iter_save), max_iters) if max_iters % iter_save != 0 else np.arange(current_iters, max_iters + 1, iter_save)
            iter_stop_point = iter_stop_point[1:]
            for it_stp in iter_stop_point:
                current_iters = len(self.emu.loss_history)
                self.logger.info(f'Continue optimization from {current_iters} to {it_stp}')
                # The decaying learning rate
                start_lr = tf.keras.optimizers.schedules.CosineDecay(initial_lr, max_iters)(current_iters)
                # Both data and uncertainty are passed to the optimizer
                self.emu.optimize(data=(X_train, Y_train), max_iters=it_stp, 
                                  initial_lr=start_lr, unfix_noise_after=500,
                                  kl_multiplier=kl_multiplier)
                self.emu.save_model(model_file)
                # Save loss_history, ind_train and emu_type
                with open(f'{model_file}.attrs', 'wb') as f:
                    self.logger.debug(f'Writing the model on {model_file}')
                    self.model_attrs = {}
                    self.model_attrs['loss_history'] = self.emu.loss_history
                    self.model_attrs['kl_history'] = self.emu.kl_history
                    #self.model_attrs['ind_train'] = ind_train
                    self.model_attrs['emu_type'] = self.emu_type
                    pickle.dump(self.model_attrs, f)
            self.logger.info(f'done with optimization {max_iters}')

    def predict(self, ind_test, model_file, train_subdir = 'train'):
        """
        Posteroir prediction of the emulator
        Parameters
        ----------
        ind_train : array
            The indices of the HF sims to be used for training
        ind_test : array
            The indices of the HF sims to be used for testing
        model_file : str
            The file to save the Emulator. If the file exists, 
            the model is loaded from the file.
        Returns
        -------
        mean_pred, var_pred : (array, array)
            The mean and variance of the predicted 
            log10(xi(r)) for the test sims.
        """
        # Get the median function for the HF sims used for training
        if not hasattr(self, 'hf_median_func'):
            mask = np.ones(self.Y[1].shape[0], dtype=bool)
            mask[ind_test] = False
            self.hf_median_func = np.nanmedian(self.Y[1][mask], axis=0)
        try:
            with open(op.join(self.data_dir, train_subdir, f'{model_file}.attrs'), 'rb') as f:
                self.model_attrs = pickle.load(f)
        except:
            self.logger.warning(f'No model attributes found for {model_file}.attrs')
            self.model_attrs = {}
        #ind_train = self.model_attrs['ind_train']
        #self.emu_type = self.model_attrs['emu_type']
        #self.train(ind_train, model_file, force_train=False, train_subdir=train_subdir)
        self.train(model_file=model_file, force_train=False, train_subdir=train_subdir)
        
        # Add the fidelity indocators
        X_test = np.hstack([self.X[1][ind_test], np.ones((ind_test.size, 1))]).astype(np.float64)
        mean_pred, var_pred = self.emu.predict_f(X_test)
        if not self.use_rho:
            mean_pred += self.hf_median_func
        
        return mean_pred, var_pred