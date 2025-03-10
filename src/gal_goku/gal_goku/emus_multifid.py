"""
To evaluate all the emulators for the Halo Mass Function.
Evaluating multifid emulator, HF and L2 pairs. Using Ming-Feng's thin wrappers of Emukit.
"""

import logging
import h5py
import numpy as np
from . import summary_stats
#from . import single_fid
from . import gpemulator_singlebin as gpemu
from mfgpflow.linear_svgp import LatentMFCoregionalizationSVGP
import sys
import copy

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
        if self.emu_type['multi-fid']:
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
    

class HmfNativeBins(BaseStatEmu):
    """
    A class using the native bins of the Halo Mass Function and using `LinearMFCoregionalizationSVGP`
    to do the dimensionality reduction of the output space and train the emulator. 
    """
    def __init__(self, data_dir, no_merge=True,  emu_type={'wide_and_narrow':True}, logging_level='INFO'):
        """
        Parameters
        ----------
        data_dir : str
            The directory where the data is stored.
        no_merge : bool
            If True, do not merge the last bins of the Halo Mass Function with
            lower counts than 20 halos.
        emu_type : dict
            wide_and_narrow : bool
                If True, use both wide and narrow simulations.
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.no_merge = no_merge
        self.X = []
        self.Y = []
        self.labels = []
        # Fix a few features of the emulator
        emu_type.update({'multi-fid':True, 'single-bin':False, 'linear':True, 'mf-svgp':True})
        fids = ['L2', 'HF']
        for fd in fids:
            # Goku-wide sims
            hmf = summary_stats.HMF(data_dir=data_dir, fid = fd,  narrow=False, no_merge=no_merge, logging_level=logging_level)
            # Train on the hmf values on native bins
            Y_wide, self.mbins = hmf.load()
            print(f'Y_wide: {Y_wide.shape}')
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
                # train on the hmf values on native bins
                Y, self.mbins = hmf.load()
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y), axis=0))
                self.X.append(np.concatenate((X_wide, hmf.get_params_array()), axis=0))
                self.labels.append(np.concatenate((labels_wide, hmf.get_labels()), axis=0))
        if rank==0:
            self.logger.info(f'X: {len(self.X), np.array(self.X[0]).shape}, Y: {len(self.Y), np.array(self.Y[0]).shape}')

        self.X = np.vstack(self.X)
        X_normed, X_min, X_max, Y_normed = self.normalize(self.X, self.Y)


        super().__init__(X=self.X, Y=self.Y, logging_level=logging_level, emu_type=emu_type, n_optimization_restarts=5)
    
        
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
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
        X_normalized = (X-X_min)/(X_max-X_min)
        Y_normalized = []
        # The zeros row is the LF
        # Since each sim has different number of bins, we need iterate 
        Y_normalized.append(Y[0,:] - np.mean(Y[0,:], axis=0))
        # Don't subtract the mean for the HF, The MF GP will match
        # the HF mean
        Y_normalized.append(Y[1,:])

        return X_normalized, X_min, X_max, Y_normalized