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


class BaseStatEmu():

    def __init__(self, X, Y, logging_level='info', emu_type={'multi-fid':False, 'single-bin':False, 'linear':True}):
        """The base emu to be inherited by single fidelity emulators built on any summary statistics
        This is the interface for all classses above.
        :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
        :param Y_train:  (n_fidelities, n_points, n_nins) list of matter power spectra.
        """
        if emu_type['multi-fid'] and emu_type['linear'] and emu_type['single-bin']:
            # Class for multi-fidelity emulators
            self.emu = gpemu.SingleBinLinearGP
        self.logger = self.configure_logging(logging_level)
        self.X = X
        self.Y = Y
        self.n_fidelities = len(X)
        self.n_points = []
        self.n_dims = []
        self.n_bins = []
        for n in range(self.n_fidelities):
            self.n_points.append(X[n].shape[0])
            self.n_dims.append(X[n].shape[1])
            self.n_bins.append(Y[n].shape[1])
        self.logger.info(f'Fidelities: {self.n_fidelities}, Points: {self.n_points}, Dimensions: {self.n_dims}, Bins: {self.n_bins}')


    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('BaseStatEmu')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    def loo_train_pred(self, savefile):
        """
        Get the leave one out predictions
        """
        mean_pred = np.zeros((self.n_points[-1], self.n_dims[-1], self.n_bins[-1]))
        var_pred = np.zeros((self.n_points[-1], self.n_bins[-1]))
        for i, s in enumerate(self.labels[-1]):
            self.logger.info(f'Leaving out {s}')
            X_train = [self.X[0]]
            X_train.append(np.delete(self.X[-1], i, axis=0))
            Y_train = [self.Y[0]]
            Y_train.append(np.delete(self.Y[-1], i, axis=0))
            X_test = self.X[-1][i][np.newaxis, :]
            Y_test = self.Y[-1][i]
            #self.logger.info(X_train[0].shape, X_train[1].shape, Y_train[0].shape, Y_train[1].shape, X_test.shape, Y_test.shape)
            model = self.emu(X_train, Y_train, n_fidelities=self.n_fidelities, kernel_list=None)
            model.optimize(n_optimization_restarts=10)
            mean_pred[i], var_pred[i] = model.predict(X_test)
        
        with h5py.File(savefile, 'w') as f:
            f.create_dataset('pred', data=mean_pred)
            f.create_dataset('var_pred', data=var_pred)
            f.create_dataset('truth', data=self.Y)
            f.create_dataset('X', data=self.X)
            f.create_dataset('bins', data=self.mbins)
            f.create_dataset('labels', data=self.labels)
    
    def train_pred_all_sims(self, savefile=None):
        """
        Train the model on all simulations and comapre with the truth
        """
        model = self.emu(self.X, self.Y, n_fidelities=self.n_fidelities, kernel_list=None)
        model.optimize(n_optimization_restarts=10)
        mean_pred, var_pred = model.predict(self.X[-1])
        if savefile is not None:
            with h5py.File(savefile, 'w') as f:
                f.create_dataset('pred', data=mean_pred)
                f.create_dataset('var_pred', data=var_pred)
                f.create_dataset('truth', data=self.Y[-1])
                f.create_dataset('X', data=self.X[-1])
                f.create_dataset('bins', data=self.mbins)
                f.create_dataset('labels', data=self.labels)
        
    
    def predict(self, X_test):
        """
        Predict the mean and variance of the emulator
        """
        model = self.emu(self.X, self.Y, n_fidelities=self.n_fidelities)
        model.optimize(n_optimization_restarts=10)
        mean_pred, var_pred = model.predict(X_test)
        return mean_pred, var_pred


class Hmf(BaseStatEmu):
    def __init__(self, data_dir, fid=['L2'], logging_level='INFO', narrow=False, no_merge=True, emu_type={'multi-fid':False, 'single-bin':False, 'linear':True}):
        """
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.no_merge = no_merge
        self.emu_type = emu_type
        halo_func = []
        self.Y = []
        self.X = []
        self.labels = []
        #self.sim_specs =[]
        self.mbins = np.arange(11.1, 13.5, 0.1)
        # Iterate over the fidelities
        if emu_type['multi-fid']:
            pairs = summary_stats.get_pairs(data_dir=data_dir, eval_mbins=self.mbins, narrow=narrow)
            self.Y.append(np.log10(pairs[2]['L2']))
            self.Y.append(np.log10(pairs[2]['HF']))
            self.X.append(pairs[4]['L2'])
            self.X.append(pairs[4]['HF'])
            self.labels.append(pairs[5]['L2'])
            self.labels.append(pairs[5]['HF'])
        
        self.logger.info(f'X: {np.array(self.X).shape}, Y: {np.array(self.Y[0]).shape}')

        #self.sim_specs.append(halo_func[-1].get_sims_specs())

        #assert np.all(np.isfinite(self.X)), "Some parameters are not finite"
        #assert np.all(np.isfinite(self.Y)), f"Some Y values are not finite"   

        super().__init__(X=self.X, Y=self.Y, logging_level=logging_level, emu_type=emu_type)