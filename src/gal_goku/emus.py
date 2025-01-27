"""
To evaluate all the emulators for the Halo Mass Function.
This is the interface of `single_fid.py` and `summary_stats.py` to work with the
generated emulated Halo Mass Fucntion functions.
"""

import logging
import argparse
import numpy as np
from . import summary_stats
from . import single_fid

class BaseStatEmu():

    def __init__(self, X, Y, model_error, logging_level='info', multi_bin=False):
        """The base emu to be inherited by single fidelity emulators built on any summary statistics
        This is the interface for all classses above.
        """
        if multi_bin:
            self.evaluate = single_fid.EvaluateSingleFidMultiBins(X, Y, model_error, logging_level)
        else:
            self.evaluate = single_fid.EvaluateSingleFid(X, Y, model_error, logging_level)

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('BaseStatEmu')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    def loo_train_pred(self, savefile, narrow=0):
        """
        Get the leave one out predictions
        """
        if narrow:
            sub_sample = np.where(self.sim_specs['narrow'] == 1)[0]
            self.logger.info('Narrow sims are selected, sub_sample size is %d', len(sub_sample))
        else:
            sub_sample = None
        self.evaluate.loo_train_pred(mbins=self.mbins, savefile=savefile, labels=self.labels, sub_sample=sub_sample)
    
    def train_pred_all_sims(self):
        """
        Train the model on all simulations and comapre with the truth
        """
        self.evaluate.train()
        pred, var_pred = self.evaluate.predict(self.X)
        return pred, self.Y, self.mbins
    
    def leave_bunch_out(self, n_out=5, narrow=0):
        """
        Leaves out a random bunch of samples out
        n_out: Number of samples to leave out
        """
        if narrow:
            sub_sample = np.where(self.sim_specs['narrow'] == 1)[0]
        else:
            sub_sample = None
        X_test, Y_test, Y_pred, var_pred = self.evaluate.leave_bunch_out(n_out=n_out, sub_sample=sub_sample)
        return X_test, Y_test, Y_pred, var_pred
    
    def predict(self, X_test):
        """
        Predict the mean and variance of the emulator
        """
        self.evaluate.train()
        return self.evaluate.predict(X_test)


class Hmf(BaseStatEmu):
    def __init__(self, data_dir, y_log=True, fid='L2', multi_bin=False, logging_level='INFO', narrow=False, no_merge=True):
        """
        data_dir: Directory where the data is stored
        r_range: Range of r to consider
        fid: The fiducial cosmology to consider, default is 'L2'
        logging_level: Logging level
        cleaning_method: Method to clean the negative bins, 
                default is 'linear_interp'; otherwuse replace with a small
                number, e.g. 1e-10
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.no_merge = no_merge
        halo_func = summary_stats.HMF(data_dir, fid=fid, narrow=narrow, no_merge=no_merge)
        hmfs, _ = halo_func.load()
        num_sims = len(hmfs)
        del hmfs
        self.mbins = np.arange(11.1, 13.5, 0.1)
        self.Y =  halo_func.get_smoothed(self.mbins)
        self.labels = halo_func.sim_tags
        self.sim_specs = halo_func.get_sims_specs()
        if y_log:
            self.Y = np.log10(self.Y)

        model_error = np.zeros_like(self.Y)
        self.X = halo_func.get_params_array()
        assert np.all(np.isfinite(self.X)), "Some parameters are not finite"
        assert np.all(np.isfinite(self.Y)), f"Some Y values are not finite"        
        #assert len(sim_tags) == self.Y.shape[0]
        super().__init__(X=self.X, Y=self.Y, model_error=model_error, logging_level=logging_level, multi_bin=multi_bin)