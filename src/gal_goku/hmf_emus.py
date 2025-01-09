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

class SingleFid():
    def __init__(self, data_dir, y_log=True, r_range=(0,30), fid='L2', cleaning_method='linear_interp', multi_bin=False, logging_level='INFO'):
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
        self.r_range = r_range
        self.data_dir = data_dir
        self.cleaning_method = cleaning_method


        proj = summary_stats.ProjCorr(data_dir=self.data_dir, fid=fid, logging_level=self.logging_level)
        self.sim_specs = proj.get_sims_specs()
        self.rp, wp, self.model_err = proj.get_mean_std(r_range=self.r_range)
        self.X = proj.get_params_array()
        # CLeaningthe missing bins
        self.Y = self.clean_mssing_bins(wp, y_log=y_log)
        
        # Get sim labels
        self.labels = proj.get_labels()
        assert len(self.labels) == self.Y.shape[0]
        
        if multi_bin:
            self.evaluate = single_fid.EvaluateSingleFidMultiBins(X=self.X, Y=self.Y,
                                                                   model_err=self.model_err,
                                                                   logging_level=self.logging_level)
        else:
            self.evaluate = single_fid.EvaluateSingleFid(X=self.X, Y=self.Y, 
                                                        model_err=self.model_err, 
                                                        logging_level=self.logging_level)
    
    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('ProjCorrEmus')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def clean_mssing_bins(self, wp, y_log):
        """Replace the negative bins with the average of the two nearest non-zero bins"""
        if self.cleaning_method == 'linear_interp':
            self.logger.info('Cleaning the negative bins with linear interpolation')
            for i in range(wp.shape[0]):
                non_zero_indices = np.where(wp[i] > 0)[0]
                for j in range(wp.shape[1]):
                    if wp[i,j] <= 0:
                        left = non_zero_indices[non_zero_indices < j]
                        right = non_zero_indices[non_zero_indices > j]
                        if len(left) > 0 and len(right) > 0:
                            left_idx = left[-1]
                            right_idx = right[0]
                            wp[i,j] = wp[i,left_idx] + (wp[i,right_idx] - wp[i,left_idx]) * (j - left_idx) / (right_idx - left_idx)
                        elif len(left) > 0:
                            wp[i,j] = wp[i, left[-1]]
                        elif len(right) > 0:
                            wp[i,j] = wp[i, right[0]]
        elif self.cleaning_method == 'small_number':
            self.logger.info('Cleaning the negative bins with a small number')
            wp[wp <= 0] = 1e-10
        if y_log:
            return np.log10(wp)
        else:
            return np.exp(-wp)
