"""
To evaluate all the emulators for different summary statistics.
This is the interface of `single_fid.py` and `summary_stats.py` to work with the
generated Projectedcorrelation functions.
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
    
    def loo_train_pred(self, savefile, narrow=0):
        """
        Get the leave one out predictions
        """
        if narrow:
            sub_sample = np.where(self.sim_specs['narrow'] == 1)[0]
        else:
            sub_sample = None
        self.evaluate.loo_train_pred(rp=self.rp, savefile=savefile, labels=self.labels, sub_sample=sub_sample)
    
    def train_pred_all_sims(self):
        """
        Train the model on all simulations and comapre with the truth
        """
        self.evaluate.sf.train()
        pred, var_pred = self.evaluate.predict(self.X)
        return pred, self.Y, self.rp
    
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
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get LOO for the single fidelity emulator')
    parser.add_argument('--data_dir', type=str, default='/home/qezlou/HD2/HETDEX/cosmo/data/corr_projected_corrected/', help='Directory where the data is stored')
    parser.add_argument('--emu_type', type=str, default='SingleFid', help='Type of the emulator')
    parser.add_argument('--multi_bin', type=int, default=0, help='Build one mu per rp bin?')
    parser.add_argument('--r_range', type=float, nargs=2, default=[0,30], help='Range of r to consider')
    parser.add_argument('--y_log', type=int, default=1, help='Wether to train on log10 of wp')
    parser.add_argument('--savefile', type=str, default= '/home/qezlou/HD2/HETDEX/cosmo/data/corr_projected_corrected/train/loo_pred_lin_interp_not_log_y.hdf5', help='Save the results to a file')
    parser.add_argument('--narrow', type=int, default=0, help='Use only the narrow simulations')
    args = parser.parse_args()
    
    if args.emu_type == 'SingleFid':
        emu = SingleFid(data_dir=args.data_dir, y_log=args.y_log, r_range=args.r_range, multi_bin=args.multi_bin, logging_level='INFO')
        emu.loo_train_pred(savefile=args.savefile, narrow=args.narrow)




    
