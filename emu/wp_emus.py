"""
To evaluate all the emulators for different summary statistics.
This is the interface of `single_fid.py` and `summary_stats.py` to work with the
generated Projectedcorrelation functions.
"""
import logging
import argparse
import numpy as np
import summary_stats
import single_fid

class LogLogSingleFid():
    def __init__(self, data_dir, r_range=(0,30), fid='L2', logging_level='INFO'):
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.r_range = r_range
        self.data_dir = data_dir

        proj = summary_stats.ProjCorr(data_dir=self.data_dir, fid=fid, logging_level=self.logging_level)
        self.rp, wp,  self.model_err = proj.get_mean_std(r_range=self.r_range)
        self.X = proj.get_params_array()
        # CLeaningthe missing bins
        self.Y = self.clean_mssing_bins(wp)
        
        # Get sim labels
        self.labels = proj.get_labels()
        assert len(self.labels) == self.Y.shape[0]
        
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

    def clean_mssing_bins(self, wp):
        """Remove the missing bins from the projected correlation function"""
        # To DO: Perfrom some kind of interpolation or soemthing smarter
        wp[wp < 0] = 1e-10
        return np.log10(wp)
    
    def loo_train_pred(self, savefile):
        """
        Get the leave one out predictions
        """
        self.evaluate.loo_train_pred(rp=self.rp, savefile=savefile, labels=self.labels)
    
    def train_pred_all_sims(self):
        """
        Train the model on all simulations and comapre with the truth
        """
        self.evaluate.sf.train()
        pred, var_pred = self.evaluate.sf.predict(self.X)
        return pred, self.Y, self.rp
    
    def leave_bunch_out(self, n_out=5):
        """
        Leaves out a random bunch of samples out
        n_out: Number of samples to leave out
        """
        model, out_indices = self.evaluate.leave_bunch_out(n_out=n_out)
        X_test = self.X[out_indices]
        Y_test = self.Y[out_indices]

        return  model, X_test, Y_test
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get LOO for the single fidelity emulator')
    parser.add_argument('--data_dir', type=str, default='/home/qezlou/HD2/HETDEX/cosmo/data/corr_projected_corrected/', help='Directory where the data is stored')
    parser.add_argument('--emu_type', type=str, default='LogLogSingleFid', help='Type of the emulator')
    parser.add_argument('--r_range', type=float, nargs=2, default=[0,30], help='Range of r to consider')
    parser.add_argument('--savefile', type=str, default= '/home/qezlou/HD2/HETDEX/cosmo/data/corr_projected_corrected/train/loo_pred.hdf5', help='Save the results to a file')
    args = parser.parse_args()
    
    if args.emu_type == 'LogLogSingleFid':
        emu = LogLogSingleFid(data_dir=args.data_dir, r_range=args.r_range, logging_level='INFO')
        emu.loo_train_pred(savefile=args.savefile)




    
