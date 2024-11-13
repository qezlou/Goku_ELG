"""
Loading the computed summary statistics from the data files.
"""
import numpy as np
import h5py
import os
import os.path as op
import logging
import json
import re
from matplotlib import pyplot as plt
class ProjCorr:
    """Projected correlation function, w_p"""

    def __init__(self, data_dir, fid, logging_level='INFO', ic_file='all_ICs.json'):

        self.rank = 0
        self.ic_file = ic_file
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.rp = None
        self.params_list = ['omega0', 'omegab', 'hubble', 'scalar_amp', 'ns',
                             'w0_fld', 'wa_fld', 'N_ur',  'alpha_s', 'm_nu']
       
        # All the files in the data directory
        if fid == 'HF':
            pref = 'Box1000_Part3000'
        elif fid == 'L1':
            pref = 'Box1000_Part750'
        elif fid == 'L2':
            pref = 'Box250_Part750'

        self.data_files = [op.join(data_dir, f) for f in os.listdir(self.data_dir) if pref in f]
        self.logger.info(f'Total snapshots: {len(self.data_files)}')

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('get ProjCorr')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_data(self):
        """Load the data from the files
        Returns:
        rp: the projected separation
        w_rp_pi: the projected correlation, shape = (n_files, n_realizations, n_rp, n_pi)
        """
        with h5py.File(op.join(self.data_dir, self.data_files[0]), 'r') as f:
            self.rp = f['r'][:]
            w_rp_pi = np.zeros((len(self.data_files), f['corr'].shape[0], f['corr'].shape[1], f['corr'].shape[2]))
            w_rp_pi[0] = f['corr'][:]
            self.logger.info(f'Orignial corr shape {f["corr"].shape}')
        for i in range(1, len(self.data_files)):
            with h5py.File(op.join(self.data_dir, self.data_files[i]), 'r') as f:
                # averagee along the line of sight, \Pi
                w_rp_pi[i] = f['corr'][:]
        return self.rp, w_rp_pi

    def get_wp(self):
        """Take the avergae along pi direction"""
        _, w_rp_pi = self.load_data()
        neg_frac = np.where(w_rp_pi < 0)[0].size / w_rp_pi.size
        self.logger.info(f'Found {100*neg_frac:.1f} % of W_rp_pi is negative')
        neg_frac= np.where(np.all(w_rp_pi < 0, axis=-1))[0].size / np.prod(w_rp_pi.shape[0:-1])
        self.logger.info(f'Found {100*neg_frac:.1f} % of W_rp is all negeative along the pi direction')
        wp = np.mean(w_rp_pi, axis=-1)
        return self.rp, wp
    
    def get_mean_std(self,  r_range=(0, 100)):
        """get the mean and std of the projected correlation function,
        the std computed over the different realizations of the same HOD 
        and cosmology"""
        rp, wp = self.get_wp()
        ind = np.where((rp>r_range[0]) & (rp<r_range[1]))[0]
        rp = rp[ind]
        # Take average and tsd accross many HOD realizations
        mean = np.mean(wp[:,:,ind], axis=1).squeeze()
        std = np.std(wp[:,:,ind], axis=1).squeeze()
        return rp, mean, std
    

    def load_ics(self):
        """
        Load the IC json file
        """
        self.logger.info(f'Load IC file from {self.ic_file}')
        # Load JSON file as a dictionary
        with open(self.ic_file, 'r') as file:
            data = json.load(file)
        return data

    def get_labels(self):
        """Get the labels we use for each simulation, they are in this format ``cosmo_10p_Box{BoxSize}_Par{Npart}_0001``"""
        labels = [re.search(r'cosmo_10p_Box\d+_Part\d+_\d{4}',pl).group(0) for pl in self.data_files]
        return labels        

    def get_cosmo_params(self):
        """
        get comological parameters from the simulations listed in the labels
        """
        
        ics = self.load_ics()
        labels = self.get_labels()
        cosmo_params = []
        for lb in labels:
            for ic in ics:
                if ic['label'] == lb:
                    cosmo_params.append({k:ic[k] for k in self.params_list})
                    break
        assert len(cosmo_params) == len(labels), f'Some labels not found in the ICs file, foumd = {len(cosmo_params)}, asked for = {len(labels)}'
        return cosmo_params
    
    def get_params_array(self):
        """Get the cosmological parameters as an array"""
        params_dict = self.get_cosmo_params()
        return np.array([[cp[p] for p in self.params_list] for cp in params_dict])
    
    def find_nan_log10_bins(self):

        rp, wp = self.load_data()
        log10_wp = np.log10(wp)
        nan_bins = np.where(np.isnan(log10_wp))
        self.logger.info(f'Found {100*nan_bins[0].size/wp.size:.1f} % of W_p is nan')