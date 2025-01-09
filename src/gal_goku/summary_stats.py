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
from scipy.interpolate import InterpolatedUnivariateSpline as ius

class BaseSummaryStats:
    """Base class for summary statistics"""
    def __init__(self, data_dir, fid, logging_level='INFO'):
        self.rank = 0
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.ic_file = op.join(self.data_dir, 'all_ICs.json')
        self.param_names = ['omega0', 'omegab', 'hubble', 'scalar_amp', 'ns',
                        'w0_fld', 'wa_fld', 'N_ur',  'alpha_s', 'm_nu']

        # All the files in the data directory
        if fid == 'HF':
            self.pref = 'Box1000_Part3000'
        elif fid == 'L1':
            self.pref = 'Box1000_Part750'
        elif fid == 'L2':
            self.pref = 'Box250_Part750'
    
    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('get ProjCorr')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

class ProjCorr(BaseSummaryStats):
    """Projected correlation function, w_p"""
    def __init__(self, data_dir, fid, logging_level='INFO'):

        super().__init__(data_dir, fid, logging_level)
        self.rp = None
        self.data_files = [op.join(data_dir, f) for f in os.listdir(self.data_dir) if self.pref in f]
        self.logger.info(f'Total snapshots: {len(self.data_files)}')



    def load_ics(self):
        """
        Load the IC json file
        """
        self.logger.info(f'Load IC file from {self.ic_file}')
        # Load JSON file as a dictionary
        with open(self.ic_file, 'r') as file:
            data = json.load(file)
        return data

    def get_sims_specs(self):
        """
        Get the simulation specs from the ICs file
        """
        all_ics = self.load_ics()
        not_computed_sims = []
        labels = self.get_labels()

        matched_labels = []
        ics_existing_sims = []

        for label in labels:
            for ic in all_ics:
                if ic['label'] == label:
                    ics_existing_sims.append(ic)
                    break
        self.logger.debug(f'Found {len(ics_existing_sims)} matching labels')

        del all_ics
        sim_specs = {}
        for k in ['box','npart']:
            sim_specs[k] = [ic[k] for ic in ics_existing_sims]

        sim_specs['narrow'] = np.zeros(len(ics_existing_sims))
        for i, ic in enumerate(ics_existing_sims):
            if 'narrow' in ic['label']:
                sim_specs['narrow'][i] = 1
        return sim_specs
        
    def get_labels(self):
        """Get the labels we use for each simulation, they are in this format ``cosmo_10p_Box{BoxSize}_Par{Npart}_0001``"""
        labels = []
        for df in self.data_files:
            label = re.search(r'10p_Box\d+_Part\d+_\d{4}',df).group(0)
            if 'narrow' in df:
                label += '_narrow'
            labels.append(label)

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
                    cosmo_params.append({k:ic[k] for k in self.param_names})
                    break
        assert len(cosmo_params) == len(labels), f'Some labels not found in the ICs file, foumd = {len(cosmo_params)}, asked for = {len(labels)}'
        return cosmo_params

    def get_params_array(self):
        """Get the cosmological parameters as an array"""
        params_dict = self.get_cosmo_params()
        return np.array([[cp[p] for p in self.param_names] for cp in params_dict])
    

    def get_wp(self):
        """Take the avergae along pi direction"""
        with h5py.File(op.join(self.data_dir, self.data_files[0]), 'r') as f:
            self.rp = f['r'][:]
            w_p = np.zeros((len(self.data_files), f['corr'].shape[0], f['corr'].shape[1]))
            w_p[0] = f['corr'][:]
        for i in range(1, len(self.data_files)):
            with h5py.File(op.join(self.data_dir, self.data_files[i]), 'r') as f:
                w_p[i] = f['corr'][:]  
        return self.rp, w_p   
    
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
    

    def bin_in_param_space(self, r_range=(0,100), num_bins=2):
        """
        Computet the average wp of all the simulations with low and 
        high values of the cosmological parameters
        Parameters
        ----------
        r_range: tuple
            The range of r values to consider
        num_bins: int
            Number of bins to divide the cosmological parameters
        Returns
        -------
        rp: array
            The r values
        av_wp: array
            The average wp for each bin
        all_bins: array
            The bin edges for each parameter
        """

        rp, wp, std = self.get_mean_std(r_range=r_range)
        av_wp = np.zeros((num_bins, len(self.param_names), wp.shape[1]))
        av_wp_std = np.zeros((num_bins, len(self.param_names), wp.shape[1]))
        params_array = self.get_params_array()

        # Min and max of cosmo params acrros all the simulations
        min_param = np.min(params_array, axis=0)
        max_param = np.max(params_array, axis=0)
        all_bins = np.zeros((len(self.param_names), num_bins+1))

        for p in range(params_array.shape[1]):
            bins = np.linspace(min_param[p], max_param[p], num_bins+1)
            all_bins[p] = bins
            bin_ind = np.digitize(params_array[:,p], bins)
            for b in range(num_bins):
                ind = np.where(bin_ind == b+1)[0]
                av_wp[b, p] = np.mean(wp[ind], axis=0)
                av_wp_std[b, p] = np.mean(std[ind], axis=0)
        return rp, av_wp, av_wp_std, all_bins
        
    
    def find_nan_log10_bins(self):

        rp, wp = self.load_data()
        log10_wp = np.log10(wp)
        nan_bins = np.where(np.isnan(log10_wp))
        self.logger.info(f'Found {100*nan_bins[0].size/wp.size:.1f} % of W_p is nan')


class HMF(BaseSummaryStats):
    """
    Halo mass function
    """
    def __init__(self, data_dir, logging_level='INFO'):
        super().__init__(data_dir, logging_level)
    
    def load_hmf_sims(self, fids=[], load_coarse=False):
        """
        Load the Halo Mass Function computed for simulations 
        at a given fidelity.
        Parameters:
        --------------
        save_dir: str
            Directory where the HMF files are stored.
        fids: list
            List of fidelities to compare.
        load_coarse: bool
            If True, load the HMFs on the coarse mass bins. This is
            the bins the HMF was originally computed on and then interpolated
            to the fine bins.
        Returns:
        --------------
        hmfs: dict
            Dictionary containing the Halo Mass Function in units of Mpc^-3 h^3 dex^-1.
            The keys are the fidelities, e.g., 'HF', 'L2', 'L1'. The values are the HMFs.
        mbins: np.ndarray
            Mass bins.
        sim_tags: dict
            Dictionary containing lists of simulation tags.
        If load_coarse is True:
        hmfs_coarse: dict
            Dictionary containing the HMFs on the coarse mass bins.
        mbins_coarse: np.ndarray
            Mass bins on the coarse grid.
        """
        hmfs = {}
        sim_tags = {}
        if load_coarse:
            hmfs_coarse = {}
            hmfs_fine = {}
        for fd in fids:
            with h5py.File(op.join(self.data_dir, f'{fd}_hmfs.hdf5'), 'r') as f:
                hmfs[fd] = f['hmfs'][:]
                mbins =  0.5*(10**f['bins'][1:]+10**f['bins'][:-1])
                if load_coarse:
                    hmfs_coarse[fd] = f['hmfs_coarse'][:]
                    hmfs_fine[fd] = f['hmfs_fine'][:]
                    mbin_coarse = 0.5*(10**f['bins_coarse'][1:]+10**f['bins_coarse'][:-1])
                sim_tags[fd] = []
                # We need to convert from binary to str
                for tag in f['sim_tags']:
                    sim_tags[fd].append(tag.decode('utf-8'))
        if load_coarse:
            return hmfs, mbins, sim_tags, hmfs_coarse, mbin_coarse, hmfs_fine
        else:
            return hmfs, mbins, sim_tags

    def _sim_nums(self, sim_tags):
        """
        Get the simulation id from the simulation tags
        Parameters:
        --------------
        sim_tags: list
            List of simulation tags
        Returns:
        --------------
        sim_nums: np.ndarray
            Array of simulation numbers
        """
        sim_nums = []
        for tag in sim_tags:
            sim_nums.append(int(re.search(r'_\d{4}',tag)[0][1:]))
        sim_nums = np.array(sim_nums)
        return sim_nums

    def common_pairs(self, fids=['HF','L2']):
        """
        Get the common pairs between the different cosmologies
        Parameters:
        --------------
        save_dir: str
            Directory where the hmf files are stored
        fids: list
            List of fidelities to compare
        Returns:
        --------------
        first_corrs: list
            List of files for the first fid
        """
        hmfs, mbins, sim_tags = self.load_hmf_sims(fids)
        
        # Find the common pairs
        for fd in fids: 
            sim_nums = self._sim_nums(sim_tags[fd])
            if fd == fids[0]:
                common_nums = sim_nums
            else:
                common_nums = np.intersect1d(common_nums, sim_nums)
        
        self.logger.info(f'Found {len(common_nums)} common pairs')
        # Now keeping only the common hmfs
        for fd in fids:
            sim_nums = self._sim_nums(sim_tags=sim_tags[fd])
            ind = np.where(np.isin(sim_nums, common_nums))[0]
            # Sort based on sim # for consistency
            arg_sort = np.argsort(sim_nums[ind])
            hmfs[fd] = hmfs[fd][ind][arg_sort]
            sim_tags[fd] = np.array(sim_tags[fd])[ind][arg_sort]

        return hmfs, mbins,sim_tags