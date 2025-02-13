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
from scipy.interpolate import BSpline
from matplotlib import pyplot as plt
from . import utils
import sys

# Each MPI rank build GP for one bin
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

class BaseSummaryStats:
    """Base class for summary statistics"""
    def __init__(self, data_dir, fid, narrow=False, logging_level='INFO'):
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
        self.narrow = narrow
    
    def configure_logging(self, logging_level='INFO'):
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
    
    def load_ics(self):
        """
        Load the IC json file
        """
        if rank ==0:
            self.logger.info(f'Load IC file from {self.ic_file}')
            # Load JSON file as a dictionary
            with open(self.ic_file, 'r') as file:
                data = json.load(file)
        else:
            data = None
        if MPI is not None:
            data = comm.bcast(data)
        return data
    
    def get_ics(self, keys):
        """
        Get the desired keys from the ICs file
        """
        raw_ics = self.load_ics()
        all_ics = {}
        for k in keys:
            all_ics[k] = []
            for ic in raw_ics:
                if self.narrow:
                    if 'narrow' in ic['label']:
                        all_ics[k].append(ic[k])
                else:
                    all_ics[k].append(ic[k])
        return all_ics
    
    def get_labels(self):
        """
        Get the labels we use for each simulation, they are in this format ``10p_Box{BoxSize}_Par{Npart}_0001``"""
        raise NotImplementedError('Either define this for the child class or pass `labels` to `get_params_array`')

    def get_sims_specs(self):
        """
        Get the simulation specs from the ICs file
        """
        all_ics = self.get_ics(keys=['box', 'npart', 'label'])
        not_computed_sims = []
        labels = self.get_labels()

        matched_labels = []
        existing_sims = []

        for lb in labels:
            for i in range (len(all_ics['label'])):
                if all_ics['label'][i] in lb:
                    existing_sims.append(i)
                    break
        existing_sims = np.array(existing_sims)
        self.logger.info(f'Found {len(existing_sims)} matching labels')
        sim_specs = {}
        for k in ['box','npart']:
            sim_specs[k] = [all_ics[k][i] for i in existing_sims]

        sim_specs['narrow'] = np.zeros((len(existing_sims),))
        for i, sim in enumerate(existing_sims):
            if 'narrow' in all_ics['label'][sim]:
                sim_specs['narrow'][i] = 1
        
        return sim_specs

    def get_cosmo_params(self):
        """
        get comological parameters from the simulations listed in the labels
        """
        
        ics = self.load_ics()
        
        labels = self.get_labels()
        cosmo_params = []
        for lb in labels:
            for ic in ics:
                if ic['label'] in lb:
                    cosmo_params.append({k:ic[k] for k in self.param_names})
                    break
        assert len(cosmo_params) == len(labels), f'Some labels not found in the ICs file, foumd = {len(cosmo_params)}, asked for = {len(labels)}'
        return cosmo_params

    def get_params_array(self):
        """Get the cosmological parameters as an array"""
        params_dict = self.get_cosmo_params()
        return np.array([[cp[p] for p in self.param_names] for cp in params_dict])
    
class ProjCorr(BaseSummaryStats):
    """Projected correlation function, w_p"""
    def __init__(self, data_dir, fid, logging_level='INFO'):

        super().__init__(data_dir, fid, logging_level)
        self.rp = None
        self.data_files = [op.join(data_dir, f) for f in os.listdir(self.data_dir) if self.pref in f]
        self.logger.info(f'Total snapshots: {len(self.data_files)}')
        
    def get_labels(self):
        """Get the labels we use for each simulation, they are in this format ``10p_Box{BoxSize}_Par{Npart}_0001``"""
        labels = []
        for df in self.data_files:
            label = re.search(r'10p_Box\d+_Part\d+_\d{4}',df).group(0)
            if 'narrow' in df:
                label += '_narrow'
            labels.append(label)

        return labels
    

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
    def __init__(self, data_dir, fid, narrow=False, no_merge=True, logging_level='INFO'):
        super().__init__(data_dir, logging_level)
        self.fid = fid
        self.no_merge = no_merge
        self.sim_tags = None
        self.narrow = narrow
        self.logging_level = logging_level
        self.knots = None
    
    def get_labels(self):
        """It is just the simulation tags"""
        if self.sim_tags is None:
            raise ValueError('The simulation tags are not loaded yet, call `load_hmf_sims` first')
        return self.sim_tags

    def load(self):
        """
        Load the Halo Mass Function computed for simulations and saved on `data_dir`
        """
        if self.narrow:
            if self.no_merge:
                save_file = f'{self.fid}_hmfs_narrow_no_merge.hdf5'
            else:
                save_file = f'{self.fid}_hmfs_narrow.hdf5'
        else:
            if self.no_merge:
                save_file = f'{self.fid}_hmfs_no_merge.hdf5'
            else:
                save_file = f'{self.fid}_hmfs.hdf5'
        if rank==0:   
            with h5py.File(op.join(self.data_dir, save_file), 'r') as f:
                bins = f['bins_coarse'][:]
                hmfs = f['hmfs_coarse'][:]
                sim_tags = []
                bad_sims = []
                # We need to convert from binary to str
                for tag in f['sim_tags']:
                    sim_tags.append(tag.decode('utf-8'))
                for bd in f['bad_sims']:
                    bad_sims.append(bd.decode('utf-8'))
        else:
            sim_tags = None
            bad_sims = None
            bins = None
            hmfs = None
        if comm is not None:
            comm.barrier()
            sim_tags = comm.bcast(sim_tags)
            bad_sims = comm.bcast(bad_sims)
            bins = comm.bcast(bins)
            hmfs = comm.bcast(hmfs)
        self.sim_tags = sim_tags
        self.bad_sims = bad_sims
        self.logger.debug(f'Loadded HMFs from {save_file}') 
        return hmfs, bins

    def _do_fits(self, ind=None, *kwargs):
        """
        Fit the halo mass function with a splie.
        Parameters:
        --------------
        kwargs: dict
            Keyword arguments for utils.ConstrainedSplineFitter
        """
        hmfs, bins = self.load()
        # We fix the knots for the spline fit
        self.knots = np.array([11.1 , 11.1, 11.1,
                          11.35, 11.6 , 11.85, 
                          12.1 , 12.35, 12.6 , 
                          12.85, 13.1 , 13.35,
                          13.35, 13.35])
        if ind is None:
            ind = np.arange(len(hmfs))
        fit = utils.ConstrainedSplineFitter(*kwargs, logging_level=self.logging_level)
        splines = []
        for i in ind:
            mbins = 0.5*(bins[i][1:] + bins[i][:-1])
            splines.append(fit.fit_spline(mbins, np.log10(hmfs[i]), self.knots))
        return splines
    
    def get_coeffs(self, ind=None, *kwargs):
        """
        Retrun the spline fits in an array
        Parameters:
        --------------
        ind: np.ndarray, optional, default=None
            Index of the simulations to fit. If None, fit all simulations
        kwargs: dict
            Keyword arguments for utils.ConstrainedSplineFitter
        Returns:
        --------------
        2D array of the spline coefficients and the knots
        (coeffs, knots)
        """
        splines = self._do_fits(ind=ind, *kwargs)
        coeffs = np.zeros((len(splines), len(splines[0].c)))
        for i, spl in enumerate(splines):
            coeffs[i] = spl.c
        return coeffs, splines[0].t
    
    def get_smoothed(self, x, ind=None, *kwargs):
        """
        Get the smoothed halo mass function evaluated at x
        Parameters:
        --------------
        x: np.ndarray or list of np.ndarray
            Array of x values to evaluate the spline at
        ind: np.ndarray, optional, default=None
            Index of the simulations to fit. If None, fit all simulations
        kwargs: dict
            Keyword arguments for utils.ConstrainedSplineFitter
        """
        splines = self._do_fits(ind=ind, *kwargs)
        y = []
        for i, spl in enumerate(splines):
            if type(x) == list:
                eval_points = x[i]
            else:
                eval_points = x
            # The Spline fit was done in log space
            y.append(10**BSpline(spl.t, spl.c, spl.k)(eval_points)) 
        return y
    
    def get_pca(self,n_components=4):
        return 0

    def _sim_nums(self):
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
        for tag in self.sim_tags:
            sim_nums.append(int(re.search(r'_\d{4}',tag)[0][1:]))
        sim_nums = np.array(sim_nums)
        return sim_nums

def get_pairs(data_dir, eval_mbins, no_merge=False, narrow=False):
    """
    """            
    halo_funcs = {}
    hmfs = {}
    bins = {}
    smoothed = {}
    labels = {}
    params = {}
    #sim_specs = {}
    for fd in ['HF', 'L2']:
        halo_funcs[fd] = HMF(data_dir, fid=fd, no_merge=no_merge, narrow=narrow)
        hmfs[fd], bins[fd] = halo_funcs[fd].load()
        sim_nums = halo_funcs[fd]._sim_nums()
        if fd == 'HF':
            common_nums = sim_nums
        else:
            common_nums = np.intersect1d(common_nums, sim_nums)
    halo_funcs['HF'].logger.info(f'Found {len(common_nums)} common pairs')
    
    for fd in ['HF','L2']:
        sim_nums = halo_funcs[fd]._sim_nums()
        ind = np.where(np.isin(sim_nums, common_nums))[0]
        # sort based on the sim # for consistency
        argsort = np.argsort(sim_nums[ind])
        hmfs[fd] = hmfs[fd][ind][argsort]
        bins[fd] = bins[fd][ind][argsort]
        labels[fd] = np.array(halo_funcs[fd].get_labels())[ind][argsort]
        params[fd] = halo_funcs[fd].get_params_array()[ind][argsort]
        #sim_specs[fd] = halo_funcs[fd].get_sims_specs()[ind][argsort]
        
        #mbins[fd] = mbins[fd][ind][argsort]
        smoothed_temp = halo_funcs[fd].get_smoothed(eval_mbins, ind=ind)
        smoothed[fd] = []
        for i in argsort:
            smoothed[fd].append(smoothed_temp[i])
        smoothed[fd] = np.array(smoothed[fd])
    return hmfs, bins, smoothed, eval_mbins, params, labels