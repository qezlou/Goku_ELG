"""
Loading the computed summary statistics from the data files.
"""
import numpy as np
import h5py
import os
from glob import glob
import os.path as op
import logging
import json
import re
#from scipy.interpolate import BSpline, LSQBivariateSpline, make_lsq_spline, make_splrep, UnivariateSpline, bisplrep
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from . import utils
try:
    from . import halo_tools
except ImportError:
    halo_tools = None
    print("halo_tools not available, won't be able to use propagator.")
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gpflow
from gpflow.utilities import print_summary
import tensorflow as tf

class BaseSummaryStats:
    """Base class for summary statistics"""
    def __init__(self, data_dir, fid, narrow=False, MPI=None, logging_level='INFO'):
        # MPI is useful only when loading ICs or HMF
        # files, to avoid loading the same file on all ranks
        # Can let `MPI=None` if there aren't too many ranks
        self.MPI = MPI
        if MPI is not None:
                self.comm = MPI.COMM_WORLD
                self.rank = self.comm.Get_rank()
                self.mpi_size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.mpi_size = 1
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
    
    def configure_logging(self, logging_level='INFO', logger_name='summary_stats'):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | Rank {self.rank} | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger

    def make_big_ic_file(self, base_dirs = ['/scratch/06536/qezlou/Goku/FOF/HF/',
                                            '/scratch/06536/qezlou/Goku/FOF/L1/',
                                            '/scratch/06536/qezlou/Goku/FOF/L2/',
                                            '/scratch/06536/qezlou/Goku/FOF/L2/narrow/',
                                            '/scratch/06536/qezlou/Goku/FOF/HF/narrow/']):
        
        
        # Load JSON file as a dictionary
        def load_json_as_dict(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data

        all_ICs = []
        for bsd in base_dirs:
            # Example usage
            path = glob(op.join(bsd,f'*_10p_Box*'))
            dir_names = [p for p in path if op.isdir(p)]
            self.logger.info(f'number of files in {bsd} is {len(dir_names)}')
            for dir_n in dir_names:
                data_dict = load_json_as_dict(op.join(dir_n, 'SimulationICs.json'))
                # Get sim number, Box, Part from the directory name
                # "box" and "naprt" are not properly recorded on the SimulcationICs.json files
                snap_num = dir_n.split('_')[-1].split('.')[0] 
                box = re.search(r'Box(\d+)', dir_n).group(1)
                part = re.search(r'Part(\d+)', dir_n).group(1)
                data_dict['box'] = int(box)
                data_dict['npart']= int(part)
                data_dict['label'] = f'10p_Box{box}_Part{part}_{snap_num}'
                if 'narrow' in bsd:
                    data_dict['label'] += '_narrow'
                all_ICs.append(data_dict)
        save_file = 'all_ICs.json'
        self.logger.info(f'writing on {save_file}')
        with open(save_file, 'w') as json_file:
            json.dump(all_ICs, json_file, indent=4)

        with open(save_file, 'r') as json_file:
            data = json.load(json_file)
            self.logger.info(f'totla files = {len(data)}')

    def load_ics(self):
        """
        Load the IC json file
        """
        if self.rank ==0:
            self.logger.debug(f'Load IC file from {self.ic_file}')
            # Load JSON file as a dictionary
            with open(self.ic_file, 'r') as file:
                data = json.load(file)
        else:
            data = None
        if self.MPI is not None:
            data = self.comm.bcast(data)
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

class MatterPower(BaseSummaryStats):
    """
    Class to prepare the propagator for the emulator.
    """

    def __init__(self, data_dir, z, max_k=0.5, log_k_interp=None, fid='HF', logging_level='INFO'):
        """
        Initialize the Propagator class.

        Parameters
        ----------
        basedir : str
            Base directory where the data files are located.
        """
        self.basedir = op.join(data_dir, 'power_matter', fid)
        self.fid = fid
        self.z = z
        self.max_k = max_k
        self.log_k_interp = log_k_interp
        self.fnames = glob(op.join(self.basedir, 'power*.txt'))
        self.rank = 0
        super().__init__(data_dir, fid, logging_level)
        self.logger.info(f'Found {len(self.fnames)} files in {self.basedir}')
        if len(self.fnames) != 0:
            self.k, self.pk, self.sim_tags = self.get_powers()

    def get_labels(self):
        """It is just the simulation tags"""
        if self.sim_tags is None:
            raise ValueError('The simulation tags are not loaded yet, call `load()` first')
        return self.sim_tags

    def get_powers(self):
        """
        Loading the matter power spectrum for each simulation.
        The saved files are in plain ascii format, with the first line
        being the header and the rest being the k and P_m values.
        """
        all_pk = []
        sim_tags = []
        for i, fn in enumerate(self.fnames):
            with open(fn, 'r') as f:
                lines = f.readlines()
            p_m = []
            k = []
            for line in lines[1:]:
                if line.startswith('#'):
                    continue
                parts = line.split()
                k.append(float(parts[0]))
                p_m.append(float(parts[1]))
            p_m = np.array(p_m)
            all_pk.append(p_m)
            sim_tags.append(fn[-40:-4])
        all_pk = np.array(all_pk)
        k = np.array(k)
        # apply max_k cut
        ind = np.where(k <= self.max_k)
        k = k[ind]
        all_pk = all_pk[:,ind]

        # interpolate the power spectrum if log_k_interp is provided
        if self.log_k_interp is not None:
            all_pk = interp1d(np.log10(k), all_pk, axis=-1, bounds_error=False,
                            fill_value='extrapolate')(self.log_k_interp)
            k = 10** self.log_k_interp

        # Sort based on the simulation number:
        sorted_indices = np.argsort([int(str(x)[-4:]) for x in sim_tags])
        all_pk = all_pk[sorted_indices]
        sim_tags = [sim_tags[i] for i in sorted_indices]
        return k, all_pk, sim_tags

class HaloHaloPower(BaseSummaryStats):
    """
    Class to prepare the propagator for the emulator.
    """

    def __init__(self, data_dir, z, max_k=0.5, log_k_interp=None,  fid='HF', logging_level='INFO'):
        """
        Initialize the Propagator class.

        Parameters
        ----------
        basedir : str
            Base directory where the data files are located.
        """
        self.basedir = op.join(data_dir, 'power_bins', fid)
        self.fid = fid
        self.z = z
        self.max_k = max_k
        self.log_k_interp = log_k_interp
        self.fnames = glob(op.join(self.basedir, 'power*.hdf5'))
        self.rank = 0
        super().__init__(data_dir, fid, logging_level)
        self.logger.info(f'Found {len(self.fnames)} files in {self.basedir}')
        if len(self.fnames) != 0:
            self.k, self.pk, self.sim_tags, self.mbins, self.mpairs = self.get_powers()

    def get_labels(self):
        """It is just the simulation tags"""
        if self.sim_tags is None:
            raise ValueError('The simulation tags are not loaded yet, call `load()` first')
        return self.sim_tags

    def get_powers(self):
        """
        Loading the matter power spectrum for each simulation.
        The saved files are in plain ascii format, with the first line
        being the header and the rest being the k and P_m values.
        """
        all_pk = []
        sim_tags = []
        for i, fn in enumerate(self.fnames):
            with h5py.File(fn, 'r') as f:
                k = f['k'][:]
                mbins = f['mbins'][:]
                mpairs = f['pairs'][:]
                all_pk.append(f['power'][:])
                sim_tags.append(f['sim_tag'][()].decode('utf-8'))
        
        all_pk = np.array(all_pk)
        # Mask the nans and k=0
        mask = ~np.isnan(k)
        all_pk = all_pk[:,:,mask]
        k = k[mask]
        mask = k > 0
        all_pk = all_pk[:,:,mask]
        k = k[mask]

        # apply the max_k limit
        ind = np.where(k <= self.max_k)
        k = k[ind]
        all_pk = all_pk[:,:,ind]

        # Sort based on the simulation number:
        sorted_indices = np.argsort([int(str(x)[-4:]) for x in sim_tags])
        all_pk = all_pk[sorted_indices].squeeze()
        sim_tags = [sim_tags[i] for i in sorted_indices]

        # interpolate the power spectrum if log_k_interp is provided
        if self.log_k_interp is not None:
            all_pk = interp1d(np.log10(k), all_pk, axis=-1, bounds_error=False,
                            fill_value='extrapolate')(self.log_k_interp)
            k = 10** self.log_k_interp
        return k, all_pk, sim_tags, mbins, mpairs

class Propagator(BaseSummaryStats):
    """
    Class to prepare the propagator for the emulator.
    """

    def __init__(self, data_dir, z, max_k=0.5,  fid='HF', logging_level='INFO'):
        """
        Initialize the Propagator class.

        Parameters
        ----------
        basedir : str
            Base directory where the data files are located.
        """
        self.basedir = op.join(data_dir, 'power_hlin', fid)
        self.fid = fid
        self.z = z
        self.max_k = max_k
        self.fnames = glob(op.join(self.basedir, 'power*.hdf5'))
        self.rank = 0
        super().__init__(data_dir, fid, logging_level)
        self.logger.info(f'Found {len(self.fnames)} files in {self.basedir}')
        self.gk, self.k, self.mbins, self.sim_tags, _, _, _ = self.get_powers()
        self.params = self.get_params_array()
        self.coeffs = self._fit_gk()

    def get_labels(self):
        """It is just the simulation tags"""
        if self.sim_tags is None:
            raise ValueError('The simulation tags are not loaded yet, call `load()` first')
        return self.sim_tags

    def get_powers(self):
        """
        Get the computed propagator computed for each simulation.
        It is:
            G(k, m) = P_hlin(k, m) / P_init(k, m)
            where P_hlin is the cross correlation of the halos 
            and the initial power spectrum,
            P_init is the initial power spectrum,
            k and m are the wavenumber and mass bins respectively.
        """
        all_ratios = []
        sim_tags = []
        fluctuating_sims = []
        all_phlin = []
        all_pinit= []
        for fn in self.fnames:
            with h5py.File(fn, 'r') as f:
                ratio = f['power_hlin'][:,:] / f['power_init'][:]
                phlin = f['power_hlin'][:,:]
                pinit = f['power_init'][:]
                k = f['k_hlin'][:]
                mbins = f['mbins'][:]
                # The first k bin is nan, so we remove it
                k = k [1:]
                ratio = ratio[:, 1:]
                phlin = phlin[:, 1:]
                pinit = pinit[1:]
                # place a cap on k
                ind = np.where(k <= self.max_k)
                k = k[ind]
                ratio = ratio[:, ind]
                phlin = phlin[:, ind]
                pinit = pinit[ind]
                # Finding the sims with too much fluctuation
                if np.any(ratio < 0.1):
                    fluctuating_sims.append(f['sim_tag'])
                    print(f'Simulation {f["sim_tag"][()]} has too much fluctuation')
                sim_tags.append(f['sim_tag'][()].decode('utf-8'))
                all_ratios.append(ratio.squeeze())
                all_phlin.append(phlin.squeeze())
                all_pinit.append(pinit.squeeze())
        all_ratios = np.array(all_ratios)
        all_phlin = np.array(all_phlin)
        all_pinit = np.array(all_pinit)

        # Sort based on the simulation number:
        sorted_indices = np.argsort([int(str(x)[-4:]) for x in sim_tags])
        all_ratios = all_ratios[sorted_indices]
        all_phlin = all_phlin[sorted_indices]
        all_pinit = all_pinit[sorted_indices]
        sim_tags = [sim_tags[i] for i in sorted_indices]

        return all_ratios, k, mbins, sim_tags, all_phlin, all_pinit, fluctuating_sims

    def model(self, k, g0, g2, g4, sigma_d):
        """
        The model for the propagator G(k) = P_hm / P_lin
        G(k) = (g_0 + g_2 k^2 + g_4 k^4) exp(- sigma_d^2_lin k^2 / 2 )
        where sigma_d is the Zeldovich displacement.
        Returns:
        -------
        G(k): np.ndarray, shape=(n_sims, n_mbins)
            The propagator G(k) for each simulation and mass bin.
        """
        return (g0 + g2 * k**2 + g4 * k**4) * np.exp(-1 * sigma_d**2 * k**2 / 2)
    
    def _fit_gk(self):
        """
        Fit the equation below to G(k) = P_hm / P_lin
        G(k) = (g_0 + g_2 k^2 + g_4 k^4) exp(- sigma_d^2_lin k^2 / 2 )
        where sigma_d is the Zeldovich displacement.
        
        # TODO: Replace sigma_d with `halotools.get_zeldovich_displacement()` which
        in turn needs the p_lin at observed redshift, i.e. z=0.99. For now, we comoute using
        CAMB's output for p_lin(z). This could be off from what we get from CLASS. 

        Returns:
        -------
        coeffs: np.ndarray, shape=(n_sims, n_mbins, 3)
        The coefficients g_0, g_2, g_4 and sigma_d for each simulation and mass bin.
        fit_gks: np.ndarray, shape=(n_sims, n_mbins, n_k)
        The fitted G(k) for each simulation and mass bin.
        """
        # stores the coefficients g_0, g_2, g_4 and sigma_d for each simulation and mass bin
        coeffs = np.full((self.gk.shape[0], self.gk.shape[1], 4), np.nan)
        # iterate over the simulations
        for i in range(len(self.fnames)):
            htools = halo_tools.HaloTools(self.params[i], self.z)
            sigma_d = htools.get_zeldovich_displacement_camb_power()
            self.logger.debug(f'sigma_d = {sigma_d}')
            # iterate over the mass bins
            for m in range(self.gk.shape[1]):
                func = lambda k, g0, g2, g4: self.model(k, g0, g2, g4, sigma_d)
                popt, _ = curve_fit(func, self.k, self.gk[i,m])
                coeffs[i, m, 0:3] = popt
                coeffs[i, m, -1] = sigma_d
        return coeffs
    
    def get_model_gk(self, k):
        """
        Get the propagator G(k) = P_hm / P_lin at arbitrary k values.
        Returns:
        -------
        gk: np.ndarray, shape=(n_sims, n_mbins, n_k)
        """
        assert hasattr(self, 'coeffs'), 'Coefficients not computed yet, call `_fit_gk()` first'
        gk = np.zeros((self.gk.shape[0], self.gk.shape[1], k.shape[0]))
        for i in range(self.gk.shape[0]):
            for m in range(self.gk.shape[1]):
                g0, g2, g4, sigma_d = self.coeffs[i, m]
                gk[i, m] = self.model(k, g0, g2, g4, sigma_d)
        return gk

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

class Xi(BaseSummaryStats):
    """
    Speherically Averaged correlation function
    """
    def __init__(self, data_dir, fid, mass_range=(11.0, 12.32), narrow=False, MPI=None, logging_level='INFO'):
        """
        Calling this will load the xi(r, n1, n2) for all the simulations
        Parameters:
        --------------
        data_dir: str
            The directory containing the data
        fid: str
            The folder id
        narrow: bool, optional, default=False
            Whether it is Goku-narrow or not
        logging_level: str, optional, default='INFO'
            The logging level
        """
        super().__init__(data_dir, fid, MPI=MPI, logging_level=logging_level)
        self.data_dir = data_dir
        self.fid = fid
        self.narrow = narrow
        # Get the file counts
        if self.narrow:
            self.xi_file = op.join(self.data_dir, self.fid, 'narrow', f'xi_grid_{self.fid}_narrow.hdf5')
            self.sim_tags, self.rbins, self.xi, self.mass_pairs = self._load_data(mass_range=mass_range)
            self.sim_nums = [int(f.split('_')[-2]) for f in self.sim_tags]
        else:
            self.xi_file = op.join(self.data_dir, self.fid, f'xi_grid_{self.fid}.hdf5')
            self.sim_tags, self.rbins, self.xi, self.mass_pairs = self._load_data(mass_range=mass_range)
            self.sim_nums = [int(f.split('_')[-1]) for f in self.sim_tags]

        # Sort the dat based on the sim id
        self.sim_nums = np.array(self.sim_nums).astype(int)
        ind_sort = np.argsort(self.sim_nums)
        self.sim_tags = np.array(self.sim_tags)[ind_sort]
        self.sim_nums = np.array(self.sim_nums)[ind_sort]
        self.xi = self.xi[ind_sort]

        self.logger.debug(f'Total sims files: {len(self.sim_tags)} in {op.join(self.data_dir, self.fid)}')
        
        self.mass_pairs = np.around(self.mass_pairs, 2)
        self.mass_bins = np.round(np.unique(self.mass_pairs), 2)
        self.logger.info(f'mass_bins = {self.mass_bins}')

    def _load_data(self, mass_range):
        """
        Load the correlation function for a single simulation
        """
        sim_tags= []
        with h5py.File(self.xi_file, 'r') as f:
            rbins = f['mbins'][:]
            xi = f['corr'][:]
            mass_pairs = f['mass_pairs'][:]
            for tag in f['sim_tags']:
                sim_tags.append(tag.decode('utf-8'))
        
        ind = (mass_pairs[:,0] >= mass_range[0]) & (mass_pairs[:,0] <= mass_range[1]) & \
                (mass_pairs[:,1] >= mass_range[0]) & (mass_pairs[:,1] <= mass_range[1])
        mass_pairs = mass_pairs[ind,:]
        xi = xi[:, ind, :]
        return sim_tags, rbins, xi, mass_pairs
    
    def get_labels(self):
        """It is just the simulation tags"""
        if self.sim_tags is None:
            raise ValueError('The simulation tags are not loaded yet, call `load()` first')
        return self.sim_tags
    
    def get_xi_r(self, mass_pair, rcut = (0.2, 61)):
        """
        Get xi(r) for all simulations at fixed mass pair
        """
        ind_m = np.where( (self.mass_pairs[:,0] == mass_pair[0]) & 
                         (self.mass_pairs[:,1] == mass_pair[1]))[0]
        ind_r = np.where((self.rbins > rcut[0]) & (self.rbins < rcut[1]))[0]

        return self.rbins[ind_r], self.xi[:, ind_m, ind_r].squeeze()

    def make_3d_corr(self, corr, symmetric=False):
        """
        Pass xi(n_mass_pairs, n_rbins) and get a
        3D array (n_mass_pairs, n_mass_pairs, n_rbins)
        """
        mass_bins = self.mass_bins[::-1]
        ind_m_pair = np.digitize(self.mass_pairs, mass_bins).astype(int)
        corr_3d = np.full((mass_bins.size, mass_bins.size, corr.shape[-1]), np.nan)
        for (i,j), val in zip(ind_m_pair, corr):
            corr_3d[i,j] = val
        if symmetric:
            for i in range(mass_bins.size):
                for j in range(mass_bins.size):
                    corr_3d[j,i] = corr_3d[i,j]
        return corr_3d
    
    def _xi_sim_n1_n2(self, sim_tag, r_ind=None, symmetric=False):
        """
        Get xi(n1, n2) for a single simulation at fixed r0 = rbins[r_ind]
        """
        sim_tag, rbins, corr, mass_pairs = self._load_data(sim_tag)
        mass_bins = np.unique(mass_pairs)[::-1]
        ind_m_pair = np.digitize(mass_pairs, mass_bins).astype(int)

        corr_2d = np.full((mass_bins.size, mass_bins.size), np.nan)
            
        for (i,j), val in zip(ind_m_pair, corr):
            corr_2d[i,j] = val[r_ind]
        if symmetric:
            corr_2d = np.triu(corr_2d) + np.triu(corr_2d, 1).T
        return  rbins[r_ind], mass_bins,  corr_2d
    
    def _xi_sim_n1_n2_r(self, sim_tag, symmetric=True):
        """
        Get xi(n1, n2, r) for a single simulation
        """
        _, _, corr, _ = self._load_data(sim_tag)
        corr_3d = self.make_3d_corr(corr, symmetric=symmetric)
        return corr_3d
    
    def get_xi_n1_n2(self, r_ind, symmetric=False):
        """
        Get Xi(n1, n2) for all simulations at fixed r
        """
        all_corrs = []
        for sim_tag in self.sim_tags:
            r0, mass_bins, corr = self._xi_sim_n1_n2(sim_tag, r_ind=r_ind, symmetric=symmetric)
            all_corrs.append(corr)
        return r0, mass_bins, np.array(all_corrs)
    
    def get_xi_n1_n2_r(self):
        """
        Get xi(n1, n2, r) for all simulations
        Returns:
        --------------
        rbins: np.ndarray, shape=(n_rbins,)
            The r values
        mass_bins: np.ndarray, shape=(n_mass_bins,)
            The mass bins, storing the mass values
            in descending order. This is the order the corr is stored
        all_corrs: np.ndarray, shape=(n_sims, n_mass_bins, n_mass_bins, n_rbins)
            The xi(n1, n2, r) for all the simulations
        """
        all_corrs = []
        for i in range(len(self.sim_tags)):
            all_corrs.append(self.make_3d_corr(self.xi[i], symmetric=True))
        return np.array(all_corrs)
    
    def _replace_nan_corrs(self, log_corr, window=5):
        """
        Replace the correlation function bins with
        negative values (nan in log space) with the average
        value of the corrs at smaller mass bins with a `window`
        size.
        """
        pass

    def _sim_fit_spline_n1n2(self, sim_tag, r=None, r_ind=None, tx= None, allow_scipy=False):
        """
        Fit a bivariate to the xi(n1,n2) at r0 = rbins[r_ind]
        Parameters:
        --------------
        sim_tag: str
            The simulation tag
        r_ind: int
            The index of the r value to fit
        tx: np.ndarray, optional, default=None
            The knots for the spline fit
        Retruns:
        --------------
        r0: float
            The r value at which the fit is done
        mass_bins: np.ndarray
            The mass bins in increasing order
        corr_2d: np.ndarray
            The trimmed log10(xi(n1, n2)) array, with
            nan values replaced by -10
        spline: scipy.interpolate.LSQBivariateSpline
            The spline fit to the log10(xi(n1, n2)) at r0.
        NOTE: The spline corresponds to the "flipped"
        mass_bins, corr_2d arrays.

        """
        if r is not None:
            r_ind = np.argmin(np.abs(self.rbins - r))
        ## Load a 2D array, (sim, m1_cut, m2_cut) which are
        ## the correlation functions at fixed xi(r_0)
        r0, mass_bins, corr_2d = self._xi_sim_n1_n2(sim_tag, r_ind=r_ind, symmetric=True)
        # We need the mass bins to be in ascending order    
        mass_bins = np.around(mass_bins[::-1], 2)
        corr_2d = np.flip(corr_2d, axis=(0,1))

        corr_2d = np.log10(corr_2d)

        ## The fixed knots for the spline
        if tx is None:
            tx = np.array([11.1, 11.5, 11.9, 12.1, 12.3, 12.5, 13.0])
        tx = np.round(tx, 2)
        ty = np.copy(tx)
        common_indices = np.where(np.isin(mass_bins, tx))[0]
        assert np.all(mass_bins[common_indices] == tx), f'The knots are not in the mass bins {mass_bins[common_indices]} != {tx}'
        mask = np.meshgrid(common_indices, common_indices)
        x_g, y_g = np.meshgrid(tx, ty)
        # Don't use the bins with nan values
        corr_grid = corr_2d[mask[0], mask[1]].flatten()
        nan_mask = np.isnan(corr_grid).flatten()
        x_g = x_g.flatten()[~nan_mask]
        y_g = y_g.flatten()[~nan_mask]
        corr_grid = corr_grid[~nan_mask]
        assert len(x_g.flatten()) == len(corr_grid), f'The lengths do not match, {len(x_g.flatten())} != {len(corr_grid)}'
        # The minimum number of points required for a spline fit
        if x_g.size < (3+1)*(3+1):
            spline = None
        else:
            spline = LSQBivariateSpline(x_g.flatten(), y_g.flatten(), corr_grid, tx, ty, kx=3, ky=3)
            if allow_scipy:
                spline = bisplrep(x_g.flatten(), y_g.flatten(), corr_grid, kx=3, ky=3, s=1e-3)
        return r0, tx, mass_bins, corr_2d, spline

    def _sim_fit_spline_r(self, sim_tag, s=1e-3):
        """
        Fit a univariate spline to the xi(r) at fixed n1, n2
        """
        _, _, corr_sim, mass_pairs = self._load_data(sim_tag)
        # Replace the nan bins with a small negative number
        # And give a weight of zero to these bins while fitting
        corr_sim = np.log10(corr_sim)
        ind = np.isnan(corr_sim)
        self.logger.debug(f'Found {100*ind.sum()/corr_sim.size:.1f} % of xi(r,n1,n2) is nan')
        rbins = self.rbins
        
        # The fixed knots for the spline
        #tx = np.array([rbins[0], rbins[0], rbins[0], rbins[0], 1., 1.2, 1.5])
        #tx = np.append(tx, np.logspace(np.log10(2), np.log10(60), 3)[1:])
        #tx = np.append(tx, np.linspace(60, rbins[-1], 5)[1:])
        #tx = np.append(tx, [rbins[-1]+1, rbins[-1]+1, rbins[-1]+1])
        #assert np.all(np.sort(tx) == tx), 'The knots are not sorted'
        
        # The smapling points
        xg = np.array([0.1, 0.3, 0.5, 1, 1.2, 1.5, 2])
        xg = np.append(xg, np.logspace(np.log10(2), np.log10(10), 5)[1:])
        xg = np.append(xg, np.logspace(np.log10(10), np.log10(60), 5)[1:])
        xg = np.append(xg, np.linspace(60, rbins[-1], 5)[1:])
        xg = np.append(xg, rbins[-1])
        # Find the closest indices to the grid points
        closest_indices = np.searchsorted(rbins, xg)
        closest_indices = np.clip(closest_indices, 1, len(rbins) - 1)
        closest_indices = closest_indices - (np.abs(rbins[closest_indices - 1] - xg) < np.abs(rbins[closest_indices] - xg))
        xg  = rbins[closest_indices]

        xg = np.log10(xg)
        # Same weights for all the points
        w = np.ones_like(xg)
        w[xg < 0.5] = 0.5
        # We only use the data on the grid points
        corr_grid = corr_sim[:,closest_indices]
        # Remove the grids with no data
        nan_mask = np.isnan(corr_grid)

        all_splines = []
        for i in range(corr_sim.shape[0]):
            # Don't fit the spline if there are less than 20 (vs 40 in total) points
            if len(xg[~nan_mask[i]]) < 20:  # Need at least k+1 knots for degree k=3
                spline = None
            else:
                w =  w[~nan_mask[i]]
                spline = make_splrep(xg[~nan_mask[i]], corr_grid[i][~nan_mask[i]], w=w, k=3, s=s)
                self.logger.debug(spline.c.size)
                #spline = UnivariateSpline(xg[~nan_mask[i]], corr_grid[i][~nan_mask[i]], k=3, s=1e-4)
                #self.logger.debug(spline.get_coeffs().size)
            all_splines.append(spline)
        return xg, corr_sim, all_splines

    def spline_nan_interp(self, mass_pair, spline_s=1e-2, rcut=(0.2, 61), alpha_bad=0.35):
        """
        Fit univariate spline to xi(r) for a specific mass pair, handling NaN values.
        
        Parameters
        ----------
        mass_pair : tuple
            Mass bin pair (m1, m2) to analyze
        spline_s : float, default=1e-2
            Spline smoothing parameter
        rcut : tuple, default=(0.2, 61)
            Min/max radius range in Mpc/h
        alpha_bad : float, default=0.35
            Threshold fraction of valid points required
            
        Returns
        -------
        rbins : ndarray
            Radial bins within rcut range (Mpc/h)
        log_corr : ndarray
            Log10 of correlation function values
        eval_splines : ndarray
            Spline evaluations at each rbin, it is interpolated log10(xi(r))
        bad_sims_mask : ndarray
            Mask of simulations with a fraction of less than alpha_bad valid points
        """
        ind_r = np.where((self.rbins > rcut[0]) & (self.rbins < rcut[1]))[0]
        ind_m = np.where((self.mass_pairs[:,0] == mass_pair[0]) & (self.mass_pairs[:,1] == mass_pair[1]))[0]
        rbins = self.rbins[ind_r]
        log_corr = np.log10(self.xi[:,ind_m,ind_r]).squeeze()
        # Find the nan bins
        nan_mask = np.isnan(log_corr)
        self.logger.debug(f'Found {100*nan_mask.sum()/log_corr.size:.1f} % of xi(r,n1,n2) is nan')
         
        # Same weights for all the points
        w = np.ones_like(rbins)
        # Except for the bins with r < 1
        w[rbins < 1] = rbins[rbins < 1]**2
        eval_splines = np.full(log_corr.shape, np.nan)
        # Store the sims with non-NaN values less than 10
        bad_sims_mask = np.zeros((log_corr.shape[0],), dtype=bool)
        for s in range(log_corr.shape[0]):
            # Don't fit the spline if there are less 
            # than 40% of the points available
            if len(rbins[~nan_mask[s]]) < alpha_bad*len(rbins):
                spline = None
                bad_sims_mask[s] = True
            else:
                # Do not use the NaN bins
                w_cleaned =  w[~nan_mask[s]]
                spline = make_splrep(rbins[~nan_mask[s]], log_corr[s][~nan_mask[s]], w=w_cleaned, k=3, s=spline_s)
                #self.logger.debug(f'spline coeff.size : {spline.c.size}')
                #spline = UnivariateSpline(xg[~nan_mask[i]], log_corr_grid[i][~nan_mask[i]], k=3, s=1e-4)
                #self.logger.debug(spline.get_coeffs().size)
                eval_splines[s] = spline(rbins)
            #all_splines.append(spline)
        self.logger.debug(f'{self.fid}, narrow= {self.narrow}, Found {bad_sims_mask.sum()} sims with less than {100*alpha_bad:.0f}% of valid data points')
        return rbins, log_corr, eval_splines, bad_sims_mask

    def hard_coded_good_l2_sims(self):
        """
        Following on https://github.com/qezlou/private-gal-emu/issues/39#issuecomment-3226453844
        there are ~ 150 L2-W sims that miss mroe than 25% of their xi(r) values for
        the largest mass bins, this is the list of those sims
        """
        if self.fid == 'L2' and self.narrow == False:
            bad_sims = np.array([  4,   6,   9,  16,  18,  21,  33,  36,  48,  51,  60,  67,  70,
            76,  79,  81,  91,  99, 102, 103, 106, 107, 114, 117, 118, 123,
            132, 153, 159, 163, 165, 168, 171, 179, 180, 183, 186, 194, 196,
            198, 205, 217, 220, 227, 230, 235, 238, 250, 254, 262, 265, 268,
            271, 274, 280, 283, 286, 289, 292, 295, 296, 298, 310, 316, 317,
            326, 328, 334, 337, 343, 346, 350, 352, 358, 364, 367, 368, 385,
            388, 394, 397, 400, 410, 413, 415, 424, 430, 436, 442, 446, 448,
            451, 454, 456, 465, 471, 474, 489, 492, 498, 507, 510, 511, 517,
            519, 522, 527, 536, 537, 543, 544, 549, 553, 556, 561])
            good_sims = np.setdiff1d(np.arange(self.xi.shape[0]), bad_sims)
        else:
            good_sims = np.arange(self.xi.shape[0])
        return good_sims

    def minimal_spline_interp(self, good_sims, ind_r):
        """
        Use Cubic spline only for recovering random NaN values.
        No extrapolation is done here, if the NaN values are at the edges,
        they will be replaced by a small value of xi = 1e-4 and a very large
        uncertainty of 1e6.
        Parameters:
        --------------
        good_sims: np.ndarray
            The indices of the simulations to consider. Simulation with too many NaN
            values should be removed beforehand. A list is provided in `hard_coded_good_l2_sims()`
            which is curated for highest mass bin and is applicable for lower mass bins as well.
        ind_r: np.ndarray
            The indices of the r values to consider
        Returns:
        --------------
        interpolated_values: np.ndarray, shape=(len(good_sims), n_mass_pairs, len(ind_r))
            The xi(r, n1, n2) values for the good simulations with NaN values replaced by
            interpolated values or a small value of 1e-4.
        corr_err: np.ndarray, shape=(len(good_sims), n_mass_pairs, len(ind_r))
            The uncertainty estimates for the interpolated values
        """
        interpolated_values = np.ones((good_sims.size, self.xi.shape[1], ind_r.size)) * 1e-4
        log_corr_err = np.zeros_like(interpolated_values)
        rbins = self.rbins[ind_r]
        for i, s in enumerate(good_sims):
            # loop over mass pairs
            for j in range(self.xi.shape[1]):
                corr = self.xi[s,j,ind_r]
                # Create a cubic spline interpolator from valid data points
                ind_non_nan = np.where(~np.isnan(np.log10(corr)))[0]
                if len(ind_non_nan) == corr.size:
                    interpolated_values[i,j] = corr
                elif len(ind_non_nan) <= 2:
                    # If there are not enough points for interpolation, use a constant value
                    interpolated_values[i,j] = 1e-4
                    log_corr_err[i,j] = 20
                else:
                    interp = CubicSpline(np.log10(rbins)[ind_non_nan], 
                                         np.log10(corr)[ind_non_nan], 
                                         extrapolate=False, axis=-1)
                    # Interpolate new values for these problematic points
                    this_interp_val = interp(np.log10(rbins))
                    # Extrapolated values will be NaN, replace them with a small value
                    this_interp_val[np.isnan(this_interp_val)] = -4
                    interpolated_values[i,j] = 10**this_interp_val
                    log_corr_err[i,j][np.isnan(this_interp_val)] = 20
            
        return interpolated_values, log_corr_err


    def remove_sims(self, sim_id):
        """
        get the indices to the simulations to remove
        Parameters:
        -----------
        sim_id: list
            List of simulation IDs to remove
        Returns:
        --------
        keep_sims: np.ndarray
            Indices of the simulations to keep
        """
        ind_bad_sims = []
        sim_id = [str(s).rjust(4, '0') for s in sim_id]

        for i, s in enumerate(self.sim_tags):
            for r in sim_id:
                if r in s:
                    ind_bad_sims.append(i)
        keep_sims = np.setdiff1d(np.arange(len(self.sim_tags)), np.array(ind_bad_sims))
        return keep_sims   

    def get_wt_err_interped_deprecated(self, rcut=(0.2, 61), good_sims=False):
        """
        Parameters
        --------------
        rcut: tuple
            The range of r values to consider for fitting
        good_sims: bool
            Whether to use only the good simulations. The good
            simulations are defined as those with a low fraction of bad points
            which is hard-coded in the method `hard_coded_good_l2_sims()`
            for now. This removes ~ 150 L2 simulations.
        Returns
        --------------
        bins: np.ndarray, shape=(n,3)
            The bins of the correlation function in (r, m1, m2) format
        interped_log_corr: np.ndarray, shape=(n*n_bins,)
            The correlation function for the "remaining" bins
            at (r, m1, m2) bins. Replacing the missing values with 
            a minimal spline interpolation
        corr_err: np.ndarray, shape=(n*n_bins)
            The correlation function errors for the "remaining" bins
            at (r, m1, m2) bins
        params: np.ndarray, shape=(n*n_bins, prams_size)
            The cosmological parameters for the "remaining" bins
        """
        self.logger.info('Using reduced set of sims and minimal spline interpolation for NaN values')
        ind_r = np.where((self.rbins > rcut[0]) & (self.rbins < rcut[1]))[0]
        if good_sims:
            good_sims = self.hard_coded_good_l2_sims()
        else:
            good_sims = np.arange(self.xi.shape[0])
        
        interped_log_corr, log_corr_err = np.log10(self.minimal_spline_interp(good_sims, ind_r))
        interped_log_corr = interped_log_corr.reshape((interped_log_corr.shape[0], 
                                                     interped_log_corr.shape[1]*interped_log_corr.shape[2]))
        log_corr_err = log_corr_err.reshape((log_corr_err.shape[0], log_corr_err.shape[1]*log_corr_err.shape[2]))

        params = self.get_params_array()[good_sims]
        sim_tags = self.sim_tags[good_sims]
        rbins = self.rbins[ind_r]
        # Record the bins of ((m1, m2), r)
        bins = np.zeros((interped_log_corr.shape[1], 3))
        bins[:,2] = np.tile(rbins, len(self.mass_pairs))
        bins[:,1] = np.repeat(self.mass_pairs[:,1], rbins.size)
        bins[:,0] = np.repeat(self.mass_pairs[:,0], rbins.size)

        return bins, interped_log_corr, log_corr_err, params, sim_tags
    
    def get_wt_err(self, rcut=(0.2, 61), remove_sims=None):
        """
        Return the correlation function for all the simulations
        and the corresponding uncertainties. NOT: For now we assign very large
        uncertainties to the bins with NaN values and simulations missing more than
        alpha_bad fraction of the r-bins for some  massive mass pairs.
        ToDo: Use Jacknife resampling to get the uncertainties
        Parameters
        --------------
        rcut: tuple
            The range of r values to consider for fitting
        alpha_bad: float
            The fraction of bad points to consider
        Returns
        --------------
        bins: np.ndarray, shape=(n,3)
            The bins of the correlation function in (r, m1, m2) format
        corr: np.ndarray, shape=(n*n_bins,)
            The correlation function for the "remaining" bins
            at (r, m1, m2) bins
        corr_err: np.ndarray, shape=(n*n_bins)
            The correlation function errors for the "remaining" bins
            at (r, m1, m2) bins
        params: np.ndarray, shape=(n*n_bins, prams_size)
            The cosmological parameters for the "remaining" bins

        """
        ind_r = np.where((self.rbins > rcut[0]) & (self.rbins < rcut[1]))[0]
        rbins = self.rbins[ind_r]
        log_corr = np.log10(self.xi[:,:,ind_r]).squeeze()
        log_corr = log_corr.reshape((log_corr.shape[0], log_corr.shape[1]*log_corr.shape[2]))
        # Find the nan bins
        nan_mask = np.isnan(log_corr)
        self.logger.info(f'Found {100*nan_mask.sum()/log_corr.size:.1f} % of xi(r,n1,n2) is nan')
        # For now we assign -1 for the missing bins
        log_corr[nan_mask] = -1


        # Remove the mass_pairs of some sims that are missing more than alpha_bad
        # fraction of the rbins
        #ind = np.where(np.sum(nan_mask, axis=2) > alpha_bad*log_corr.shape[2])
        #print(ind)
        #nan_mask[ind,:] = True

        corr_err = np.zeros_like(log_corr)
        corr_err[nan_mask] = 1e6

        params = self.get_params_array()
        
        # Record the bins of ((m1, m2), r)
        bins = np.zeros((log_corr.shape[1], 3))
        bins[:,2] = np.tile(rbins, len(self.mass_pairs))
        bins[:,1] = np.repeat(self.mass_pairs[:,1], rbins.size)
        bins[:,0] = np.repeat(self.mass_pairs[:,0], rbins.size)


        ## Get the simulation labels for each data point
        #sim_labels = np.repeat(self.sim_tags, log_corr.shape[1])

        sim_tags = self.sim_tags

        if remove_sims is not None:
            keep_sims = self.remove_sims(remove_sims)
            log_corr = log_corr[keep_sims]
            corr_err = corr_err[keep_sims]
            params = params[keep_sims]
            sim_tags = self.sim_tags[keep_sims]

        return bins, log_corr, corr_err, params, sim_tags
    
    def unconcatenate(self, corr, bins):
        """
        Unconcatenate the large xi arrays used for the emulator
        Parameters
        --------------
        corr: np.ndarray, shape=(n_sims, n_bins)
            The correlation function
        bins: np.ndarray, shape=(n_bins, 3)
            The bins of the correlation function in (m1, m2, r) format
        Returns
        --------------
        corr: np.ndarray, shape=(n_sims, len(mass_pairs), len(rbins))
            The correlation function for each mass pair
        """
        return corr.reshape(-1, len(self.mass_pairs), len(np.unique(bins[:,2])))

    def _normalize(self, X=None,Y=None):
        """
        Normalzie the data for GP fitting
        Parameters:
        --------------
        X: np.ndarray, shape=(n,)
            The input data
        Y: np.ndarray, shape=(n,)
            The output data
        Returns:
        --------------
        mean_func: np.ndarray
            The mean function of the data
        X: np.ndarray
            The normalized input data
        X_min, X_max: float
            The min and max of the input data
        """
        mean_func, X_min, X_max = None, None, None
        if X is not None:
            X_min, X_max = tf.reduce_min(X), tf.reduce_max(X)
            X = (X - X_min) / (X_max - X_min)
        if Y is not None:
            medind = np.argsort(Y)[np.shape(Y)[0]//2]
            mean_func = Y[medind]
        return mean_func, X, X_min, X_max

    def _do_gp(self, X, Y):
        """
        Helper to fit a GP on one dataset
        Parameters:
        --------------
        X: np.ndarray
            The input data
        Y: np.ndarray
            The output data
        Returns:
        --------------
        model: gpflow.models.GPR
            The trained GP model
        opt_logs: dict
            The optimization logs
        """
        # Mean func of the GP
        ind_nan = np.isnan(Y)
        mean_func, _, _, _ = self._normalize(Y=Y)
        dtype = mean_func.dtype
        
        # simple mean of Y
        class MeanFunction(gpflow.mean_functions.MeanFunction):
            def __call__(self, X):
                return mean_func * tf.ones((X.shape[0], 1), dtype=dtype)

        ## 3 GPs with chagepoints at 1, 10 cMpc/h
        kernel_list = [gpflow.kernels.RBF() for _ in range(3)]
        compound_kernel = gpflow.kernels.ChangePoints(kernel_list, locations=np.log10([1, 10]))
        model = gpflow.models.VGP(data=(X[:, np.newaxis], Y[:,np.newaxis]), kernel=compound_kernel, mean_function=MeanFunction())
        gpflow.set_trainable(model.kernel.locations, False)
        ## Optimize
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        return model, opt_logs

    
    def _sim_fit_gp_r(self, sim_tag, r_range=(0.1, np.inf)):
        """
        Fit a GP to the xi(r) at fixed n1, n2. For one simulation
        with tags=`sim_tag`
        Parameters:
        --------------
        sim_tag: str
            The simulation tag
        r_range: tuple
            The range of r values to consider for fitting
        Returns:
        --------------
        rbins: np.ndarray
            The r values used for the fit
            units: Mpc/h
        corrs: np.ndarray
            The xi(r) values for the fit at 
            each mass pair
        all_models: list
            List of trained GP models for each mass pair
        """
        _, _, corrs, mass_pairs = self._load_data(sim_tag)
        ind_rbins = np.where((self.rbins > r_range[0]) & (self.rbins < r_range[1]))[0]
        rbins = np.copy(self.rbins[ind_rbins])
        corrs = np.log10(corrs[:,ind_rbins])
        rbins = np.log10(rbins)
        _, X, X_min, X_max = self._normalize(X=rbins)
        all_models = [] # store the GPs per mass pair
        for b in range(3):
            if b%10 == 0:
                self.logger.info(f'Fitting GP, prog {b}/{corrs.shape[0]}')
            Y = corrs[b]
            model, opt_logs = self._do_gp(X, Y)
            if opt_logs['success'] == False:
                self.logger.warning(f'Optimization did not converge for {sim_tag} at mass pair {mass_pairs[b]}')
                all_models.append(None)
            else:
                all_models.append(model)
            
        return 10**rbins, 10**corrs, all_models



    
class HMF(BaseSummaryStats):
    """
    Halo mass function
    """
    def __init__(self, data_dir, fid, z=2.5, mass_range=(11.1, 12.5), narrow=False, no_merge=True, chi2=False, logging_level='INFO'):
        super().__init__(data_dir, logging_level)
        self.fid = fid
        self.z = np.round(z, 1)
        if fid == 'HF':
            self.vbox = 1000**3  # Volume of the box in Mpc^3/h^3
        elif fid == 'L2':
            self.vbox = 250**3  
        self.no_merge = no_merge
        self.sim_tags = None
        self.narrow = narrow
        self.logging_level = logging_level
        self.knots = None
        self.chi2 = chi2
        self.mass_range = mass_range

    
    def get_labels(self):
        """It is just the simulation tags"""
        if self.sim_tags is None:
            raise ValueError('The simulation tags are not loaded yet, call `load()` first')
        return self.sim_tags

    def load(self):
        """
        Load the Halo Mass Function computed for simulations and saved on `data_dir`
        """
        if self.narrow:
            if self.no_merge:
                save_file = f'{self.fid}_hmfs_{self.z}_narrow_no_merge.hdf5'
            else:
                save_file = f'{self.fid}_hmfs_narrow.hdf5'
        else:
            if self.no_merge:
                save_file = f'{self.fid}_hmfs_{self.z}_no_merge.hdf5'
            else:
                save_file = f'{self.fid}_hmfs.hdf5'
        if self.rank==0:
            self.logger.debug(f'Loading HMFs from {save_file}')   
            with h5py.File(op.join(self.data_dir, 'HMF', save_file), 'r') as f:
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
        if self.comm is not None:
            self.comm.barrier()
            sim_tags = self.comm.bcast(sim_tags)
            bad_sims = self.comm.bcast(bad_sims)
            bins = self.comm.bcast(bins)
            hmfs = self.comm.bcast(hmfs)
        self.sim_tags = sim_tags
        self.bad_sims = bad_sims
        self.logger.debug(f'Loaded HMFs from {save_file}')
        # Only use bins within thr self.mass_range
        masked_mbins = []
        masked_hmfs = []
        for i in range(len(bins)):
            mbin = 0.5*(bins[i][1:] + bins[i][:-1])
            h = hmfs[i]
            ind = np.where((mbin >= self.mass_range[0]) & (mbin <= self.mass_range[1]))[0]
            masked_mbins.append(mbin[ind])
            masked_hmfs.append(h[ind])
        self.logger.info(f'loading from {save_file}')
        return masked_hmfs, masked_mbins

    def get_wt_err(self):
        """
        Return the halo mass function for all the simulations
        and the corresponding uncertainties. NOT: For now we assign very large
        uncertainties to the bins with insufficient counts.
        ToDo: Use Jacknife resampling to get the uncertainties
        Parameters
        --------------
        Returns
        --------------

        """
        hmfs, mbins = self.load()
        full_bins = np.arange(self.mass_range[0], self.mass_range[1]+0.01, 0.1)
        log_hmfs = -7*np.ones((len(hmfs), full_bins.size-1))
        log_hmf_errs = np.ones_like(log_hmfs)*5
        for i, (h, b) in enumerate(zip(hmfs, mbins)):
            ind = np.digitize(mbins[i], full_bins)-1
            log_hmfs[i,ind] = np.log10(h)
            log_hmf_errs[i,ind] = self._get_errs(log_hmf=np.log10(h), mbins=b)
        del hmfs
        del mbins
        full_mbins = 0.5*(full_bins[1:] + full_bins[:-1])
        return full_mbins, log_hmfs, log_hmf_errs, self.get_params_array(), np.array(self.sim_tags)

    def _get_errs(self, log_hmf, mbins):
        """
        Get realtistic errorbars for the input halo mass function
        \Delta log10(f(M)) = \sqrt{1/N_h} + higer order terms
        where f(M) is the halo mass function in units of 1/h^3 Mpc^3 / dex

        Where $N_h$ is the number of halos in the bin:
        N_h = f(M) * \Delta log10(M) * V_{box}

        Parameters:
        --------------
        hmf: np.ndarray
            The halo mass function to compute the errors for
        mbins: np.ndarray
            The mass bins corresponding to the hmf
        Returns:
        --------------
        errs: np.ndarray
            The errors for the hmf
        """
        # The width of the mass bins in log10(M)
        dlogM = np.diff(mbins)[0]
        # The number of halos in the bin
        N_h = 10**log_hmf * dlogM * self.vbox
        # The errors in log10(f(M))
        errs = np.sqrt(1/N_h)
        # Avoid division by zero
        errs[N_h == 0] = 1e2
        return errs

    def _do_fits(self, ind=None, delta_r=None, *kwargs):
        """
        Fit the halo mass function with a splie.
        Parameters:
        --------------
        kwargs: dict
            Keyword arguments for utils.ConstrainedSplineFitter
        """
        hmfs, mbins = self.load()
        # We fix the knots for the spline fit
        #self.knots = np.array([11.1 , 11.1, 11.1,
        #                  11.35, 11.6 , 11.85, 
        #                  12.1 , 12.35, 12.6 , 
        #                  12.85, 13.1 , 13.35,
        #                  13.35, 13.35])
        if delta_r is None:
            delta_r = 0.1
        self.knots = np.array([11.1 , 11.1])
        self.knots = np.append(self.knots, np.arange(11.1, 13.5, delta_r))
        self.knots = np.append(self.knots, [13.5, 13.5])
        if ind is None:
            ind = np.arange(len(hmfs))
        fit = utils.ConstrainedSplineFitter(*kwargs, logging_level=self.logging_level)
        splines = []
        for i in ind:
            if self.chi2:
                sigma = np.log10(hmfs[i][0]/hmfs[i][:])
            else:
                sigma = np.ones_like(hmfs[i])
            splines.append(fit.fit_spline(mbins, np.log10(hmfs[i]), self.knots, sigma=sigma))
        return splines
    
    def get_coeffs(self, ind=None, delta_r=None, *kwargs):
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
        splines = self._do_fits(ind=ind, delta_r=delta_r, *kwargs)
        coeffs = np.zeros((len(splines), len(splines[0].c)))
        for i, spl in enumerate(splines):
            coeffs[i] = spl.c
        return coeffs, splines[0].t
    
    def get_smoothed(self, x, delta_r=None, ind=None, *kwargs):
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
        splines = self._do_fits(ind=ind, delta_r=delta_r, *kwargs)
        y = []
        for i, spl in enumerate(splines):
            if type(x) == list:
                eval_points = x[i]
            else:
                eval_points = x
            # The Spline fit was done in log space
            y.append(10**BSpline(spl.t, spl.c, spl.k)(eval_points)) 
        return y
    
    def get_pca(self, x, n_components=4):
        y = np.array(self.get_smoothed(x))
        # Standardize the data
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(np.log10(y))

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(y_scaled)
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        return eigenvalues, eigenvectors, y

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