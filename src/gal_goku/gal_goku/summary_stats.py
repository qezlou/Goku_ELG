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
from scipy.interpolate import BSpline, LSQBivariateSpline, make_lsq_spline, make_splrep, UnivariateSpline, bisplrep
from matplotlib import pyplot as plt
from . import utils
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gpflow
from gpflow.utilities import print_summary
import tensorflow as tf
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
        logger = logging.getLogger('summary_stats')
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

class Xi(BaseSummaryStats):
    """
    Speherically Averaged correlation function
    """
    def __init__(self, data_dir, fid, narrow=False, logging_level='INFO'):
        super().__init__(data_dir, fid, logging_level=logging_level)
        self.data_dir = data_dir
        self.fid = fid
        self.narrow = narrow
        # Get the file counts
        if self.narrow:
            raise NotImplementedError('Narrow not implemented yet')
        else:
            # Each sim is an indivudal hdf5 file in op.join(data_dir, fid)
            self.sim_tags = [f[:-5] for f in os.listdir(op.join(self.data_dir, self.fid)) if self.pref in f and f.endswith('.hdf5')]
            self.logger.info(f'Total sims files: {len(self.sim_tags)} in {op.join(self.data_dir, self.fid)}')
        
        _, self.rbins, _, mass_pairs = self._load_data(self.sim_tags[0])
        self.mass_pairs = np.around(mass_pairs, 2)
        self.mass_bins = np.round(np.unique(mass_pairs), 2)


    def _load_data(self, sim_tag, r_cut=0.2):
        """
        Load the correlation function for a single simulation
        """
        corr_file = op.join(self.data_dir, self.fid, f'{sim_tag}.hdf5')
        with h5py.File(corr_file, 'r') as f:
            sim_tag = f['sim_tag'][()]
            rbins = f['mbins'][:]
            xi = f['corr'][:]
            mass_pairs = f['pairs'][:]
            if r_cut is not None:
                ind = np.where(rbins > r_cut)[0]
                rbins = rbins[ind]
                xi = xi[:,ind]
        return sim_tag, rbins, xi, mass_pairs
    
    def load_all_data(self):
        """
        Load all the correlation functions in original form
        """
        all_corrs = []
        for sim_tag in self.sim_tags:
            _, _, corr, _ = self._load_data(sim_tag)
            all_corrs.append(corr)
        return np.array(all_corrs)

    def make_3d_corr(self, corr, symmetric=False):
        """
        Pass xi(n_mass_pairs, n_rbins) and get a
        3D array (n_mass_pairs, n_mass_pairs, n_rbins)
        """
        mass_bins = self.mass_bins[::-1]
        ind_m_pair = np.digitize(self.mass_pairs, mass_bins).astype(int)
        corr_3d = np.full((mass_bins.size, mass_bins.size, self.rbins.size), np.nan)
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
        for sim_tag in self.sim_tags:
            corr = self._xi_sim_n1_n2_r(sim_tag)
            all_corrs.append(corr)
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
    def __init__(self, data_dir, fid, narrow=False, no_merge=True, chi2=False, logging_level='INFO'):
        super().__init__(data_dir, logging_level)
        self.fid = fid
        self.no_merge = no_merge
        self.sim_tags = None
        self.narrow = narrow
        self.logging_level = logging_level
        self.knots = None
        self.chi2 = chi2

    
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
                save_file = f'{self.fid}_hmfs_narrow_no_merge.hdf5'
            else:
                save_file = f'{self.fid}_hmfs_narrow.hdf5'
        else:
            if self.no_merge:
                save_file = f'{self.fid}_hmfs_no_merge.hdf5'
            else:
                save_file = f'{self.fid}_hmfs.hdf5'
        if rank==0:
            self.logger.debug(f'Loading HMFs from {save_file}')   
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
        self.logger.debug(f'Loaded HMFs from {save_file}') 
        return hmfs, bins

    def _do_fits(self, ind=None, delta_r=None, *kwargs):
        """
        Fit the halo mass function with a splie.
        Parameters:
        --------------
        kwargs: dict
            Keyword arguments for utils.ConstrainedSplineFitter
        """
        hmfs, bins = self.load()
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
            mbins = 0.5*(bins[i][1:] + bins[i][:-1])
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