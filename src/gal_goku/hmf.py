import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import h5py
import logging
import warnings
from . import mpi_helper
from . import get_corr
warnings.filterwarnings("ignore")

class Hmf(get_corr.Corr):
    def __init__(self, logging_level='INFO', ranks_for_nbkit=0):
        super().__init__(logging_level, ranks_for_nbkit)
        self.logger = logging.getLogger('Hmf')
        self.logger.setLevel(logging_level)
    
    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('Hmf')
        logger.setLevel(logging_level)
        try:
            from nbodykit import setup_logging
            setup_logging('warning')
        except ImportError:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if self.rank==0:
            logger.info('Logger initialized at level: %s', logging.getLevelName(logging_level))
            logger.info(f'MPI_COMM_WORLD | size = {self.size} -- Nbkit COMM | size = {self.nbkit_size}')
        
        return logger
    
    def smooth_hmf():
        """
        Fit piecewise polynomial to the halo mass function
        """
    
    def get_fof_hmf(self, pig_dir, vol,  bins, counts_min = 20):
        """
        Plot the halo mass function for the FoF halos
        Parameters:
        -----------
        pig_dir: str
            The directory containing the PIGs
        vol:
            Survey volume in Mpc/h
        bins: Array
            An array of mass bins to compute HMF in
        Returns: Array
            Halo mass function, log10(dn/log(M)) in units of
            dex^-1 hMpc^-1
        """
        halos = self.load_halo_cat(pig_dir)
        counts, bins = np.histogram(np.log10(halos['Mass']).compute(), bins=bins)
        # Combine the last bins which have less than 20 counts
        ind = np.where(counts<counts_min)[0]
        combined_counts = np.sum(counts[ind])
        i = ind[0]
        while combined_counts < counts_min:
            i -= 1
            combined_counts += counts[i]
            ind = np.insert(ind, 0, i)
        self.logger.debug(f'Deleting the last {len(ind)} bins of hmf for {pig_dir}, with counts {counts[ind]}')
        counts = np.delete(counts, ind)
        trimmed_bins = bins[:-len(ind)]
        counts = np.append(counts, combined_counts)

        bins_delta  = trimmed_bins[1::] - trimmed_bins[0:-1]
        print(f'trimmed_bins {trimmed_bins}, bins_delta {bins_delta}')
        hmf = counts/(vol*bins_delta)
        return hmf, trimmed_bins
    
    def get_all_fof_hmfs(self, base_dir, save_file, narrow=False, bins=None, z=2.5):
        """iterate over all avaiable pigs in base_dir and compue the halo mas function"""
        pigs = self.get_pig_dirs(base_dir, z=z, narrow=narrow)
        num_sims = len(pigs['sim_tags'])
        if bins is None:
            bins = np.arange(11, 13.5, 0.1)
        hmfs, trimmed_bins = [], []
        bad_sims = []
        sim_tags = []
        for i in range(num_sims):
            vol = pigs['params'][i]['box']**3
            try:
                h, tbins = self.get_fof_hmf(pigs['pig_dirs'][i], vol=vol, bins=bins)
                hmfs.append(h)
                trimmed_bins.append(tbins)
                sim_tags.append(pigs['sim_tags'][i])
            except FileNotFoundError:
                bad_sims.append(i)
                continue
        
        self.logger.info(f'{len(bad_sims)} sims could not be opened')
        
        with h5py.File(save_file, 'w') as fw:
            dtype = h5py.special_dtype(vlen=hmfs[0].dtype)
            dset = fw.create_dataset('hmfs_coarse', (len(hmfs),), dtype=dtype)
            for i, data in enumerate(hmfs):
                dset[i] = data
            dtype = h5py.special_dtype(vlen=trimmed_bins[0].dtype)
            dset = fw.create_dataset('bins_coarse', (len(trimmed_bins),), dtype=dtype)
            for i, data in enumerate(trimmed_bins):
                dset[i] = data
            fw['sim_tags'] = sim_tags