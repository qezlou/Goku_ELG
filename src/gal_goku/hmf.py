import os
import argparse
import json
from glob import glob
import numpy as np
import re
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit import CurrentMPIComm
from nbodykit.hod import Zheng07Model
from nbodykit.lab import *
from nbodykit.cosmology import Planck15 as cosmo
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
            setup_logging('info')
        except ImportError:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if self.rank==0:
            logger.info('Logger initialized at level: %s', logging.getLevelName(logging_level))
            logger.info(f'MPI_COMM_WORLD | size = {self.size} -- Nbkit COMM | size = {self.nbkit_size}')
        
        return logger
    
    def get_fof_hmf(self, pig_dir, vol,  bins):
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
        hist = np.histogram(np.log10(halos['Mass']).compute(), bins=bins)
        mass = 0.5*(hist[1][1:]+hist[1][:-1])
        hmf = hist[0]/(vol*(bins[1]-bins[0]))
        return hmf
    
    def get_all_fof_hmfs(self, base_dir, save_file, bins=None, z=2.5):
        """iterate over all avaiable pigs in base_dir and compue the halo mas function"""
        pigs = self.get_pig_dirs(base_dir, z=z)
        num_sims = len(pigs['sim_tags'])
        if bins is None:
            bins = np.arange(11, 13.5, 0.1)
        hmfs = np.zeros((num_sims, bins.size-1))
        # We use CubicSpline to refine the mass bins
        # The mass resolution we need is 0.05 dex
        mbins = 0.5*(bins[1:]+bins[:-1])
        bins_fine = np.arange(11, 13.5, 0.05)
        mbins_refined = 0.5*(bins_fine[1:]+bins_fine[:-1])
        hmfs_interp = np.zeros((hmfs.shape[0], mbins_refined.size))
        hmfs_fine = np.zeros((hmfs.shape[0], mbins_refined.size))
        
        for i in range(num_sims):
            vol = pigs['params'][i]['box']**3
            hmfs[i] = self.get_fof_hmf(pigs['pig_dirs'][i], vol=vol, bins=bins)
            hmfs_fine[i] = self.get_fof_hmf(pigs['pig_dirs'][i], vol=vol, bins=bins_fine)
            # We use CubicSpline to refine the mass bins
            spl = ius(mbins, hmfs[i], k=3)
            hmfs_interp[i] = spl(mbins_refined)
        
        with h5py.File(save_file, 'w') as fw:
            fw['sim_tags'] = pigs['sim_tags']
            fw['hmfs_coarse'] = hmfs
            fw['bins_coarse'] = bins
            fw['hmfs_fine'] = hmfs_fine
            fw['hmfs'] = hmfs_interp
            fw['bins'] = bins_fine