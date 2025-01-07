import os
import argparse
import json
from glob import glob
import numpy as np
import re

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
    
    def get_all_fof_hmfs(self, base_dir, z=2.5):
        """iterate over all avaiable pigs in base_dir and co,pue the halo mas function"""

        pigs = self.get_pig_dirs(base_dir, z=z)
        