"""
A sript to be run opn bash to get hte galaxy correlation function of Goku sims
It uses the HOD models which are either standard or I have implemented in `nbodykit`.
"""
import os
import argparse
import json
from glob import glob
import numpy as np

from nbodykit import CurrentMPIComm
from nbodykit.hod import Zheng07Model
from nbodykit.lab import *
from nbodykit.cosmology import Planck15 as cosmo
import h5py
import logging
import warnings
import mpi_helper
warnings.filterwarnings("ignore")


class Corr():
    def __init__(self, logging_level='INFO', ranks_for_nbkit=2):
        self.comm = CurrentMPIComm.get()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        # Each 2 rank will be independant Nbodykit Communicator
        # Because coumting the coor for one simulation is not the bottleneck
        # We just have many simulations, so sims should be distributed accross MPI
        # processes
        ranks_for_nbkit = int(ranks_for_nbkit)
        self.color = self.rank//ranks_for_nbkit
        self.nbkit_comm_counts = self.size//ranks_for_nbkit
        self.nbkit_comm = self.comm.Split(self.color, self.rank)
        self.nbkit_rank = self.nbkit_comm.Get_rank()
        self.nbkit_size = self.nbkit_comm.Get_size()
        
        self.logger = self.configure_logging(logging_level)

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('get corr')
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


    def get_pig_dirs(self, base_dir, z=2.5):
        """Get the directories of the PIGs at redshift z
        Parameters
        ----------
        base_dir : str
            base directory of the simulations
        z : float, not an array
            redshift of the snapshot
        Returns
        -------
        pigs : dict
            dictionary containing the PIG directories and the simulation parameters
        """
        sim_tags = [t for t in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, t))]
        sim_dirs = [os.path.join(base_dir, t) for t in sim_tags]
        pigs = {'sim_tags':sim_tags, 'pig_dirs':[], 'params':[]}
        if self.rank == 0:
            self.logger.info(f'base_dir = {base_dir} | number of sims = {len(sim_dirs)}, z = {z}')
        for sd in sim_dirs:
            self.logger.debug(f'Openning sim = {sd}')
            snaps = np.loadtxt(os.path.join(sd, 'output', 'Snapshots.txt'))
            snap_id = snaps[:,0].astype(int)
            snap_z  = 1/snaps[:,1] - 1
            # mask for the redshift
            mask = (snap_z > z-0.1)  * (snap_z < z+0.1)
            pigs['pig_dirs'].append(os.path.join(sd, 'output', f"PIG_{str(snap_id[mask][0]).rjust(3,'0')}"))

            # add the param file
            with open(os.path.join(sd, 'SimulationICs.json'), 'r') as f:
                pigs['params'].append(json.load(f))
        return pigs

    def load_halo_cat(self, pig_dir):
        """
        Load FOF tables as Nbodykit Halocatalog. It will be used to populate them with HOD models.
        Parameters:
        ------------
        pig_dir: str, 
            The path to the PIG directory
        Returns:
        -----------
        Nbodykit HaloCatalog
        """
        cat = BigFileCatalog(pig_dir, dataset='FOFGroups')
        redshift = 1/cat.attrs['Time'] - 1
        cat['Mass'] *= 1e10
        halos = HaloCatalog(cat, 
                            cosmo=cosmo, 
                            redshift=redshift,
                            mdef='vir',
                            mass='Mass',
                            position='MassCenterPosition',
                            velocity='MassCenterVelocity')
        halos.attrs['BoxSize'] /= 1000
        halos['Position'] /= 1000
        return halos

    def populate_halos(self, pig_dir, hod_model, seed=42, model_params={}):
        """
        Populate a PIG (FOF table) with central and satelite galaxies with HOD of choice
        Paramters:
        -----------
        pig_dir: str, 
            The path to the PIG directory
        hod_model: A class from `Nbodykit.hod`
            The models defined in `Nbodykit.hod`
        seed: int,
            The random seed to fix the HOD realization
        model_params:
            The parameters to be passed to the HOD model
        Returns:
        ----------
        hod catalog in `Nbodykit`'s formate
        """
        halos = self.load_halo_cat(pig_dir)
        hod = halos.populate(hod_model, seed=seed, **model_params)
        self.logger.info(f"total number of HOD galaxies = {hod.csize}")

        return hod


    def get_corr(self, pig_dir, r_edges, seeds=[42], corr_mode='1d', hod_model=Zheng07Model, model_params={}):
        """Get the correlation function for HOD populated galaxies in a FOF halo catalog.
        Parameters
        ----------
        pig_dir: str, 
            The path to the PIG directory
        r_egses: array,
            binning along r
        seeds : list/array
            seeds for the HOD model, if len(seeds) > 1, generate different realizations
            of HOD populated galaxies
        corr_mode : str
            mode of the correlation function, either '1d', '2d', 'projected', 'angular'
        hod_model: A class from `Nbodykit.hod`
            The models defined in `Nbodykit.hod`
        Returns:
        ------------
        The correlation function for different seeds
        """
        # in z-space
        los = [0, 0, 1]
        halos = self.load_halo_cat(pig_dir)
            
        all_corrs = np.zeros((seeds.size, len(r_edges)-1))
        for i, sd in enumerate(seeds):
            if i==0:
                hod = halos.populate(hod_model, seed=sd, **model_params)
            else:
                hod.repopulate(seed=sd)
            self.nbkit_comm.Barrier()
            # Apply RSD to the galaxies
            hod['RSDPosition'] = hod['Position'] + hod['VelocityOffset'] * los
            corr_gal_zspace = SimulationBox2PCF(data1=hod, mode=corr_mode, edges=r_edges,  position='RSDPosition')
            corr_gal_zspace.run()
            self.nbkit_comm.Barrier()
            all_corrs[i] = corr_gal_zspace.corr['corr'][:]
        return all_corrs

    def fix_hod_all_pigs(self, base_dir, seeds=[42], z=2.5, savedir = '/work2/06536/qezlou/Goku/corr_funcs_test'):
        """
        Iterate over all FOF Catalogs at redshift z in `base_dir` directory keeping
        HOD paramters the same. We get manny realizations of the HOD populated catalogs.
        Parameters:
        -------------
        base_dir: str,
            The directory containing the FOF catalogs
        z: int,
            The redshift of interest
        savedir:
            The directory to save the correlation functions at
        """
        hod_model = Zheng07Model
        pigs = self.get_pig_dirs(base_dir, z=z)
        keep = self._remove_computed_snaps(pigs, savedir)

        # Decide the binning based on the boxsize
        if 'Box250' in pigs['sim_tags'][0]:
            r_edges = np.logspace(0, np.log10(80), 80)
        else:
            r_edges = np.logspace(0, np.log10(200), 100)
        if self.rank ==0:
            self.logger.info(f'Total existing sims {len(pigs["sim_tags"])}, but only {keep.size} of them are not yet computed')
            self.logger.info(f'r_edges = {r_edges}')
        # Distribute the jobs accross the available tasks, i.e. self.nbkit_comm_counts
        ind_pig_rank = mpi_helper.distribute_array_split_comm(self.nbkit_comm_counts, self.color, keep)
        self.logger.info(f'rank = {self.rank} | color = {self.color} | load = {ind_pig_rank.size}')
        
        for i in ind_pig_rank:
            pig_dir = pigs['pig_dirs'][i]
            try:
                corr = self.get_corr(pig_dir, r_edges, seeds=seeds, hod_model=hod_model)
            except Exception as e:
                if self.rank ==0:
                    self.logger.info(f'Skipping, {pigs["sim_tags"][i]} because {e}')
                continue
            
            save_file=f'Zheng07_seeds_{pigs["sim_tags"][i]}.hdf5'
            save_file = os.path.join(savedir, save_file)
            if self.nbkit_rank ==0:
                self.logger.info(f'save_file = {save_file}')
                with h5py.File(save_file, 'w') as f:
                    f['r'] = ( r_edges[:-1] + r_edges[1:] ) / 2
                    f['corr'] = corr
                    f['seeds'] = seeds
        self.comm.Barrier()
        
    def _remove_computed_snaps(self, pigs, save_dir):
        """
        Remove the simulations from the list
        """
        if self.rank == 0:
            remove = []
            for i in range(len(pigs['sim_tags'])):
                save_file=f'Zheng07_seeds_{pigs["sim_tags"][i]}.hdf5'
                savefile = os.path.join(save_dir, save_file)
                if os.path.exists(savefile):
                    remove.append(i)
            remove = np.array(remove)
            keep = np.array(list(set(np.arange(len(pigs['sim_tags']))) - set(remove)), dtype='i')
            keep = np.ascontiguousarray(keep, dtype='i')
            keep_size = keep.size
        else:
            keep = None
            keep_size= None
        self.comm.Barrier()
        keep_size = self.comm.bcast(keep_size, root=0)
        if self.rank !=0:
            keep = np.empty(keep_size, dtype='i')
        self.comm.Barrier()
        self.comm.Bcast(keep, root=0)
        return keep
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--base_dir', type=str, help='Base directory of the simulations')
    parser.add_argument('--z', type=float, help='Redshift of the snapshot')
    parser.add_argument('--savedir', type=str, help='Directory to save the correlation functions')
    parser.add_argument('--logging_level', required=False, type=str, default=logging.INFO, help='Logging level')
    parser.add_argument('--ranks_for_nbkit', required=False, type=str, default=2, help='number of rank for each nbodykit communicator')
    args = parser.parse_args()
    if args.logging_level == 'INFO':
        args.logging_level = 20
    elif args.logging_level == 'DEBUG':
        args.logging_level = 10
    corr = Corr(args.logging_level, ranks_for_nbkit=args.ranks_for_nbkit)

    # Get the Corrs for fixed HOD parameters of Zheng+07, but difference seeds
    # for all avaialble cosmologies
    seeds = np.unique(np.random.randint(0, 1000_000, size=100))
    with CurrentMPIComm.enter(corr.nbkit_comm):
        corr.fix_hod_all_pigs(args.base_dir, seeds=seeds, z=args.z)