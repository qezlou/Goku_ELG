"""
A sript to be run opn bash to get hte galaxy correlation function of Goku sims
It uses the HOD models which are either standard or I have implemented in `nbodykit`.
"""
import os
import argparse
import json
import numpy as np

from nbodykit import CurrentMPIComm
from nbodykit.hod import Zheng07Model
from nbodykit.lab import *
from nbodykit.cosmology import Planck15 as cosmo
import h5py
import logging
import warnings
warnings.filterwarnings("ignore")

comm = CurrentMPIComm.get()
rank = comm.Get_rank()
size = comm.Get_size()

class Corr():
    def __init__(self, logging_level='INFO'):
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
        if rank==0:
            logger.debug('Logger initialized at level: %s', logging.getLevelName(logging_level))
        
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
        import os
        sim_tags = [t for t in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, t))]
        sim_dirs = [os.path.join(base_dir, t) for t in sim_tags]
        pigs = {'sim_tags':sim_tags, 'pig_dirs':[], 'params':[]}
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
        
            # Apply RSD to the galaxies
            hod['RSDPosition'] = hod['Position'] + hod['VelocityOffset'] * los
            corr_gal_zspace = SimulationBox2PCF(data1=hod, mode=corr_mode, edges=r_edges,  position='RSDPosition')
            corr_gal_zspace.run()
            all_corrs[i] = corr_gal_zspace.corr['corr'][:]
        return all_corrs

    def fix_hod_all_pigs(self, base_dir, seeds=[42], z=2.5, savedir = '/work2/06536/qezlou/Goku/corr_funcs'):
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
        r_edges = np.logspace(0, np.log10(200), 100)
        for i, pig_dir in enumerate(pigs['pig_dirs']):
            if rank==0:
                self.logger.info(f"pig_dir = {pig_dir} | progress = {np.round(i/len(pigs['pig_dirs']), 2)*100} %")
            try:
                corr = self.get_corr(pig_dir, r_edges, seeds=seeds, hod_model=hod_model)
            except ValueError:
                print(f'Issues with {pig_dir}')
                continue
            
            save_file=f'Zheng07_seeds_{pigs["sim_tags"][i]}.hdf5'
            comm.Barrier()
            if rank ==0:
                with h5py.File(os.path.join(savedir, save_file), 'w') as f:
                    f['r'] = ( r_edges[:-1] + r_edges[1:] ) / 2
                    f['corr'] = corr
                    f['seeds'] = seeds
            comm.Barrier()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--base_dir', type=str, help='Base directory of the simulations')
    parser.add_argument('--z', type=float, help='Redshift of the snapshot')
    parser.add_argument('--savedir', type=str, help='Directory to save the correlation functions')
    parser.add_argument('--logging_level', required=str, type=str, default=logging.INFO, help='Logging level')
    args = parser.parse_args()
    if args.logging_level == 'INFO':
        args.logging_level = 20
    elif args.logging_level == 'DEBUG':
        args.logging_level = 10
    corr = Corr(args.logging_level)

    # Get the Corrs for fixed HOD parameters of Zheng+07, but difference seeds
    # for all avaialble cosmologies
    seeds = np.unique(np.random.randint(0, 1000_000, size=100))
    corr.fix_hod_all_pigs(args.base_dir, seeds=seeds, z=args.z)