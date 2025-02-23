"""
A sript to be run on bash to get the galaxy correlation function of Goku sims
We do the HOD imn postprocessing
"""
import os
import os.path as op
import argparse
import json
from glob import glob
import numpy as np
import re
import json

from nbodykit import CurrentMPIComm
from nbodykit.hod import Zheng07Model
from nbodykit.lab import *
from nbodykit.cosmology import Planck15 as cosmo
import h5py
import logging
import warnings
from . import mpi_helper
warnings.filterwarnings("ignore")


class Corr():
    def __init__(self, logging_level='INFO', ranks_for_nbkit=0):
        self.comm = CurrentMPIComm.get()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        # Each 2 rank will be independant Nbodykit Communicator
        # Because coumting the coor for one simulation is not the bottleneck
        # We just have many simulations, so sims should be distributed accross MPI
        # processes
        ranks_for_nbkit = int(ranks_for_nbkit)
        if ranks_for_nbkit != 0:
            self.color = self.rank//ranks_for_nbkit
            self.nbkit_comm_counts = self.size//ranks_for_nbkit
            self.nbkit_comm = self.comm.Split(self.color, self.rank)
            self.nbkit_rank = self.nbkit_comm.Get_rank()
            self.nbkit_size = self.nbkit_comm.Get_size()
        else:
            self.color = 0
            self.nbkit_comm_counts = 1
            self.nbkit_comm = self.comm
            self.nbkit_rank = self.rank
            self.nbkit_size = self.size
        
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

    def make_big_ic_file(self, base_dirs = ['/scratch/06536/qezlou/Goku/FOF/HF/',
                                            '/scratch/06536/qezlou/Goku/FOF/L1/',
                                            '/scratch/06536/qezlou/Goku/FOF/L2/',
                                            '/scratch/06536/qezlou/Goku/FOF/L2/narrow/']):
        
        
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

    def get_pig_dirs(self, base_dir, z=2.5, narrow=False):
        """Get the directories of the PIGs at redshift z
        Parameters
        ----------
        base_dir : str
            base directory of the simulations, if `narrow`, the
            narrow sims should be in `/narrow` subdirectory
        z : float, not an array
            redshift of the snapshot
        Returns
        -------
        pigs : dict
            dictionary containing :
            'pig_dirs': The PIG directories 
            'params': The simulation parameters
            'sim_tags': The directory name holding the sims
        """
        if narrow:
            base_dir = os.path.join(base_dir, 'narrow')
        # Find all the directories in `base_dir` that look like a snapshot
        sim_tags = [t for t in os.listdir(base_dir) if ( os.path.isdir(os.path.join(base_dir, t)) and ('Box' in os.path.join(base_dir, t)) )]
        sim_tags = sorted(sim_tags)
        sim_dirs = [os.path.join(base_dir, t) for t in sim_tags]
        if narrow:
            sim_tags_new = [t+'_narrow' for t in sim_tags]
            sim_tags = sim_tags_new
        pigs = {'sim_tags':sim_tags, 'pig_dirs':[], 'params':[]}
        if self.rank == 0:
            self.logger.info(f'base_dir = {base_dir} | number of sims = {len(sim_dirs)}, z = {z}')
        for sd in sim_dirs:
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
    
    def find_pair_pigs(self, base_dirs):
        """
        Get the pair of pigs existing in list of base_dirs, i.e. between L1, L2, HF
        Parameters:
        -------
        base_dirs, list
            List of paths to the FOF directories, i.e.e path to the parent 
            directory holding L1, L2 and HF FOF tables.
        """
        nums= []
        for b in base_dirs:
            pigs = self.get_pig_dirs(b)
            pattern = r'_(\d{4})'
            nums.append([re.search(pattern, path).group(1) for path in pigs['sim_tags']])
        pairs = list(set(nums[0]).intersection(*nums[1:]))
        return pairs
            
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


    def get_power(self, pig_dir, mesh_res=0.25, seeds=[42], mode='1d', hod_model=Zheng07Model, model_params={}):
        """Get the powerspectrum for HOD populated galaxies in a FOF halo catalog.
        Parameters
        ----------
        pig_dir: str, 
            The path to the PIG directory
        r_egses: array,
            binning along r
        seeds : list/array
            seeds for the HOD model, if len(seeds) > 1, generate different realizations
            of HOD populated galaxies
        mode : str
            mode of the powerspectrum, either '1d', '2d', 'projected', 'angular'
        hod_model: A class from `Nbodykit.hod`
            The models defined in `Nbodykit.hod`
        Returns:
        ------------
        The powerspectrum for different seeds
        """
        # in z-space
        los = [0, 0, 1]
        halos = self.load_halo_cat(pig_dir)
        Nmesh = int(halos.attrs['BoxSize'][0] / mesh_res)
            
        all_pks = []
        for i, sd in enumerate(seeds):
            if (i in [10, 25, 50, 75]) and self.nbkit_rank==0:
                self.logger.debug(f'progress in seeds pool {i} %')
            if i==0:
                hod = halos.populate(hod_model, seed=sd, **model_params)
            else:
                hod.repopulate(seed=sd)
            self.nbkit_comm.Barrier()
            # Apply RSD to the galaxies
            hod['RSDPosition'] = (hod['Position'] + hod['VelocityOffset'] * los)%halos.attrs['BoxSize']
            mesh = hod.to_mesh(position='RSDPosition', Nmesh=Nmesh, compensated=True)
            pk_gal_zspace = FFTPower(mesh, mode=mode).power
            self.nbkit_comm.Barrier()
            all_pks.append(pk_gal_zspace['power'].real)
        all_pks = np.array(all_pks)
        k = pk_gal_zspace['k'][:]
        return all_pks, k
    
    def _get_corr(self, pig_dir, mass_th, z=2.5):
        """Get the correlation function for two halo catalogs with 2 mass thresholds
        Parameters
        ----------
        pig_dir: str, 
            The path to the PIG directory
        mass_th: tuple of floats
            The mass threshold for the first and second halo samples
        r_egses: array,
            binning along r
        seeds : list/array
            seeds for the HOD model, if len(seeds) > 1, generate different realizations
            of HOD populated galaxies
        mode : str
            mode of the correlation function, either '1d', '2d', 'projected', 'angular'
        Returns:
        ------------
        The correlation function for different seeds
        """

        # in z-space
        los = [0, 0, 1]
        halos = self.load_halo_cat(pig_dir)
        # Apply RSD to the galaxies
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        halos['RSDPosition'] = (halos['Position'] + halos['Velocity'] * los * rsd_factor)%halos.attrs['BoxSize']

        data1 = halos[halos['Mass'] >= mass_th[0]]
        data2 = halos[halos['Mass'] <= mass_th[1]]
        
        r_edges = np.logspace(np.log10(0.01), np.log10(0.1), 4)
        r_edges = np.append(r_edges, np.logspace(np.log10(0.1), np.log10(2), 15)[1:])
        r_edges = np.append(r_edges, np.logspace(np.log10(2), np.log10(60), 15)[1:])
        r_edges = np.append(r_edges, np.linspace(60, 80, 20)[1:])
        if self.nbkit_rank ==0:
            self.logger.debug(f'r_edges.size = {r_edges.size}')
        corr = SimulationBox2PCF(data1=data1, data2=data2, mode='1d', edges=r_edges,  position='RSDPosition')
        corr.run()
        self.nbkit_comm.Barrier()
        result = corr.corr['corr'][:]
        mbins =  np.array([(r_edges[i]+r_edges[i+1])/2 for i in range(r_edges.size-1)])
        """
            if self.rank == 0:
                self.logger.info(f'Getting power for FOF, mesh_res = {mesh_res} cMpc/h')
            Nmesh = int(halos.attrs['BoxSize'][0] / mesh_res)
            mesh = halos.to_mesh(position='RSDPosition', Nmesh=1000, compensated=True)
            pk_gal_zspace = FFTPower(mesh, mode=mode).power
            self.nbkit_comm.Barrier()
            result = pk_gal_zspace['power'].real
            mbins = pk_gal_zspace['k'].real
        """
            
        return result, mbins

    def _corr_on_grid(self, pig_dir, z=2.5):
        """
        Get the 3D correlation fucntion on a grid with increasing mass thresholds
        """
        mbins = np.arange(13, 10.9,-0.1 )
        idx = np.triu_indices(len(mbins), k=0)
        pairs = np.column_stack((mbins[idx[0]], mbins[idx[1]]))
        corr_hh = []
        for i, m_pair in enumerate(pairs):
            if i in [0, 10, 25, 50, 100, 150, 200]:
                if self.nbkit_rank == 0:
                    self.logger.info(f'progress {100*i/len(pairs)} %')
            corr_fof, mbins = self._get_corr(pig_dir,mass_th=10**m_pair)
            corr_hh.append(corr_fof)
        return np.array(corr_hh), mbins, pairs

    def get_corr_on_grid(self, base_dir, save_dir, chunk, narrow=False, z=2.5, num_chunks=20):
        """
        """
        pigs = self.get_pig_dirs(base_dir, z=z, narrow=narrow)
        num_sims = len(pigs['sim_tags'])
        per_chunk = num_sims//num_chunks
        start = chunk*per_chunk
        if chunk == num_chunks-1:
            end = num_sims
        else:
            end = start + per_chunk

        hmfs, trimmed_bins = [], []
        bad_sims = []
        sim_tags = []
        for i in range(start, end):
            save_file = os.path.join(save_dir, pigs["sim_tags"][i]+'.hdf5')
            if os.path.exists(save_file):
                if self.nbkit_rank ==0:
                    self.logger.info(f'skipping {pigs["sim_tags"][i]} since it is already computed')
            else:
                if self.nbkit_rank==0:
                    self.logger.info(f'working on {pigs["sim_tags"][i]}')
                try:
                    corr_hh, mbins, pairs = self._corr_on_grid(pigs['pig_dirs'][i], z=z)
                    if self.nbkit_rank ==0:
                        self._save_corr_on_grid(corr_hh, mbins, pairs, pigs['sim_tags'][i], save_file)
                    self.nbkit_comm.Barrier()
                except FileNotFoundError as e:
                    self.logger.info(f'{e} for {pigs["pig_dirs"][i]}')
                    bad_sims.append(pigs['sim_tags'][i])
        if self.nbkit_rank ==0:
            self.logger.info(f'{len(bad_sims)} sims could not be opened')

    def _save_corr_on_grid(self, corr_hh, mbins, pairs, sim_tag, save_file):
        self.logger.info(f'Writing on {save_file}')
        with h5py.File(save_file, 'w') as fw:
            fw['corr'] = corr_hh
            fw['mbins'] = mbins
            fw['pairs'] = pairs
            fw['sim_tag'] = sim_tag