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
from nbodykit.cosmology import Cosmology
import h5py
import logging
import warnings
from . import mpi_helper
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
import bigfile


class InitialDensity():
    
    def __init__(self, pig_dir):
        temp_dir = '/'+op.join(op.join(*op.normpath(pig_dir).split(os.sep)[:-2]), 'ICS')
        # Find the directory that ends with '_99'
        all_ic_dirs = [d for d in os.listdir(temp_dir) if d.endswith('_99') and os.path.isdir(os.path.join(temp_dir, d))]
        if all_ic_dirs:
            ic_dir = op.join(temp_dir, all_ic_dirs[0])
        else:
            raise FileNotFoundError("The IC mesh is not found, you need to rerun `MP-GenIC`")
        self.ic_dir = ic_dir
    
    def get_dens_mesh(self, mesh_res=1):    
        """
        Load the initial density field from the IC mesh file.
        Parameters
        ----------
        mesh_res : float, optional
            The resolution of the mesh in Mpc/h, default is 1 Mpc/h
        Returns
        -------
        mesh : Nbodykit Mesh object
        """
        cat = BigFileCatalog(self.ic_dir, dataset='1')
        cat['Position'] /= 1000  # Convert from kpc/h to Mpc/h
        cat.attrs['BoxSize'] /= 1000  # Convert from kpc/h to Mpc/h
        # Create a mesh from the particle catalog
        Nmesh = int(cat.attrs['BoxSize'][0] / mesh_res)
        mesh = cat.to_mesh(Nmesh=Nmesh, BoxSize=cat.attrs['BoxSize'], compensated=True)
        return mesh
    
    def get_power(self, mesh_res=1):
        """
        Get the power spectrum of the initial density field.
        Parameters
        ----------
        mesh_res : float, optional
            The resolution of the mesh in Mpc/h, default is 1 Mpc/h
        Returns
        -------
        k_plin : ndarray
            Wavenumbers of the power spectrum.
        p_plin : ndarray
            Power spectrum values.
        """

        mesh = self.get_dens_mesh(mesh_res=mesh_res)
        # Compute the power spectrum
        plin = FFTPower(mesh, mode='1d').run()[0]
        k_plin = plin['k']
        p_plin = plin['power'].real
        return k_plin, p_plin


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

    def get_cosmo(self, param):
        """
        Return an Nbodykit.cosmology.Cosmology module satisfying the 
        cosmology parmeters in param
        """
        cosmo = Cosmology(Omega_cdm=param['omega0'],
                            Omega_b= param['omegab'],
                            h=param['hubble'],
                            n_s=param['ns'],
                            N_ur=param['N_ur'],
                            extra_pars={"A_s":param['scalar_amp'], 
                                        "n_s":param['ns'],
                                        "m_ndm":param['m_nu'],
                                        "alpha_s":param['alpha_s']})
        return cosmo
    
    def halo_exclusion(self, pos, mass, ex_radius, boxsize=None):
        """
        Perform halo exclusion by removing smaller halos within the ex_radius of more massive halos.

        Parameters
        ----------
        pos : ndarray of shape (N, 3)
            Positions of halos (in Mpc/h).
        mass : ndarray of shape (N,)
            Masses of halos (in Msun/h), used to sort halos.
        ex_radius : ndarray of shape (N,)
            The radius of exclusion for each halo (in Mpc/h). 
            Halos within this radius of a more massive halo will be excluded.
        boxsize : float, optional
            If given, apply periodic boundary conditions.

        Returns
        -------
        keep_mask : ndarray of bool
            Boolean mask array indicating which halos are kept.
        """

        N = len(mass)
        idx_sorted = np.argsort(-mass)  # Sort by decreasing mass
        pos = pos[idx_sorted]
        ex_radius = ex_radius[idx_sorted]

        keep_mask = np.ones(N, dtype=bool)
        tree = cKDTree(pos, boxsize=boxsize)

        for i in range(N):
            if not keep_mask[i]:
                continue  # Already excluded

            # Query neighbors within this halo's ex_radius
            neighbors = tree.query_ball_point(pos[i], ex_radius[i])
            for j in neighbors:
                if j <= i:
                    continue  # Only exclude lower-mass halos
                keep_mask[j] = False

        # Map keep_mask back to original order
        inverse_sort = np.argsort(idx_sorted)
        return keep_mask[inverse_sort]

    def load_halo_cat(self, pig_dir, cosmo, ex_rad_fac = 10):
        """
        Load FOF tables as Nbodykit Halocatalog. It will be used to populate them with HOD models.
        Parameters:
        ------------
        pig_dir: str, 
            The path to the PIG directory
        cosmo: Nbodykit.cosmology.Cosmology instance
            output of `get_cosmo()`
        ex_rad_fac: float, optional
            The factor by which to multiply the R200 radius of each halo to determine the exclusion radius.
            If ex_rad_fac > 0, smaller halos within this radius of larger halos will be excluded.
            If ex_rad_fac <= 0, no exclusion will be performed.
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
        # If exclusion is True, we will exclude smaller halos within the exclusion radius of larger halos
        if ex_rad_fac > 0:
            # Gather data to rank 0 for exclusion
            all_pos = self.nbkit_comm.gather(halos['Position'].compute(), root=0)
            all_mass = self.nbkit_comm.gather(halos['Mass'].compute(), root=0)
            all_r200 = self.nbkit_comm.gather(ex_rad_fac*halos['Radius'].compute(), root=0)

            if self.nbkit_rank == 0:
                pos_concat = np.concatenate(all_pos)
                mass_concat = np.concatenate(all_mass)
                r200_concat = np.concatenate(all_r200)
                keep_mask_all = self.halo_exclusion(pos_concat, mass_concat, r200_concat, boxsize=halos.attrs['BoxSize'])
            else:
                keep_mask_all = None

            # Broadcast the final keep mask
            keep_mask_all = self.nbkit_comm.bcast(keep_mask_all, root=0)

            # Apply the mask on this rank's halos
            ## Fist, find the cumulative halo count to this rank
            n_halos_local = len(halos)
            counts = self.nbkit_comm.allgather(n_halos_local)
            offset = sum(counts[:self.nbkit_rank])
            keep_mask = keep_mask_all[offset:offset + len(halos)]
            halos = halos[keep_mask]
            self.logger.info(f'Rank {self.nbkit_rank} removed {np.sum(~keep_mask)/len(keep_mask)*100:.2f}% of halos due to exclusion within radius {ex_rad_fac}*R200')
            if self.nbkit_rank == 0:
                self.logger.info(f'In total, {np.sum(~keep_mask_all)/len(keep_mask_all)*100:.2f}% of halos were removed due to exclusion within radius {ex_rad_fac}*R200')

        return halos

    def get_matter_density(self, pig_dir, cosmo, mesh_res=1, compute_mesh=False):
        """
        Get the matter density field from the Particle catalog
        Parameters
        ----------
        pig_dir: str,
            The path to the PIG directory
        cosmo: Nbodykit.cosmology.Cosmology instance
            output of `get_cosmo()`
        mesh_res: float, optional
            The resolution of the mesh in Mpc/h, default is 1 Mpc/h
        compute_mesh: bool, optional
            If True, compute the density field as a numpy array, otherwise return the Nbodykit Mesh object
        Returns:
        ------------
        if compute_mesh is False, returns:
            Nbodykit Mesh object containing the matter density field
        if compute_mesh is True, returns:
            the density field as a numpy array
        """
        # Load the particle catalog
        part_cat = BigFileCatalog(pig_dir, dataset='1')
        # Set the cosmology with the parameters from the simulation
        part_cat.attrs['BoxSize'] /= 1000  # Convert from kpc to Mpc
        part_cat['Position'] /= 1000  # Convert from kpc to Mpc

        # Apply RSD to the galaxies
        z = 1/part_cat.attrs['Time'] - 1
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        part_cat['RSDPosition'] = (part_cat['Position'] + part_cat['Velocity'] * rsd_factor) % part_cat.attrs['BoxSize']
        
        # Create a mesh from the particle catalog
        Nmesh = int(part_cat.attrs['BoxSize'][0] / mesh_res)
        mesh = part_cat.to_mesh(position='RSDPosition', Nmesh=1000, compensated=True, BoxSize=part_cat.attrs['BoxSize'])
        if compute_mesh:
            # Compute the density field as a numpy array
            density_field = mesh['density'].compute()
            # Convert the density field to a numpy array
            density_field = density_field.reshape((Nmesh, Nmesh, Nmesh))
            return density_field
        else:
            # Return the mesh object
            return mesh
        
    def _get_corr(self, pig_dir, cosmo, mass_th, ex_rad_fac=10, z=2.5):
        """Get the correlation function for two halo catalogs with 2 mass thresholds
        Parameters
        ----------
        pig_dir: str, 
            The path to the PIG directory
        cosmo: Nbodykit.cosmology.Cosmology instance
            output of `get_cosmo()`
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
        halos = self.load_halo_cat(pig_dir, cosmo=cosmo, ex_rad_fac=ex_rad_fac)

        # Apply RSD to the galaxies
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        halos['RSDPosition'] = (halos['Position'] + halos['Velocity'] * los * rsd_factor)%halos.attrs['BoxSize']

        data1 = halos[halos['Mass'] >= mass_th[0]]
        data2 = halos[halos['Mass'] >= mass_th[1]]
        
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
        if self.rank==0:
            self.logger.info(f'Gert corr for sim {start} to {end}')
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

    def _get_power(self, pig_dir, params, mass_th, z, mode='1d', mesh_res=1, ex_rad_fac=0):
        """Get the powerspectrum for HOD populated galaxies in a FOF halo catalog.
        Parameters
        ----------
        pig_dir: str, 
            The path to the PIG directory
        cosmo: Nbodykit.cosmology.Cosmology instance
            output of `get_cosmo()`
        mass_th: tuple of floats
            The mass threshold for the first and second halo samples
        Returns:
        ------------
        The powerspectrum for different seeds
        """
        # Set the cosmology with the parameters from the simulation
        cosmo = self.get_cosmo(params)
        # in z-space
        los = [0, 0, 1]
        halos = self.load_halo_cat(pig_dir, cosmo=cosmo, ex_rad_fac=ex_rad_fac)
        # Apply RSD to the galaxies
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        halos['RSDPosition'] = (halos['Position'] + halos['Velocity'] * los * rsd_factor)%halos.attrs['BoxSize']

        data1 = halos[halos['Mass'] >= mass_th[0]]
        data2 = halos[halos['Mass'] >= mass_th[1]]

        Nmesh = int(halos.attrs['BoxSize'][0] / mesh_res)
        fund_mode = 2*np.pi/halos.attrs['BoxSize'][0]
        power = FFTPower(first=data1, second=data2,  Nmesh=Nmesh, BoxSize=halos.attrs['BoxSize'], los=los, mode=mode, dk=0.3*fund_mode)
        return power.run()[0]
    
    def _get_cross_power(self, pig_dir, params, mass_th, z=2.5, mode='1d', mesh_res=1, ex_rad_fac=0):
        """
        Get the cross power for halos and matter
        Parameters
        ----------
        pig_dir: str,
            The path to the PIG directory
        params: dict,
            The simulation parameters, output of `get_pig_dirs()`
        mass_th: float
            The mass threshold for the halos
        z: float, not an array
            redshift of the snapshot
        mode: str
            mode of the power spectrum, either '1d', '2d', 'projected', 'angular'
        mesh_res: float, optional
            The resolution of the mesh in Mpc/h, default is 1 Mpc/h
        Returns:
        ------------
        The cross power spectrum for halos and matter
        """
        # Set the cosmology with the parameters from the simulation
        cosmo = self.get_cosmo(params)
        # in z-space
        los = [0, 0, 1]
        halos = self.load_halo_cat(pig_dir, cosmo=cosmo, ex_rad_fac=ex_rad_fac)
        # Apply RSD to the galaxies
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        halos['RSDPosition'] = (halos['Position'] + halos['Velocity'] * los * rsd_factor)%halos.attrs['BoxSize']
        # Get the matter density field as a mesh
        matter_mesh = self.get_matter_density(pig_dir, cosmo=cosmo, mesh_res=mesh_res, compute_mesh=False)
        # Get the halos with the mass threshold
        data1 = halos[halos['Mass'] >= mass_th]
        Nmesh = int(halos.attrs['BoxSize'][0] / mesh_res)
        # Compute the cross power spectrum
        pk_hm = FFTPower(first=data1, second=matter_mesh, Nmesh=Nmesh, 
                         BoxSize=halos.attrs['BoxSize'], los=los, mode=mode, 
                         dk=0.3*(2*np.pi/halos.attrs['BoxSize'][0]))
        # Run the power spectrum computation
        return pk_hm.run()[0]

    def _get_ph_lin(self, pig_dir, params, mass_th, z=2.5, mesh_res=1, mode='1d', ex_rad_fac=0):
        """
        Geet the cross power spectrum for halos and initial density field
        Parameters
        ----------
        pig_dir: str,
            The path to the PIG directory
        params: dict,
            The simulation parameters, output of `get_pig_dirs()`
        mass_th: float
            The mass threshold for the halos
        z: float, not an array
            redshift of the snapshot
        mesh_res: float, optional
            The resolution of the mesh in Mpc/h, default is 1 Mpc/h
        mode: str
            mode of the power spectrum, either '1d', '2d', 'projected', 'angular'
        ex_rad_fac: float, optional
            The factor by which to multiply the R200 radius of each halo to determine the exclusion radius.
            If ex_rad_fac > 0, smaller halos within this radius of larger halos will be excluded.
            If ex_rad_fac <= 0, no exclusion will be performed.
        Returns:
        ------------
        pwoe, Nbodykit's BinnedStatistics
        The cross power spectrum for halos and initial density field,
        use the keys 'k' and 'power' to get the wavenumbers and power spectrum values
        """
        # Set the cosmology with the parameters from the simulation
        cosmo = self.get_cosmo(params)
        # in z-space
        los = [0, 0, 1]
        halos = self.load_halo_cat(pig_dir, cosmo=cosmo, ex_rad_fac=ex_rad_fac)
        # Apply RSD to the galaxies
        rsd_factor = (1+z) / (100 * cosmo.efunc(z))
        halos['RSDPosition'] = (halos['Position'] + halos['Velocity'] * los * rsd_factor)%halos.attrs['BoxSize']
        # Get the halos with the mass threshold
        data1 = halos[halos['Mass'] >= mass_th]

        # Gat a  mesh of the initial density field
        init = InitialDensity(pig_dir)
        init_mesh = init.get_dens_mesh(mesh_res=mesh_res)

        # Compute the cross power spectrum
        Nmesh = int(halos.attrs['BoxSize'][0] / mesh_res)
        pk_hm = FFTPower(first=data1, second=init_mesh, Nmesh=Nmesh,
                            BoxSize=halos.attrs['BoxSize'], los=los, mode=mode)
                            #dk=0.3*(2*np.pi/halos.attrs['BoxSize'][0]))
        # Run the power spectrum computation
        return pk_hm.run()[0]
    
    def _get_initial_power(self, pig_dir, params, mode='1d', z=2.5, mesh_res=1):
        """
        Get the initial power spectrum of the matter density field
        Parameters
        ----------
        pig_dir: str,
            The path to the PIG directory
        params: dict,
            The simulation parameters, output of `get_pig_dirs()`
        z: float, not an array
            redshift of the snapshot
        mesh_res: float, optional
            The resolution of the mesh in Mpc/h, default is 1 Mpc/h
        Returns:
        ------------
        """
        if self.nbkit_rank == 0:
            self.logger.info(f'Get initial power for {pig_dir} at z={z} with mesh_res={mesh_res} cMpc/h')
        # along z-space
        los = [0, 0, 1]
        # Gat a  mesh of the initial density field
        init = InitialDensity(pig_dir)
        init_mesh = init.get_dens_mesh(mesh_res=mesh_res)

        # Compute the cross power spectrum
        Nmesh = int(init_mesh.attrs['BoxSize'][0] / mesh_res)
        fftpow = FFTPower(first=init_mesh, Nmesh=Nmesh,
                         BoxSize=init_mesh.attrs['BoxSize'], 
                         los=los, mode=mode) 
        pk = fftpow.run()[0]
        return pk['k'], pk['power'].real
        

    def _power_on_grid(self, pig_dir, params, z=2.5):
        """
        Get the 3D correlation fucntion on a grid with increasing mass thresholds
        """
        mbins = np.arange(12.3, 10.95,-0.1 )
        idx = np.triu_indices(len(mbins), k=0)
        pairs = np.column_stack((mbins[idx[0]], mbins[idx[1]]))
        pow_hh = []
        for i, m_pair in enumerate(pairs):
            #if i in [0, 10, 25, 50, 100, 150, 200]:
            if self.nbkit_rank == 0:
                self.logger.info(f'progress {100*i/len(pairs)} %')
            pow = self._get_power(pig_dir, params, mass_th=10**m_pair, z=z)
            k = pow['k']
            pow_hh.append(pow['power'].real)
        return k, np.array(pow_hh), mbins, pairs
    
    def _cross_power_on_grid(self, pig_dir, params, z=2.5):
        """
        Get the cross power spectrum for halos and matter on a grid with increasing mass thresholds
        """
        mbins = np.arange(12.3, 10.9,-0.5 )
        phm = []
        for i, mth in enumerate(mbins):
            if self.nbkit_rank == 0:
                self.logger.info(f'progress {100*i/len(mbins)} %')
            pk_hm = self._get_cross_power(pig_dir, params, mass_th=10**mth, z=z)
            k = pk_hm['k']
            phm.append(pk_hm['power'].real)
        return k, np.array(phm), mbins
    
    def _ph_lin_on_grid(self, pig_dir, params, z=2.5):
        """
        Get the linear power spectrum of halos on a grid with increasing mass thresholds
        """
        mbins = np.arange(13, 10.95,-0.1 )
        if self.nbkit_rank == 0:
            self.logger.info(f'mass bins are {mbins}')
        ph_lin = []
        for i, mth in enumerate(mbins):
            if self.nbkit_rank == 0:
                self.logger.info(f'progress {100*i/len(mbins)} %')
            pk_hm = self._get_ph_lin(pig_dir, params, mass_th=10**mth, z=z)
            k = pk_hm['k']
            ph_lin.append(pk_hm['power'].real)
        return k, np.array(ph_lin), mbins

    def get_power_on_grid(self, base_dir, save_dir, chunk, power_type='hh', narrow=False, z=2.5, num_chunks=20):
        """
        Get Power spectrum of halo x halo ro halo x matter on larger scales, r > 40 Mpc/h on a grid of mass thresholds
        Parameters
        ----------
        base_dir : str
            base directory of the simulations, if `narrow`, the
            narrow sims should be in `/narrow` subdirectory
        save_dir : str
            directory to save the power spectrum
        chunk : int
            chunk number to process
        power_type : str
            type of power spectrum to compute, either 'hh' for halo x halo or 'hm' for halo x matter
        narrow : bool
            if True, the narrow sims should be in `/narrow` subdirectory
        z : float, not an array
            redshift of the snapshot
        num_chunks : int
            number of chunks to split the work
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
        if self.rank==0:
            self.logger.info(f'Get power for sim {start} to {end}')
        for i in range(start, end):
            save_file = os.path.join(save_dir, 'power_'+pigs["sim_tags"][i]+'.hdf5')
            if os.path.exists(save_file):
                if self.nbkit_rank ==0:
                    self.logger.info(f'skipping {pigs["sim_tags"][i]} since it is already computed')
            else:
                if self.nbkit_rank==0:
                    self.logger.info(f'working on {pigs["sim_tags"][i]}')
                try:
                    if power_type == 'hh':
                        k, pk, mbins, pairs = self._power_on_grid(pigs['pig_dirs'][i], pigs['params'][i], z=z)
                    elif power_type == 'hm':
                        k, pk, mbins = self._cross_power_on_grid(pigs['pig_dirs'][i], pigs['params'][i], z=z)
                    elif power_type == 'hlin':
                        k_hlin, pk_hlin, mbins = self._ph_lin_on_grid(pigs['pig_dirs'][i], pigs['params'][i], z=z)
                        # For the linitial density field:
                        k_init, pk_init = self._get_initial_power(pigs['pig_dirs'][i], pigs['params'][i], z=z)
                    else:
                        raise ValueError(f'Unknown power_type: {power_type}. Use "hh", "hm" or "hlin".')
                    if self.nbkit_rank ==0:
                        if power_type == 'hh':
                            self._save_power_on_grid(k, pk, mbins, pairs, pigs['sim_tags'][i], save_file)
                        elif power_type == 'hm':
                            # For cross power or linear power
                            self._save_cross_power_on_grid(k, pk, mbins, pigs['sim_tags'][i], save_file)    
                        elif power_type == 'hlin':
                            self._save_pk_hlin_on_grid(k_hlin, k_init, pk_hlin, pk_init, mbins, pigs['sim_tags'][i], save_file)
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

    def _save_power_on_grid(self, k, power_hh, mbins, pairs, sim_tag, save_file):
        self.logger.info(f'Writing on {save_file}')
        with h5py.File(save_file, 'w') as fw:
            fw['power'] = power_hh
            fw['mbins'] = mbins
            fw['pairs'] = pairs
            fw['sim_tag'] = sim_tag
            fw['k'] = k
    
    def _save_cross_power_on_grid(self, k, power_hm, mbins, sim_tag, save_file):
        self.logger.info(f'Writing on {save_file}')
        with h5py.File(save_file, 'w') as fw:
            fw['power'] = power_hm
            fw['mbins'] = mbins
            fw['sim_tag'] = sim_tag
            fw['k'] = k
    
    def _save_pk_hlin_on_grid(self, k_hlin, k_init, power_hlin, power_init, mbins, sim_tag, save_file):
        self.logger.info(f'Writing on {save_file}')
        with h5py.File(save_file, 'w') as fw:
            fw['power_hlin'] = power_hlin
            fw['power_init'] = power_init
            fw['mbins'] = mbins
            fw['sim_tag'] = sim_tag
            fw['k_hlin'] = k_hlin
            fw['k_init'] = k_init

