"""
A sript to be run opn bash to get hte galaxy correlation function of Goku sims
It uses the HOD models which are either standard or I have implemented in `nbodykit`.
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
import mpi_helper
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
                                            '/scratch/06536/qezlou/Goku/FOF/L2/']):
        
        
        # Load JSON file as a dictionary
        def load_json_as_dict(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data

        all_ICs = []
        for bsd in base_dirs:
            # Example usage
            fnames = glob(op.join(bsd,f'*_10p_Box*/SimulationICs.json'))
            self.logger.info(f'number of files in {bsd} is {len(fnames)}')
            for fn in fnames:
                data_dict = load_json_as_dict(fn)
                num = data_dict['outdir'].split('_')[-1].split('.')[0]
                data_dict['label'] = f'10p_Box{data_dict["box"]}_Part{data_dict["npart"]}_{num}'
                all_ICs.append(data_dict)
        save_file = 'all_ICs.json'
        self.logger.info(f'writing on {save_file}')
        with open(save_file, 'w') as json_file:
            json.dump(all_ICs, json_file, indent=4)

        with open(save_file, 'r') as json_file:
            data = json.load(json_file)
            self.logger.info(f'totla files = {len(data)}')

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
            dictionary containing :
            'pig_dirs': The PIG directories 
            'params': The simulation parameters
            'sim_tags': The directory name holding the sims
        """
        # Find all the directories in `base_dir` that look like a snapshot
        sim_tags = [t for t in os.listdir(base_dir) if ( os.path.isdir(os.path.join(base_dir, t)) and ('Box' in os.path.join(base_dir, t)) )]
        sim_dirs = [os.path.join(base_dir, t) for t in sim_tags]
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


    def get_corr(self, pig_dir, r_edges, seeds=[42], mode='1d', pimax=None, Nmu=50, fft_model=False, mesh_res=0.15, hod_model=Zheng07Model, model_params={}):
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
        mode : str
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
        if mode=='1d':
            all_corrs = np.zeros((seeds.size, len(r_edges)-1))
        elif mode=='projected':
            all_corrs = np.zeros((seeds.size, len(r_edges)-1))
        elif mode=='2d':
            all_corrs = np.zeros((seeds.size, len(r_edges)-1, Nmu))
        for i, sd in enumerate(seeds):
            if (i in [0, 10, 25, 50, 75]) and self.nbkit_rank==0:
                self.logger.info(f'Color {self.color}, progress in seeds pool {i} % for {pig_dir}')
            if i==0:
                hod = halos.populate(hod_model, seed=sd, **model_params)
            else:
                hod.repopulate(seed=sd)
            self.nbkit_comm.Barrier()
            # Apply RSD to the galaxies
            hod['RSDPosition'] = (hod['Position'] + hod['VelocityOffset'] * los)%hod.attrs['BoxSize']
            if fft_model:
                if mode=='projected':
                    raise NotImplementedError(f"No support for corr with fftmodel in '{mode}' mode ")
                Nmesh = int(hod.attrs['BoxSize'][0] / mesh_res)
                corr_gal_zspace = FFTCorr(hod, mode=mode, Nmesh=Nmesh,los=los, edges=r_edges)
            else:
                corr_gal_zspace = SimulationBox2PCF(data1=hod, mode=mode, edges=r_edges, pimax=pimax, position='RSDPosition')
            corr_gal_zspace.run()
            self.nbkit_comm.Barrier()
            if mode == 'projected':
                all_corrs[i] = corr_gal_zspace.wp['corr'][:]
            else:
                all_corrs[i] = corr_gal_zspace.corr['corr'][:]
        return all_corrs
    

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
    
    def get_corr_fof(self, pig_dir, r_edges=None, mesh_res=0.05, z=2.5,  stat='corr', mode='1d', pimax=40, false_positive_ratio = 0):
        """Get the correlation function for a FOF halo catalog, with no HOD.
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
        # Add uniformly distributed random false positives if asked for
        if false_positive_ratio >= 0:
            fp_size = int(halos.csize * false_positive_ratio / (1- false_positive_ratio))
            int(halos.csize*false_positive_ratio)
            if self.rank ==0:
                self.logger.info(f'False positive: ratio = {false_positive_ratio}, {fp_size} added to {halos.csize} sources')
            pos = np.random.uniform(low=0.0, high=halos.attrs['BoxSize'], size=(fp_size, 3))
            pos = np.append(halos['RSDPosition'].compute(), pos, axis=0)
            halos = ArrayCatalog({'RSDPosition':pos}, BoxSize=[250.0])
        if stat == 'corr':
            corr = SimulationBox2PCF(data1=halos, mode=mode, edges=r_edges, pimax=pimax,  position='RSDPosition')
            corr.run()
            self.nbkit_comm.Barrier()
            result = corr.corr['corr'][:]
            mbins =  np.array([r_edges[i]+r_edges[i+1] for i in range(r_edges.size-1)])
        elif stat == 'power':
            if self.rank == 0:
                self.logger.info(f'Getting power for FOF, mesh_res = {mesh_res} cMpc/h')
            Nmesh = int(halos.attrs['BoxSize'][0] / mesh_res)
            mesh = halos.to_mesh(position='RSDPosition', Nmesh=1000, compensated=True)
            pk_gal_zspace = FFTPower(mesh, mode=mode).power
            self.nbkit_comm.Barrier()
            result = pk_gal_zspace['power'].real
            mbins = pk_gal_zspace['k'].real
            
        return result, mbins
    
    def fix_hod_all_pigs(self, base_dir, hod_model, avoid_sims=[], stat='corr', mode='1d', pimax=None, seeds=[42], z=2.5, fft_model=False, savedir = '/work2/06536/qezlou/Goku/corr_funcs_smaller_bins/'):
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
        
        pigs = self.get_pig_dirs(base_dir, z=z)
        keep = self._remove_computed_snaps(pigs, savedir, avoid_sims)

        r_edges = np.logspace(-1.5, np.log10(2), 8)
        r_edges = np.append(r_edges, np.logspace(np.log10(2), np.log10(30), 10)[1:])
        # Decide the binning based on the boxsize
        if 'Box250' in pigs['sim_tags'][0]:
            r_edges = np.append(r_edges, np.logspace(np.log10(30), np.log10(80), 50)[1:])
        else:
            r_edges = np.append(r_edges, np.logspace(np.log10(30), np.log10(200), 100)[1:])
        
        if self.rank ==0:
            self.logger.info(f'Total existing sims {len(pigs["sim_tags"])}, but only {keep.size} of them are not yet computed or not avoided, stat = {stat}, mode = {mode}')
            self.logger.info(f'r_edges = {r_edges}')
            self.logger.info(f'len(seeds) = {len(seeds)}')
        # Distribute the jobs accross the available tasks, i.e. self.nbkit_comm_counts
        ind_pig_rank = mpi_helper.distribute_array_split_comm(self.nbkit_comm_counts, self.color, keep)
        self.logger.info(f'rank = {self.rank} | color = {self.color} | load = {ind_pig_rank.size}')
        for i in ind_pig_rank:
            #self.logger.debug(f'Progress {int}')
            pig_dir = pigs['pig_dirs'][i]
            
            # I commented out the `try`,`except` to 
            # see the full error messages from `nbodykit`
            try:
                if hod_model is not None:
                    if stat == 'corr':
                        result = self.get_corr(pig_dir, r_edges, seeds=seeds, hod_model=hod_model, mode=mode, pimax=pimax, fft_model=fft_model)
                        save_file=f'Zheng07_seeds_{pigs["sim_tags"][i]}.hdf5'
                    elif stat == 'power':
                        result, k = corr.get_power(pig_dir, mode=mode, seeds=seeds, hod_model=hod_model)
                        save_file=f'Zheng07_power_seeds_{pigs["sim_tags"][i]}.hdf5'
                else:
                    if stat == 'corr':
                        result = self.get_corr_fof(pig_dir, r_edges)
                        save_file=f'fof_{pigs["sim_tags"][i]}.hdf5'
                    elif stat == 'power':
                        raise NotImplementedError('Power for FOF halos (no hod applied) is not implemented')

            except Exception as e:
                if self.rank ==0:
                    self.logger.info(f'color {self.color} | Skipping, {pigs["sim_tags"][i]} because {e}')
                continue
            self.nbkit_comm.Barrier()
            
            save_file = os.path.join(savedir, save_file)
            if self.nbkit_rank ==0:
                self.logger.info(f'Color {self.color} | save_file = {save_file}')
                with h5py.File(save_file, 'w') as f:
                    if stat == 'corr':
                        f['r'] = ( r_edges[:-1] + r_edges[1:] ) / 2
                    elif stat == 'power':
                        f['k'] = k
                    f[stat] = result
                    f['seeds'] = seeds
        self.comm.Barrier()
        
    def _remove_computed_snaps(self, pigs, save_dir, avoid_sims):
        """
        Remove the simulations from the list
        """
        if len(avoid_sims) > 0:
            avoid_sims_string = [str(a).rjust(4,'0') for a in avoid_sims]
        else:
            avoid_sims_string = []
        if self.rank == 0:
            remove = []
            for i in range(len(pigs['sim_tags'])):
                save_file=f'Zheng07_seeds_{pigs["sim_tags"][i]}.hdf5'
                savefile = os.path.join(save_dir, save_file)
                if os.path.exists(savefile) or pigs["sim_tags"][i].split("_")[-1] in avoid_sims_string:
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
    parser.add_argument('--savedir', type=str, required=True, help='Directory to save the correlation functions')
    parser.add_argument('--logging_level', required=False, type=str, default=logging.INFO, help='Logging level')
    parser.add_argument('--ranks_for_nbkit', required=False, type=str, default=2, help='number of rank for each nbodykit communicator')
    parser.add_argument('--stat', required=False, type=str, default='corr', help='"power" or "corr"')
    parser.add_argument('--mode', required=False, type=str, default='1d', help='Corr mode')
    parser.add_argument('--pimax', required=False, type=int, default=0, help='Only if `mode =="projected", number of bins along line-of-sight')
    parser.add_argument('--fft_model', required=False, type=int, default=0, help='Whether to use FFTCorr or the paircounting')
    parser.add_argument('--hod_model', required=False, type=str, default='Zheng07Model', help='"power" or "corr"')
    parser.add_argument('--seed', required=False, type=int, default=127, help='fix to have same list of seeds defined below')
    parser.add_argument('--realizations', required=False, type=int, default=100, help='fix to have same list of seeds defined below')
    parser.add_argument('--avoid_sims', type=int, nargs='*', default=[], help='Range of r to consider')

    args = parser.parse_args()
    if args.logging_level == 'INFO':
        args.logging_level = 20
    elif args.logging_level == 'DEBUG':
        args.logging_level = 10
    corr = Corr(args.logging_level, ranks_for_nbkit=args.ranks_for_nbkit)

    if args.hod_model == 'Zheng07Model':
        hod_model = Zheng07Model
    elif  args.hod_model == 'None':
        hod_model = None
    # Get the Corrs for fixed HOD parameters of Zheng+07, but difference seeds
    # for all avaialble cosmologies
    np.random.seed(args.seed)
    seeds = np.unique(np.random.randint(0, 1000_000, size=args.realizations))

    if args.pimax == 0:
        args.pimax = None
    print(args.avoid_sims)
    
    with CurrentMPIComm.enter(corr.nbkit_comm):
        corr.fix_hod_all_pigs(args.base_dir, hod_model=hod_model, seeds=seeds, 
                              z=args.z, stat=args.stat, savedir=args.savedir,
                              mode=args.mode, pimax= args.pimax,
                              fft_model=args.fft_model,
                              avoid_sims=args.avoid_sims)
    