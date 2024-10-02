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

comm = CurrentMPIComm.get()
rank = comm.Get_rank()
size = comm.Get_size()

def configure_logging(logging_level):
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


def get_pig_dirs(base_dir, z=2.5):
    """Get the directories of the PIGs
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
    logger.info(f'base_dir = {base_dir} | number of sims = {len(sim_dirs)}, z = {z}')
    for sd in sim_dirs:
        logger.debug(f'Openning sim = {sd}')
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

logger = configure_logging(logging.DEBUG)

def load_hlo_cat(pig_dir):
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

def populate_halos(pig_dir, hod_model, seed=42, model_params={}):
    """
    
    """
    halos = load_hlo_cat(pig_dir)
    hod = halos.populate(hod_model, seed=seed, **model_params)
    print("total number of HOD galaxies = ", hod.csize)
    print(hod.columns)

    logger.debug(f"number of centrals = { hod.compute((hod['gal_type']==0).sum())}")
    logger.debug(f"number of satellites = {hod.compute((hod['gal_type']==1).sum())}")


    #cen_idx = hod['gal_type'] == 0
    #sat_idx = hod['gal_type'] == 1

    #cens = hod[cen_idx]
    #sats = hod[sat_idx]

    #return hod, cens, sats
    return hod


def get_corr(pig_dir, r_edges, seed=42, corr_mode='1d', hod_model=Zheng07Model):
    """Plot the corrfunction of the galaxies, centrals, and satellites.
    Parameters
    ----------
    seed : int
        seed for the HOD model
    corr_mode : str
        mode of the correlation function, either '1d', '2d', 'projected', 'angular'
    """
    #hod, cens, sats = populate_halos(seed=seed, hod_model=hod_model)
    # in real space
    #corr_gal_real = SimulationBox2PCF(data1=hod, mode=corr_mode, edges=r_edges, position='Position')
    #corr_gal_real.run()
    hod = populate_halos(pig_dir, hod_model=hod_model, seed=seed)
    # in z-space
    LOS = [0, 0, 1]
    hod['RSDPosition'] = hod['Position'] + hod['VelocityOffset'] * LOS
    corr_gal_zspace = SimulationBox2PCF(data1=hod, mode=corr_mode, edges=r_edges,  position='RSDPosition')
    corr_gal_zspace.run()

    #cen_idx = hod['gal_type'] == 0
    #sat_idx = hod['gal_type'] == 1
    #cens = hod[cen_idx]
    #sats = hod[sat_idx]
    #corr_cen_zspace = SimulationBox2PCF(data1=cens, mode=corr_mode, edges=r_edges, position='RSDPosition')
    #corr_cen_zspace.run()
    #corr_sat_zspace = SimulationBox2PCF(data1=sats, mode=corr_mode, edges=r_edges, position='RSDPosition')
    #corr_sat_zspace.run()

    return corr_gal_zspace

def iterate_over_pigs(base_dir,  z=2.5, savedir = '/work2/06536/qezlou/Goku/corr_funcs'):

    hod_model = Zheng07Model
    pigs = get_pig_dirs(base_dir, z=z)
    r_edges = np.logspace(0, np.log10(200), 100)
    corr_func = np.zeros((len(pigs['sim_tags']), len(r_edges)-1))
    for i, pig_dir in enumerate(pigs['pig_dirs']):
        corr_gal_zspace = get_corr(pig_dir, r_edges, seed=42, hod_model=hod_model)
        corr_func[i] = corr_gal_zspace.corr['corr']
    with h5py.File(os.path.join(savedir, f'corr_gal_zspace_z{np.round(z,2)}.hdf5'), 'w') as f:
        f['r'] = ( r_edges[:-1] + r_edges[1:] ) / 2
        f['corr'] = corr_func


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--base_dir', type=str, help='Base directory of the simulations')
    parser.add_argument('--z', type=float, help='Redshift of the snapshot')
    parser.add_argument('--savedir', type=str, help='Directory to save the correlation functions')
    parser.add_argument('--logging_level', required=False, type=int, default=logging.INFO, help='Logging level')
    args = parser.parse_args()
    iterate_over_pigs(args.base_dir, z=args.z, savedir=args.savedir)


