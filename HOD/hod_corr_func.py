# Take care of path import on TACC, helps to import mpi4py
import sys
sys.path.insert(0, '/home1/06536/qezlou/miniconda3/envs/goku/lib/python3.8/site-packages')
path_to_exclude = '/opt/apps/intel19/impi19_0/python3/3.9.7/lib/python3.9/site-packages/'
#sys.path.remove()

if path_to_exclude in sys.path:
    sys.path.remove(path_to_exclude)
########


import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


import numpy as np
import h5py
import nbodykit

from nbodykit import CurrentMPIComm, style, setup_logging
from nbodykit.lab import BigFileCatalog, HaloCatalog, cosmology, FFTPower, SimulationBox2PCF

comm = CurrentMPIComm.get()
rank = comm.Get_rank()
size = comm.Get_size()


cosmo = cosmology.Planck15
#from nbodykit.hod import Zheng07Model
#from nbodykit.hod import HmqModel
from nbodykit.hod import Hadzhiyska23Model

setup_logging()

def load_hlo_cat(extra_attrs_file=None):
    pig_file = '/work/06536/qezlou/ls6/Goku/output/PIG_016'
    cat = BigFileCatalog(pig_file, dataset='FOFGroups')
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
    if extra_attrs_file is not None:
        with h5py.File(extra_attrs_file, 'r') as f:
            extra_attrs = f.attrs
        for k in list(extra_attrs.keys()):
            halos[k] =  f[k]
    return halos

def populate_halos(model, extra_attrs_file=None,  seed=42, **kwargs):
    halos = load_hlo_cat(extra_attrs_file)
    #hod = halos.populate(Zheng07Model, alpha=0.5, sigma_logM=0.40, seed=seed)
    hod = halos.populate(model)

    cen_idx = hod['gal_type'] == 0
    sat_idx = hod['gal_type'] == 1

    cens = hod[cen_idx]
    sats = hod[sat_idx]

    print(f'rank = {rank}, hod # = {hod.size}, cens = {cens.size}, sats = {sats.size}', flush=True)
    

    return hod, cens, sats

def get_corr(model, extra_attrs_file=None, seed=42, **kwrags):
    """Plot the power spectrum of the galaxies, centrals, and satellites."""
    hod, _, _ = populate_halos(model, extra_attrs_file, seed)
    r_edges = np.arange(0.1, 1, 0.2)
    r_edges = np.append(r_edges, np.arange(1,30,1))
    r_edges = np.append(r_edges, np.arange(30, 200, 5))
    corr_gal_real = SimulationBox2PCF(data1=hod, mode='1d', edges=r_edges, position='Position')
    corr_gal_real.run()
    

    # z-space corr function
    LOS = [0, 0, 1]
    hod['RSDPosition'] = hod['Position'] + hod['VelocityOffset'] * LOS
    corr_gal_zspace = SimulationBox2PCF(data1=hod, mode='1d', edges=r_edges,  position='RSDPosition')
    corr_gal_zspace.run()

    cen_idx = hod['gal_type'] == 0
    sat_idx = hod['gal_type'] == 1
    cens = hod[cen_idx]
    sats = hod[sat_idx]

    corr_cen_zspace = SimulationBox2PCF(data1=cens, mode='1d', edges=r_edges, position='RSDPosition')
    corr_cen_zspace.run()

    corr_sat_zspace = SimulationBox2PCF(data1=sats, mode='1d', edges=r_edges, position='RSDPosition')
    corr_sat_zspace.run()


    if rank == 0:
        with h5py.File(f'halo_corr_hmq_{seed}.hdf5', 'w') as f:
            f['r'] = corr_gal_real.corr['r']
            f['corr/real'] = corr_gal_real.corr['corr']
            f['corr/zspace'] = corr_gal_zspace.corr['corr']
            f['corr/cen_zspace'] = corr_cen_zspace.corr['corr']
            f['corr/sat_zspace'] = corr_sat_zspace.corr['corr']
    comm.Barrier()

all_seeds = np.random.randint(1, 1_000_000, 100)

for seed in all_seeds:
    get_corr(seed=seed
             model=Hadzhiyska23Model, 
             extra_attrs_file=None
             )

