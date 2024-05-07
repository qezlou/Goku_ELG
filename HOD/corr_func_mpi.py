import numpy as np
import h5py
import nbodykit

from nbodykit import CurrentMPIComm, style, setup_logging
from nbodykit.lab import BigFileCatalog, HaloCatalog, cosmology, FFTPower, SimulationBox2PCF

comm = CurrentMPIComm.get()
rank = comm.Get_rank()
size = comm.Get_size()


cosmo = cosmology.Planck15
from nbodykit.hod import Zheng07Model


def load_hlo_cat():
    pig_file = '/home/qezlou/HD1/simulations/Goku/PIG_016'
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
    return halos


halos = load_hlo_cat()
halos['Position'] /= 1000
r_edges = np.arange(60,200,5)
corr = SimulationBox2PCF(data1=halos, mode='1d', edges=r_edges, position='Position', nthreads=12)
corr.run()

if  rank == 0:
    with h5py.File('halo_corr.hdf5', 'w') as f:
        f['r'] = corr.corr['r']
        f['corr'] = corr.corr['corr']
