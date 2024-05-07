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

import h5py
import argparse
import numpy as np
import nbodykit

from nbodykit.lab import *
from nbodykit import CurrentMPIComm
from nbodykit import style, setup_logging

comm = CurrentMPIComm.get()
rank = comm.Get_rank()
size = comm.Get_size()


# Set the redshift
redshift = 1.9
# Set the cosmology
cosmo = cosmology.Planck15

def generate_lognormal_mock(nbar=3e-3, BoxSize=1000, Nmesh=256, bias=2, seed=4):
    """Generate mock lognormal catalog"""
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 2.0
    # Catalog holds the list of particles that are samples from the lognorrmal field
    cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=Nmesh, bias=bias, seed=seed)

    return cat

def get_corr(cat):
    r_edges = 10**(np.linspace(0, np.log10(200), 100))
    corr = SimulationBox2PCF(data1=cat, mode='1d', edges=r_edges, nthreads=1)
    corr.run()
    return corr


if __name__ == '__main__':
    boxsize = 3000
    Nmesh = int(boxsize/10)
    
    seeds = np.random.randint(0, 2001, size=50)
    for s in seeds:
        print(f'seed = {s}', flush=True)
        fname = f'corr_box_{boxsize}_seed{s}.hdf5'
        cat = generate_lognormal_mock(nbar = 5e-4, BoxSize=boxsize, Nmesh=Nmesh, seed=s)
        
        #print(f"rank={rank}   |  cat = {cat.csize}", flush=True)
        corr= get_corr(cat)
        comm.Barrier()
        if rank==0:
            with h5py.File(fname, 'w') as f:
                f['r'] = corr.corr['r'][:]
                f['corr'] = corr.corr['corr'][:]
        comm.Barrier()