# Take care of path import on TACC, helps to import mpi4py
import sys
#sys.path.insert(0, '/home1/06536/qezlou/miniconda3/envs/goku/lib/python3.8/site-packages')
#path_to_exclude = '/opt/apps/intel19/impi19_0/python3/3.9.7/lib/python3.9/site-packages/'
#sys.path.remove()

#if path_to_exclude in sys.path:
#    sys.path.remove(path_to_exclude)
########


import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import h5py
import argparse
import numpy as np

from nbodykit.lab import cosmology, LogNormalCatalog, SimulationBox2PCF
from nbodykit import CurrentMPIComm, setup_logging

comm = CurrentMPIComm.get()
rank = comm.Get_rank()
size = comm.Get_size()

setup_logging('debug')

setup_sources = {'oii':{'z':0.25, 
                        'bias':2, 
                        'nbar': 1.190e-3, # cMpc/h ^-3
                        'BoxSize': 805, # cMpc/h
                 },'lae':{'z':2.2, 
                        'bias':2, 
                        'nbar': 6.97e-5, # cMpc/h ^-3
                        'BoxSize': 2_470, # cMpc/h
                 }}

# Set the cosmology
cosmo = cosmology.Planck15

def generate_lognormal_mock(stype='lae', res=4, seed=21):
    """Generate mock lognormal catalog"""
    specs = setup_sources[stype]
    Plin = cosmology.LinearPower(cosmo, specs['z'], transfer='EisensteinHu')
    Nmesh = int(specs['BoxSize'] /res)
    # Catalog holds the list of particles that are samples from the lognorrmal field
    cat = LogNormalCatalog(Plin=Plin, nbar=specs['nbar'], 
                           BoxSize=specs['BoxSize'], Nmesh=Nmesh, 
                           bias=specs['bias'], seed=seed)

    return cat

def get_corr(cat, r_edges):
    
    corr = SimulationBox2PCF(data1=cat, mode='1d', edges=r_edges, nthreads=1)
    corr.run()
    return corr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stype', type=str, required=True, help='')
    args = parser.parse_args()

    np.random.seed(22)
    seeds = np.random.randint(6001, 8001, size=50)

    r_edges = 10**(np.linspace(0, np.log10(200), 100))
    for s in seeds:
        if rank ==0:
            print(f'seed = {s}', flush=True)
        fname = f'corr_{args.stype}_seed{s}.hdf5'
        cat = generate_lognormal_mock(stype=args.stype, seed=s)
        if rank == 0:
            print(f'cat.csize = {cat.csize}')
        corr= get_corr(cat, r_edges=r_edges)
        comm.Barrier()
        if rank==0:
            with h5py.File(fname, 'w') as f:
                f['r'] = corr.corr['r'][:]
                f['corr'] = corr.corr['corr'][:]
                f.create_group('Header')
                f['Header'].attrs['stype'] = args.stype
                for k, v in setup_sources[args.stype].items():
                    f['Header'].attrs[k] = v

        comm.Barrier()