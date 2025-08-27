"""
Test using the SubhaloRankGr for filtering centrals and compute the correlation function
"""

import numpy as np
from nbodykit.lab import SimulationBox2PCF
from nbodykit.source.catalog import BigFileCatalog
import h5py
import os.path as op
import argparse
from nbodykit import setup_logging, CurrentMPIComm
setup_logging()

# Set up MPI communicator
comm = CurrentMPIComm.get()
rank = comm.Get_rank()

r_edges = np.logspace(np.log10(0.01), np.log10(0.1), 4)
r_edges = np.append(r_edges, np.logspace(np.log10(0.1), np.log10(2), 15)[1:])
r_edges = np.append(r_edges, np.logspace(np.log10(2), np.log10(60), 15)[1:])
r_edges = np.append(r_edges, np.linspace(60, 80, 20)[1:])

m_low = 9
m_up = 9.2

def get_corr(cat, savefile):
    # Use comm for distributed computation
    corr = SimulationBox2PCF(data1=cat, mode='1d', edges=r_edges, position='Position', BoxSize=250, show_progress=True)
    corr.run()
    result = corr.corr['corr'][:]
    mbins = np.array([(r_edges[i]+r_edges[i+1])/2 for i in range(r_edges.size-1)])
    if rank == 0:
        with h5py.File(savefile, 'w') as f:
            f.create_dataset('mbins', data=mbins)
            f.create_dataset('result', data=result)

def get_power(cat, savefil):
    """
    """
    pass

def subhalos():
    path = '/work2/06536/qezlou/astrid_sfrh/FOF/SubGroups'
    # No dataset argument: load from root of SubGroups
    cat = BigFileCatalog(path, comm=comm)
    if rank == 0:
        print("BigFileCatalog columns:", cat.columns)
    mass = np.log10(cat['SubhaloMass']) + 10
    mask_mass = (mass >= m_low) & (mass <= m_up)
    mask_cen = cat['SubhaloRankInGr'] == 0

    cat_mass = cat[mask_mass]
    get_corr(cat_mass, f'corr_all_m_{m_low}_{m_up}.h5')
    cat_cen = cat[mask_mass & mask_cen]
    get_corr(cat_cen, f'corr_cen_m_{m_low}_{m_up}.h5')

def fof():
    if rank == 0:
        print(f'getting FOF')
    path = '/work2/06536/qezlou/astrid_sfrh/FOF/FOFGroups'
    save_dir = '/scratch/06536/qezlou/Goku/processed_data/halo_exclusion'
    # No dataset argument: load from root of FOFGroups
    cat = BigFileCatalog(path, comm=comm)
    cat['Position'] = cat['MassCenterPosition'][:]
    if rank == 0:
        print("BigFileCatalog columns:", cat.columns)
    mass = np.log10(cat['Mass']) + 10
    mask_mass = (mass >= m_low) & (mass <= m_up)
    cat_mass = cat[mask_mass]
    get_corr(cat_mass, op.join(save_dir, f'corr_fof_all_m_{m_low}_{m_up}.h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='type of halo catalog, fof or subhalo')
    args = parser.parse_args()
    if args.type == 'fof':
        fof()
    elif args.type == 'subhalo':
        subhalos()
