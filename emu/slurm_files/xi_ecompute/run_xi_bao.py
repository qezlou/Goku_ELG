import numpy as np
import h5py
from gal_goku_sims import xi
from mpi4py import MPI
import os
import os.path as op
import argparse

fid = 'HF'
z = 2.5
mass_th = 10**np.array([11.6, 11.6])
r_edges = np.arange(80, 150, 5)

base_dir=f'/scratch/06536/qezlou/Goku/FOF/{fid}'
save_file = '/scratch/06536/qezlou/Goku/processed_data/test/'

comm_size = MPI.COMM_WORLD.Get_size()
corr = xi.Corr(ranks_for_nbkit=comm_size, logging_level='DEBUG')

pigs = corr.get_pig_dirs(base_dir, z=z, narrow=True)

ind = -1
cosmo = corr.get_cosmo(pigs['params'][ind])
corr_hh, mbins = corr._get_corr(pig_dir=pigs['pig_dirs'][ind], cosmo=cosmo, mass_th=mass_th, r_edges=r_edges, z=z)

save_file = op.join(save_file, f'xi_hh_large_scale_{pigs["sim_tags"][ind]}.h5')
corr._save_corr_on_grid(corr_hh, mbins, mass_th, pigs['sim_tags'][ind], save_file)

