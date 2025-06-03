import numpy as np
import h5py
import importlib
from gal_goku_sims import xi
importlib.reload(xi)

base_dir = '/scratch/06536/qezlou/Goku/FOF/L2/'
z = 2.5
narrow = False

corr= xi.Corr()
pigs = corr.get_pig_dirs(base_dir, z=2.5, narrow=narrow)
#corr.load_halo_cat(pigs['pig_dirs'][0], cosmo=corr.get_cosmo(pigs['params'][0]))
xi_h, rbins = corr._get_corr(pigs['pig_dirs'][10], 
                             cosmo=corr.get_cosmo(pigs['params'][0]), 
                             mass_th=[12.0,12.0], z=z, ex_rad_fac=20)

if corr.nbkit_rank == 0:
    with h5py.File(f'test_xi_ex_rad_fac_20.h5', 'w') as f:
        f.create_dataset('xi_h', data=xi_h)
        f.create_dataset('rbins', data=rbins)
    print(f'xi_h shape: {xi_h.shape}')
    print(f'rbins shape: {rbins.shape}')