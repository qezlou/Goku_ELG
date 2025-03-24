import numpy as np
import h5py
import importlib
import matplotlib.pyplot as plt
from gal_goku import emus_multifid
from gal_goku import summary_stats
importlib.reload(emus_multifid)
import os.path as op
import pickle
import warnings
warnings.filterwarnings('ignore')

def loo_mean_err_wide_narrow():
    data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'

    # Get a list of mass_pairs
    xi = summary_stats.Xi(data_dir, fid='HF', logging_level='ERROR')
    all_mass_pairs = xi.mass_pairs
    all_frac_errs = np.full((len(all_mass_pairs), 36, 26), np.nan)
    for m, mass_pair in enumerate(all_mass_pairs):
        print(f'mass_pair: {mass_pair}')
        frac_err_mass_pair = []
        for s in range(36):
            row, col = divmod(s, 4)
            model_file = f'Xi_Native_emu_mapirs2_spline_{mass_pair[0]}_{mass_pair[1]}_wide_narrow_leave_{s}.pkl'
            emu_type = {'wide_and_narrow':True}
        
            # Plot true vs prediction
            xi_emu = emus_multifid.XiNativeBins(data_dir, interp='spline', mass_pair=mass_pair, logging_level='ERROR', emu_type=emu_type)
            # Make sure the test sim is not missing
            try:
                # Predict
                mean, var = xi_emu.predict(ind_test=np.array([s]), model_file=model_file)
                all_frac_errs[m, s, :] = 10**mean[0]/10**xi_emu.Y[1][s] -1
                rbins = xi_emu.mbins
            except FileNotFoundError:
                print(f'{model_file} not found')
            
    
    with h5py.Fileop.join(data_dir, 'train', 'median_loo_err.hdf5', 'w') as f:
        f.create_dataset('frac_errs', data=all_frac_errs)
        f.create_dataset('rbins', data=rbins)

loo_mean_err_wide_narrow()