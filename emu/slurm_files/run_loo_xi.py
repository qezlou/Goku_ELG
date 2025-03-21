"""
Run Leave-One-Out (LOO) cross-validation for xi(r,n1, n2)
"""

import numpy as np
import importlib
from gal_goku import emus_multifid
importlib.reload(emus_multifid)
import time

start_time = time.time()
data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
mass_pair = (11.8, 11.8)
emu_type = {'wide_and_narrow':True}

xi_emu = emus_multifid.XiNativeBins(data_dir,  interp='spline', mass_pair=mass_pair, logging_level='ERROR', emu_type=emu_type)
num_hf = xi_emu.Y[1].shape[0]

i = 25
ind_test = np.array([i])
ind_train=np.delete(np.arange(num_hf), ind_test)
xi_emu.train(ind_train=ind_train, ind_test=ind_test, model_file=f'Xi_Native_emu_spline_{mass_pair[0]}_{mass_pair[1]}._leave{i}pkl')


ind_test = np.array([2,12,14])
ind_train=np.delete(np.arange(36), ind_test)
xi_emu.train(ind_train=ind_train, ind_test=ind_test, model_file='Xi_Native_emu_mapirs2_spline_11.8_11.8.pkl')

end_time = time.time()
print(f'Elapsed time: {(end_time-start_time)/60} m')