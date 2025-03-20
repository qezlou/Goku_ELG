import numpy as np
import importlib
import matplotlib.pyplot as plt
from gal_goku import emus_multifid
importlib.reload(emus_multifid)

data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
emu_type = {'wide_and_narrow':True}
ind_test = np.array([2,12,18, 25, 35])
ind_train=np.delete(np.arange(36), ind_test)
mass_pair = (11.8,11.8)
model_file = 'Xi_Native_emu_mpair_11.8_11.8_wide_narrow.pkl'
xi_emu = emus_multifid.XiNativeBins(data_dir, mass_pair=mass_pair, logging_level='INFO', emu_type=emu_type)
mean, var = xi_emu.predict(ind_train=ind_train, ind_test=ind_test, model_file=model_file)
rbins = xi_emu.mbins