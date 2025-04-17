import numpy as np
import importlib
from gal_goku import emus_multifid
importlib.reload(emus_multifid)

data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
train_subdir = 'train_hetero'
emu = emus_multifid.XiNativeBinsFullDimReduc(data_dir=data_dir,
                                             num_inducing=100, num_latents=40,
                                             logging_level='DEBUG')
ind_train = np.arange(emu.Y[0].shape[0])
emu.train(ind_train,
          train_subdir=train_subdir, 
          opt_params={'max_iters':10_000, 'initial_lr':5e-3}, 
          model_file=f'xi_emu_combined_inducing_300_latents_100.pkl')

