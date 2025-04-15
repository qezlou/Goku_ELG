import numpy as np
import importlib
from gal_goku import emus_multifid
importlib.reload(emus_multifid)

data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
train_subdir = 'train_hetero'
emu = emus_multifid.XiNativeBinsFullDimReduc(data_dir=data_dir, logging_level='DEBUG')
emu.train(train_subdir=train_subdir, 
          opt_params={'max_iters':10_000, 'initial_lr':5e-3},
          num_inducing=100, num_latents=40, 
          model_file=f'xi_emu_combined_inducing_300_latents_100.pkl')

