from gal_goku import emus_multifid as emus

data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/'

savefile = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/train/loo_HF_L2_single_bin_nonlinear.hdf5'

emu_type = {'multi-fid':True, 'single-bin':True, 'linear':True }

emu = emus.Hmf(data_dir=data_dir,fid =['L2', 'HF'], emu_type=emu_type, narrow=False)
#emu.loo_train_pred(savefile=savefile)
emu.train_pred_all_sims()