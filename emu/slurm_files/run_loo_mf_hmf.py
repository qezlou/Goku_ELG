from gal_goku import emus_multifid as emus

data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/'



emu_type = {'multi-fid':True, 'single-bin':True, 'linear':True }
narrow = 0 # Test on Goku-narrow sims or not
no_merge = True # If `no_merge` is True, the emulator won't merge the bins

savefile = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/train/loo_L2_mf_no_merge.hdf5'

emu = emus.Hmf(data_dir=data_dir,
               fid =['L2', 'HF'], 
               emu_type=emu_type, 
               narrow=narrow, 
               no_merge=no_merge)
emu.loo_train_pred(savefile=savefile)
#emu.train_pred_all_sims(savefile=savefile)