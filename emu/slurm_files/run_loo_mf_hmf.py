from gal_goku import emus_multifid as emus

data_dir = '/scratch/06536/qezlou/Goku/processed_data/HMF/'
#data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/'


emu_type = {'multi-fid':True, 'single-bin':True, 'linear':False }
narrow = 0 # Test on Goku-narrow sims or not
no_merge = True # If `no_merge` is True, the emulator won't merge the bins

savefile = '/scratch/06536/qezlou/Goku/processed_data/HMF/train/traine_all_L2_non_linear_mf_no_merge.hdf5'
#savefile = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/train/loo_L2_mf_no_merge.hdf5'

emu = emus.Hmf(data_dir=data_dir,
               emu_type=emu_type, 
               narrow=narrow, 
               no_merge=no_merge)
#emu.loo_train_pred(savefile=savefile)
emu.train_pred_all_sims(savefile=savefile)