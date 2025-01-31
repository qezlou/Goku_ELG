from gal_goku import emus

data_dir = '/scratch/06536/qezlou/Goku/processed_data/HMF/'
y_log = 1 # Train the emulator on the log of the correlation function
multi_bin = 0 # Build one emualotr per rp bin
narrow = 0 # Test on Goku-narrow sims or not
no_merge = False # If `no_merge` is True, the emulator won't merge the bin s

savefile = f'{data_dir}train/loo_L2_narrow.hdf5'

emu = emus.Hmf(data_dir=data_dir, y_log=y_log, fid='L2', multi_bin=multi_bin, narrow=narrow, no_merge=no_merge, logging_level='INFO')
#emu.loo_train_pred(savefile=savefile)
emu.train_pred_all_sims(data_dir=data_dir)