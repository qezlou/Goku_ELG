from gal_goku import emus

data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/'
y_log = 1 # Train the emulator on the log of the correlation function
multi_bin = 0 # Build one emualotr per rp bin
narrow = 0 # Test on Goku-narrow sims or not
no_merge = False # If `no_merge` is True, the emulator won't merge the bins

savefile = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/train/loo_L2.hdf5'

emu = emus.Hmf(data_dir=data_dir, y_log=y_log, fid='L2', multi_bin=multi_bin, narrow=narrow, no_merge=no_merge, logging_level='INFO')
emu.loo_train_pred(savefile=savefile)