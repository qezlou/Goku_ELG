from gal_goku import wp_emus


data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/corr_projected_corrected/'
y_log = 1 # Train the emulator on the log of the correlation function
r_range = (0, 30) # The range of r values to consider
multi_bin = 0 # Build one emualotr per rp bin
narrow = 1 # Test on Goku-narrow sims or not

savefile = '/home/qezlou/HD2/HETDEX/cosmo/data/corr_projected_corrected/train/loo_pred_lin_interp_log_y_narrow.hdf5'

emu = wp_emus.SingleFid(data_dir=data_dir, y_log=y_log, r_range=r_range, multi_bin=multi_bin, logging_level='INFO')
emu.loo_train_pred(savefile=savefile, narrow=narrow)