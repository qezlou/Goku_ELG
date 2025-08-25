from gal_goku_sims import xi

base_dir = '/scratch/06536/qezlou/Goku/FOF/HF/'
z = 2.5
narrow = False

corr= xi.Corr()
pigs = corr.get_pig_dirs(base_dir, z=2.5, narrow=narrow)

for i in range(len(pigs['pig_dirs'])):
    cosmo = corr.get_cosmo(pigs['params'][i])
    halos = corr.load_halo_cat(pigs['pig_dirs'][i], cosmo=cosmo, ex_rad_fac=2)
