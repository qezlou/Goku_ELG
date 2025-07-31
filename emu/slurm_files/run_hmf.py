import numpy as np
from gal_goku import hmf as halo_mass_func

narrow= True
merge = False
for fid in ['L2']:
    save_File = f'/scratch/06536/qezlou/Goku/processed_data/HMF/{fid}_hmfs'
    if narrow:
        base_dir = f'/scratch/06536/qezlou/Goku/FOF/{fid}/narrow/'
        save_File += '_narrow'
    else:
        base_dir = f'/scratch/06536/qezlou/Goku/FOF/{fid}/'
    if merge:
        save_File += '.hdf5'
    else:
        save_File +='_no_merge.hdf5'
    print(save_File)
    hmf = halo_mass_func.Hmf()
    hmf.get_all_fof_hmfs(base_dir=base_dir, save_file=save_File, narrow=narrow, merge= merge)
    print(f'fid = {fid} is done!')