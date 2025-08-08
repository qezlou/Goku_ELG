import numpy as np
from gal_goku_sims import hmf as halo_mass_func

merge = False
z = 0.5
for narrow in [True]:
   for fid in ['HF', 'L2']:
      save_File = f'/scratch/06536/qezlou/Goku/processed_data/HMF/{fid}_hmfs_{np.round(z,1)}'
      if narrow:
         base_dir = f'/scratch/06536/qezlou/Goku/FOF/{fid}/'
         save_File += '_narrow'
      else:
         base_dir = f'/scratch/06536/qezlou/Goku/FOF/{fid}/'
      if merge:
         save_File += '.hdf5'
      else:
         save_File +='_no_merge.hdf5'
      print(save_File)
      hmf = halo_mass_func.Hmf()
      hmf.get_all_fof_hmfs(base_dir=base_dir, save_file=save_File, narrow=narrow, merge= merge, z=z)
      print(f'fid = {fid} is done!')
