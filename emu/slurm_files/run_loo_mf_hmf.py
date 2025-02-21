from  mpi4py import MPI
from gal_goku import emus_multifid as emus

#data_dir = '/rhome/mqezl001/bigdata/HETDEX/data/HMF/'
#data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/'
data_dir = '/scratch/06536/qezlou/Goku/processed_data/HMF/'


emu_type = {'multi-fid':True, 'single-bin':True, 'linear':True, 'wide_and_narrow':False }
narrow = 0 # Test on Goku-narrow sims or not
no_merge = True # If `no_merge` is True, the emulator won't merge the bins

if emu_type['linear']:
    savefile = f'{data_dir}train/loo_L2_wide_linear_mf_no_merge_on_coeffs.hdf5'
else:
    savefile = f'{data_dir}train/loo_L2_wide_non_linear_mf_no_merge_on_coeffs.hdf5'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f'rank = {rank}', flush=True)
#savefile = '/home/qezlou/HD2/HETDEX/cosmo/data/HMF/train/loo_L2_mf_no_merge.hdf5'

emu = emus.Hmf(data_dir=data_dir,
               emu_type=emu_type, 
               no_merge=no_merge,
               logging_level='INFO')
emu.loo_train_pred(savefile=savefile)
#emu.train_pred_all_sims(savefile=savefile)