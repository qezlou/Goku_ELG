import numpy as np
import h5py
from gal_goku import emus_multifid
from gal_goku import summary_stats
import os.path as op
import argparse
import warnings
warnings.filterwarnings('ignore')
from gal_goku import mpi_helper
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def loo_mean_err_wide_narrow(data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'):

    # Get a list of mass_pairs
    xi = summary_stats.Xi(data_dir, fid='HF', MPI=None, logging_level='ERROR')
    all_mass_pairs = xi.mass_pairs
    all_frac_errs = np.zeros((len(all_mass_pairs), 36, 26), dtype=np.float32)
    starts, ends = mpi_helper.into_chunks(comm, all_mass_pairs.shape[0])
    s, e = starts[rank], ends[rank]
    bad_sims = np.zeros((len(all_mass_pairs), 36), dtype=bool)
    print(f'rank {rank} | s: {s}, e: {e}', flush=True)
    for i,m in enumerate(range(s, e)):
        #print(f'rank {rank} | mass_pair: {all_mass_pairs[m]}', flush=True)
        print(f'rank {rank} | prog = {100*i/(e-s):.1f}%', flush=True)
        xi_emu = emus_multifid.XiNativeBins(data_dir, interp='spline', mass_pair=all_mass_pairs[m], logging_level='ERROR', emu_type={'wide_and_narrow':True})
        for sim in range(36):
            model_file = f'Xi_Native_emu_mapirs2_spline_{all_mass_pairs[m][0]}_{all_mass_pairs[m][1]}_wide_narrow_leave_{sim}.pkl'
            
            # Make sure the test sim is not missing
            try:
                # Predict
                mean, var = xi_emu.predict(ind_test=np.array([sim]), model_file=model_file)
                all_frac_errs[m, sim, :] = 10**mean[0]/10**xi_emu.Y[1][sim] -1
                rbins = xi_emu.mbins
            except FileNotFoundError:
                bad_sims[m, sim] = True
                print(f'{model_file} not found')
    
    comm.Barrier()
    # MPI communication
    ## bad sims
    bad_sims = np.ascontiguousarray(bad_sims, dtype=bool)
    comm.Allreduce(MPI.IN_PLACE, bad_sims, op=MPI.LOR)
    # all the frac errs
    all_frac_errs = np.ascontiguousarray(all_frac_errs, dtype=np.float32)
    comm.Allreduce(MPI.IN_PLACE, all_frac_errs, op=MPI.SUM)
    if rank == 0:
        all_frac_errs[bad_sims] = np.nan
        print('Saving...', flush=True)
        with h5py.File(op.join(data_dir, 'train', 'median_loo_err.hdf5'),'w') as f:
            f.create_dataset('frac_errs', data=all_frac_errs)
            f.create_dataset('rbins', data=rbins)
    comm.Barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    loo_mean_err_wide_narrow(args.data_dir)