"""
Run Leave-One-Out (LOO) cross-validation for xi(r,n1, n2)
"""
import argparse
import numpy as np
import importlib
from gal_goku import emus_multifid
from gal_goku import summary_stats
importlib.reload(emus_multifid)
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

def run_it(num_chunks, chunk):
    #data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
    data_dir = '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
    #mass_pair = (11.4,11.4)
    mass_pair = (11.2, 11.2)
    emu_type = {'wide_and_narrow':True}

    xi = summary_stats.Xi(data_dir=data_dir, fid = 'HF',  narrow=False, logging_level='ERROR')
    # Distribute the work load of iterating over
    # all mass piars across avaoilable number of ranks
    all_mass_pairs = xi.mass_pairs
    s, e = distribute(xi.mass_pairs.shape[0], num_chunks, chunk)

    for i in range(s,e):
        mass_pair = all_mass_pairs[i]
        print(f'mass_pair = {mass_pair}')
        xi_emu = emus_multifid.XiNativeBins(data_dir,  interp='spline', mass_pair=mass_pair, logging_level='INFO', emu_type=emu_type)
        model_file= f'Xi_Native_emu_mapirs2_spline_{mass_pair[0]}_{mass_pair[1]}_wide_narrow_leave_{int(rank)}.pkl'
        num_hf_sims = xi_emu.Y[1].shape[0]
        # some HF sims may be exluded as they don't have enough valid xi bins
        # let the ranks with no HF sim wait
        if rank < num_hf_sims:
            ind_train=np.delete(np.arange(num_hf_sims), np.array(int(rank)))
            xi_emu.train(ind_train=ind_train, model_file=model_file)
        comm.Barrier()
    end_time = time.time()

    print(f'rank {rank}, elappsed {(end_time - start_time)/60} minutes')

def distribute(counts, num_chunks, chunk):
    per_chunk = counts//num_chunks
    start = chunk*per_chunk
    if chunk == num_chunks-1:
        end = counts
    else:
        end = start + per_chunk
    
    return start, end

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--num_chunks', required=False, default=10, type=int, help='')
    parser.add_argument('--chunk', required=True, type=int, help='')
    

    args = parser.parse_args()
    run_it(args.num_chunks, args.chunk)