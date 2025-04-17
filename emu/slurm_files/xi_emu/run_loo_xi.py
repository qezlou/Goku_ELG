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




def run_it(num_chunks, chunk):
    #data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
    data_dir = '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
    emu_type = {'wide_and_narrow':True}

    xi = summary_stats.Xi(data_dir=data_dir, fid = 'HF',  narrow=False, logging_level='ERROR')
    # Distribute the work load of iterating over
    # all mass piars across available number of ranks
    # all mass piars across available number of ranks
    all_mass_pairs = xi.mass_pairs
    #all_mass_pairs = np.array([(12.8, 12.8)])
    start_chunk, end_cunk = distribute(xi.mass_pairs.shape[0], num_chunks, chunk)
    # Get the start and end index for this rank
    s, e = distribute(end_cunk-start_chunk, size, rank)
    s = start_chunk + s
    e = start_chunk + e
    print(f'rank = {rank}, (s, e)= {(s,e)} from {(start_chunk, end_cunk)}', flush=True)
    start_chunk, end_cunk = distribute(xi.mass_pairs.shape[0], num_chunks, chunk)
    # Get the start and end index for this rank
    s, e = distribute(end_cunk-start_chunk, size, rank)
    s = start_chunk + s
    e = start_chunk + e
    print(f'rank = {rank}, (s, e)= {(s,e)} from {(start_chunk, end_cunk)}', flush=True)

    for i in range(s,e):
        start_time = time.time()
        start_time = time.time()
        mass_pair = all_mass_pairs[i]
        xi_emu = emus_multifid.XiNativeBins(data_dir,  interp='spline', mass_pair=mass_pair, logging_level='INFO', emu_type=emu_type)
        # Iterate over the leave-one-out simulations
        # Only use the good simulations, the ones with enough
        # non-nan values
        for c, j in enumerate(range(xi_emu.good_sim_ids[-1].size)):
            sim = xi_emu.good_sim_ids[-1][j]
            model_file= f'Xi_Native_emu_mapirs2_spline_{mass_pair[0]}_{mass_pair[1]}_wide_narrow_leave_{sim}.pkl'
            num_hf_sims = xi_emu.Y[1].shape[0]
            ind_train=np.delete(np.arange(xi_emu.Y[-1].shape[0]), np.array([c]))
        # Iterate over the leave-one-out simulations
        # Only use the good simulations, the ones with enough
        # non-nan values
        for c, j in enumerate(range(xi_emu.good_sim_ids[-1].size)):
            sim = xi_emu.good_sim_ids[-1][j]
            model_file= f'Xi_Native_emu_mapirs2_spline_{mass_pair[0]}_{mass_pair[1]}_wide_narrow_leave_{sim}.pkl'
            num_hf_sims = xi_emu.Y[1].shape[0]
            ind_train=np.delete(np.arange(xi_emu.Y[-1].shape[0]), np.array([c]))
            # The optimization hyper parameters
            opt_params = get_opt_params(mass_pair)
            xi_emu.train(ind_train=ind_train, model_file=model_file, opt_params=opt_params)
            end_time = time.time()
        print(f'rank {rank} | Progress: {i-s}/{e-s} |  mass_pair: {mass_pair}, elappsed {((end_time - start_time)/60):.1f} minutes, {xi_emu.good_sim_ids[-1].size} sims', flush=True)
    comm.Barrier()
            end_time = time.time()
        print(f'rank {rank} | Progress: {i-s}/{e-s} |  mass_pair: {mass_pair}, elappsed {((end_time - start_time)/60):.1f} minutes, {xi_emu.good_sim_ids[-1].size} sims', flush=True)
    comm.Barrier()

def get_opt_params(mass_pair):
    """
    Larger masses requite harder training. 
    Get the `max_iters` and `initial_lr` for 
    ptimization process
    """
    if mass_pair[0] >= 12.0:
        opt_params = {'max_iters':30_000, 'initial_lr':5e-3}
    else:
        opt_params = {'max_iters':20_000, 'initial_lr':5e-3}
    return opt_params

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