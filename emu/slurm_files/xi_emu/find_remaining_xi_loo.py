"""
Run Leave-One-Out (LOO) cross-validation for xi(r,n1, n2)
"""
import pickle
import numpy as np
import os.path as op
from gal_goku import summary_stats
import time


def run_it(num_chunks=1, chunk=0):
    #data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
    data_dir = '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
    emu_type = {'wide_and_narrow':True}

    xi = summary_stats.Xi(data_dir=data_dir, fid = 'HF',  narrow=False, logging_level='ERROR')
    # Distribute the work load of iterating over
    # all mass piars across avaoilable number of ranks
    all_mass_pairs = xi.mass_pairs

    curr, goal =  get_remaining_sims(data_dir, all_mass_pairs, num_sims=36)
    return curr, goal


def get_opt_params(mass_pair):
    """
    Larger masses requite harder training. 
    Get the `max_iters` and `initial_lr` for 
    ptimization process
    """
    #if mass_pair[0] >= 12.5:
    #    opt_params = {'max_iters':40_000, 'initial_lr':5e-3}
    if mass_pair[0] >= 12.0:
        opt_params = {'max_iters':10_000, 'initial_lr':5e-3}
    else:
        opt_params = {'max_iters':4_000, 'initial_lr':5e-3}
    return opt_params

def get_remaining_sims(data_dir, all_mass_pairs, num_sims=36):
    """
    Get the sims that have not yet reached the max_iters
    """
    s_time= time.time()
    remaing = []
    current_iters = []
    goal_iters = []
    mpair_ind_grid, sim_grid = np.meshgrid(np.arange(all_mass_pairs.shape[0]), np.arange(num_sims))
    for i in range(mpair_ind_grid.size):
        mass_pair = all_mass_pairs[mpair_ind_grid.flatten()[i],:]
        sim = int(sim_grid.flatten()[i])
        attr_file= f'Xi_Native_emu_mapirs2_spline_{mass_pair[0]}_{mass_pair[1]}_wide_narrow_leave_{sim}.pkl.attrs'
        attr_file = op.join(data_dir, 'train', attr_file)
        max_iters = get_opt_params(mass_pair)['max_iters']
        try:
            with open(attr_file, 'rb') as f:
                attr = pickle.load(f)
                if len(attr['loss_history']) >= max_iters:
                    continue
                else:
                    #print(f'{len(attr['loss_history'])} < {max_iters}')
                    remaing.append(i)
                    current_iters.append(len(attr['loss_history']))
                    goal_iters.append(max_iters)
        except FileNotFoundError:
            print(f'attr file not found: {attr_file}', flush=True)
            continue
    remaing = np.array(remaing)
    e_time = time.time()
    print(f'Found {remaing.size}/{mpair_ind_grid.size} remaining training. Computed in {(e_time-s_time)/60} mins', flush=True)
    #print(current_iters, goal_iters)
    current_iters = np.array(current_iters)
    goal_iters = np.array(goal_iters)
    curr_sorted = []
    for g_it in [40_000, 30_000, 20_000]:
        ind = np.where(goal_iters=g_it)
        curr_sorted.append(current_iters[ind])
    return current_iters, goal_iters
    #return mpair_ind_grid.flatten()[remaing], sim_grid.flatten()[remaing]