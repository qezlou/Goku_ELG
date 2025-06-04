import numpy as np
import h5py
from gal_goku_sims import xi
from mpi4py import MPI
import os
import os.path as op
import argparse

def run_it(fid, narrow, num_chunks, chunk, stat_type):
    
    comm_size = MPI.COMM_WORLD.Get_size()
    corr = xi.Corr(ranks_for_nbkit=comm_size)


    basedir = f'/scratch/06536/qezlou/Goku/FOF/{fid}'
    if narrow == 0 and stat_type == 'xi_hh':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/xi_bins/{fid}'
    elif narrow == 1 and stat_type == 'xi_hh':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/xi_bins/{fid}/narrow'
    elif narrow == 0 and stat_type == 'pk_hh':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/power_bins/{fid}'
    elif narrow == 1 and stat_type == 'pk_hh':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/power_bins/{fid}/narrow'
    elif narrow == 0 and stat_type == 'pk_hm':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/power_hm_bins/{fid}'
    elif narrow == 1 and stat_type == 'pk_hm':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/power_hm_bins/{fid}/narrow'
    elif narrow == 0 and stat_type == 'pk_hlin':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/power_hlin_bins/{fid}'
    elif narrow == 1 and stat_type == 'pk_hlin':
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/power_hlin_bins/{fid}/narrow'
    else:
        raise ValueError(f"Unexpected combination of narrow={narrow} and stat_type={stat_type}")

    if not op.exists(save_dir):
        raise ValueError(f"Save directory {save_dir} does not exist. Please create it before running the script.")

    if stat_type == 'xi_hh':
        corr.get_corr_on_grid(base_dir=basedir, save_dir=save_dir, narrow=narrow, chunk=chunk, num_chunks=num_chunks)
    elif stat_type == 'pk_hh':
        corr.get_power_on_grid(base_dir=basedir, save_dir=save_dir, narrow=narrow, power_type='hh', chunk=chunk, num_chunks=num_chunks)
    elif stat_type == 'pk_hm':
        corr.get_power_on_grid(base_dir=basedir, save_dir=save_dir, narrow=narrow, power_type='hm', chunk=chunk, num_chunks=num_chunks)
    elif stat_type == 'pk_hlin':
        corr.get_power_on_grid(base_dir=basedir, save_dir=save_dir, narrow=narrow, power_type='hlin', chunk=chunk, num_chunks=num_chunks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--fid', required=False, default='L2', type=str, help='')
    parser.add_argument('--narrow', required=False, default=0, type=int, help='')
    parser.add_argument('--numchunks', required=False, default=20, type=int, help='')
    parser.add_argument('--chunk', required=True, type=int, help='')
    parser.add_argument('--stat_type', required=False, default='xi_hh', type=str, help='To compute "xi_hh", "pk_hh", "pk_hm" or "pk_hlin"')
    

    args = parser.parse_args()
    run_it(args.fid, args.narrow, args.numchunks, args.chunk, stat_type=args.stat_type)