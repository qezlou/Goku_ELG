import numpy as np
import h5py
from gal_goku_sims import xi
from mpi4py import MPI
import os
import os.path as op
import argparse

def run_it(fid, narrow, num_chunks, chunk):
    
    comm_size = MPI.COMM_WORLD.Get_size()
    corr = xi.Corr(ranks_for_nbkit=comm_size)


    basedir = f'/scratch/06536/qezlou/Goku/FOF/{fid}'
    if narrow:
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/xi_bins/{fid}/narrow'
    else:    
        save_dir = f'/scratch/06536/qezlou/Goku/processed_data/xi_bins/{fid}'
    if not op.exists(save_dir):
        os.mkdir(save_dir)


    corr.get_corr_on_grid(base_dir=basedir, save_dir=save_dir, narrow=narrow, chunk=chunk, num_chunks=num_chunks)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--fid', required=False, default='L2', type=str, help='')
    parser.add_argument('--narrow', required=False, default=0, type=int, help='')
    parser.add_argument('--numchunks', required=False, default=20, type=int, help='')
    parser.add_argument('--chunk', required=True, type=int, help='')
    

    args = parser.parse_args()
    run_it(args.fid, args.narrow, args.numchunks, args.chunk)