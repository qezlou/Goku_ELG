import numpy as np
import h5py
import get_corr
from mpi4py import MPI
import os.path as op
import argparse

def run_it(stat, fp_frac):
    
    comm_size = MPI.COMM_WORLD.Get_size()
    corr = get_corr.Corr(ranks_for_nbkit=comm_size)


    sim_tag = 'cosmo_10p_Box250_Part750_0000'

    basedir = '/scratch/06536/qezlou/Goku/FOF/L2'
    pig_dir = f'{basedir}/{sim_tag}/output/PIG_003/'
    save_dir = '/scratch/06536/qezlou/Goku/processed_data/false_positive/'
    assert op.exists(save_dir)
    if stat=='corr':
        #save_file = op.join(save_dir,f'wp_fof_uniform_fp_percent_{sim_tag}.hdf5')
        assert not op.exists(save_file), f'file exists: {save_file}'

        r_edges = np.logspace(-1.5, np.log10(2), 8)
        r_edges = np.append(r_edges, np.logspace(np.log10(2), np.log10(30), 10)[1:])
        r_edges = np.append(r_edges, np.logspace(np.log10(30), np.log10(80), 50)[1:])


        corr_fof, mbins = corr.get_corr_fof(pig_dir, r_edges, mode='projected', false_positive_ratio=fp_frac)

        if corr.rank ==0:
            corr.logger.info(f'save_file = {save_file}')
            with h5py.File(save_file, 'w') as f:
                f['r'] = np.array([r_edges[i]+r_edges[i+1] for i in range(r_edges.size-1)])
                f['corr'] = corr_fof
    
    elif stat=='power':
        save_file = op.join(save_dir,f'power_fof_uniform_fp_10percent_{sim_tag}.hdf5')
        assert not op.exists(save_file), f'file exists: {save_file}'
        power_fof, mbins = corr.get_corr_fof(pig_dir, stat='power',  mode='1d', false_positive_ratio=fp_frac)  
        if corr.rank ==0:
            corr.logger.info(f'save_file = {save_file}')
            with h5py.File(save_file, 'w') as f:
                f['k'] = mbins
                f['power'] = power_fof


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--stat', type=str, help='corr or power')
    parser.add_argument('--fp_frac', type=float, help='Base directory of the simulations')
    args = parser.parse_args()
    run_it(args.stat, args.fp_frac)