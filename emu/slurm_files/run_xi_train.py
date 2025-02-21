import numpy as np
import h5py
from gal_goku_sims import xi_train
from mpi4py import MPI
import os.path as op
import argparse

def run_it(save_file='test.hdf5'):
    
    comm_size = MPI.COMM_WORLD.Get_size()
    corr = xi_train.Corr(ranks_for_nbkit=comm_size)


    sim_tag = 'cosmo_10p_Box250_Part750_0000'

    basedir = '/scratch/06536/qezlou/Goku/FOF/L2'
    pig_dir = f'{basedir}/{sim_tag}/output/PIG_003/'
    save_dir = '/scratch/06536/qezlou/Goku/processed_data/corrs_bins'
    assert op.exists(save_dir)
    save_file = op.join(save_dir,save_file)
    mass_thresh = (1e11, 3e11)

    corr_fof, mbins = corr._get_corr(pig_dir,mass_thresh)

    if corr.rank ==0:
        corr.logger.info(f'save_file = {save_file}')
        with h5py.File(save_file, 'w') as f:
            f['mbins'] = mbins
            f['corr'] = corr_fof
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')

    args = parser.parse_args()
    run_it()