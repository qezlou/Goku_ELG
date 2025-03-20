import numpy as np
import h5py
from glob import glob
import os.path as op
import os
import argparse

def combine_xi_files(fid, data_dir, narrow=False):
    """
    Combine individually saved correlation function files into a single file
    :param fid: str, folder name
    :param data_dir: str, path to the data directory
    :param narrow: bool, whether to use the narrow bins
    """
    if narrow:
        save_file = op.join(data_dir, fid, 'narrow', f'xi_grid_{fid}_narrow.hdf5')
    else:
        save_file = op.join(data_dir, fid, f'xi_grid_{fid}.hdf5')


    assert not os.path.exists(save_file), f'{save_file} already exists'
    if narrow:
        corr_dir = op.join(data_dir, fid, 'narrow')
    else:
        corr_dir = op.join(data_dir, fid)
    corr_files = [op.join(corr_dir,f) for f in os.listdir(corr_dir) if  f.endswith('.hdf5')]
    print(len(corr_files))
    sim_tags = []
    corrs = []
    for cfile in corr_files:
        with h5py.File(cfile, 'r') as f:
            sim_tags.append(f['sim_tag'][()])
            rbins = f['mbins'][:]
            corrs.append(f['corr'][:])
            mass_pairs = f['pairs'][:]


    with h5py.File(save_file, 'w') as f:
        f.create_dataset('sim_tags', data=np.array(sim_tags, dtype='S'))
        f.create_dataset('mbins', data=rbins)
        f.create_dataset('corr', data=np.array(corrs))
        f.create_dataset('mass_pairs', data=mass_pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fid', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--narrow', type=bool, default=False)
    args = parser.parse_args()
    combine_xi_files(args.fid, args.data_dir, args.narrow)