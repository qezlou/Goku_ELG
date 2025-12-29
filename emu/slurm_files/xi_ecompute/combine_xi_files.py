"""
Combine all individual xi files into one for each fidelity and narrow setting
"""
import numpy as np
import h5py
from os import path as op
from glob import glob
import re
import argparse

#base_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
base_dir = '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
def combine(fid, narrow):

    if narrow:
            save_dir = f'{base_dir}{fid}/narrow'
    else:
            save_dir = f'{base_dir}{fid}'
    print(f'save_dir: {save_dir}')

    for z in [0.0, 0.2, 1.0, 2.0, 3.0, 4.0]:

        box = {'L2': 250, 'HF':1000}
        parts = {'L2': 750, 'HF':3000}

        fnames = glob(op.join(save_dir, f'compressed_*z{z}.hdf5'))
        #print(fnames)
        print(f'Found {len(fnames)} files to combine.')
        numbers = []
        for fname in fnames:
            if narrow:
                match = re.search(r'_(\d+)_narrowz', fname)
            else:
                match = re.search(r'_(\d+)z', fname)
            if match:
                numbers.append(int(match.group(1)))
        #print(numbers)
        ind_nums = np.argsort(numbers)
        fnames_sorted = [fnames[i] for i in ind_nums]
        numbers = sorted(numbers)

        with h5py.File(fnames_sorted[0], 'r') as fr:
            print(fr.keys())
            for key in fr.keys():
                print(key, fr[key].shape)
            #print(fr['sim_tag'][()].decode('utf-8'))
            corr_shape = fr['corr'].shape
            pairs_shape = fr['pairs'].shape
            mbins = fr['mbins'][:]

        corrs = np.zeros((len(fnames_sorted), *corr_shape), dtype=np.float32)
        pairs = np.zeros((len(fnames_sorted), *pairs_shape), dtype=np.float32)
        sim_tags = []

        for fn in fnames_sorted:
            with h5py.File(fn, 'r') as fr:
                if narrow:
                    idx = numbers.index(int(re.search(r'_(\d+)_narrowz', fn).group(1)))
                else:
                    idx = numbers.index(int(re.search(r'_(\d+)z', fn).group(1)))
                corrs[idx] = fr['corr'][:]
                pairs[idx] = fr['pairs'][:]
                try:
                    sim_tags.append(fr['sim_tag'][()].decode('utf-8'))
                except:
                    raise ValueError(f'Error reading sim_tag from file {fn}')
        save_fname = op.join(save_dir, f'all_compressed_10p_Box{box[fid]}_Part{parts[fid]}_narrowz{z}.hdf5' if narrow else f'all_compressed_10p_Box{box[fid]}_Part{parts[fid]}_z{z}.hdf5')

        print(f'Saving combined file to {save_fname}')
        print(f'shapes corrs: {corrs.shape}, pairs: {pairs.shape}, mbins: {mbins.shape}, num sim_tags: {len(sim_tags)}')

        # Write to new file
        with h5py.File(save_fname, 'w') as fw:
            fw.create_dataset('corrs', data=corrs, compression='gzip')
            fw.create_dataset('pairs', data=pairs, compression='gzip')
            fw.create_dataset('mbins', data=mbins)
            fw.create_dataset('sim_id', data=np.array(numbers))
            dt = h5py.string_dtype(encoding='utf-8')
            sim_tag_ds = fw.create_dataset('sim_tags', (len(sim_tags),), dtype=dt)
            for i, tag in enumerate(sim_tags):
                sim_tag_ds[i] = tag


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Combine xi files')
    #parser.add_argument('--z', required=False, default=0.5, type=float, help='Redshift of interest')
    parser.add_argument('--fid', required=False, default='L2', type=str, help='Fidelity level, e.g., L2 or HF')
    parser.add_argument('--narrow', required=False, default=0, type=int, help='0 for standard xi, 1 for narrow xi')

    args = parser.parse_args()
    combine(fid=args.fid, narrow=args.narrow)