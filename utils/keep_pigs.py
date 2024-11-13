from os import path
from os import system
from glob import glob
import numpy as np
import argparse

def cleanup(fid='L1'):
    fid_dir = f'/scratch/06536/qezlou/Goku/FOF/{fid}/'
    sim_dirs = glob(f'{fid_dir}cosmo*/')
    print(f'nun of sims in fid {fid}: {len(sim_dirs)}')

    for sd in sim_dirs:
        print(f'sim = {sd}')
        snaps = np.loadtxt(f'{sd}output/Snapshots.txt')
        snap_id = snaps[:,0].astype(int)
        snap_z  = 1/snaps[:,1] - 1
        # Remove all snaps with z > 3.5
        mask = snap_z > 3.5

        #remove matter power and
        system(f'rm {sd}output/snap*.fits')
        system(f'rm -rf {sd}output/PART*')
        print(f'removing {snap_id[mask]}')

        for remove in snap_id[mask]:
            system(f"rm -rf {sd}output/PIG_{str(remove).rjust(3,'0')}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fid', type=str, required=True)
    args = parser.parse_args()
    cleanup(args.fid)

