import argparse
import numpy as np
import importlib
from gal_goku import emus_multifid
import json
import os.path as op


def run_it(ind_test, z, train_subdir, machine='stampede3', num_latents=14, w_type='diagonal', norm_type='subtract_mean', noise_floor=0.0, loss_type='gaussian'):
    """Run the emulator training and prediction.
    """
    
    num_inducing=500


    
    #train_subdir = 'HMF/train_l14_full_comp_kernel/'

    if machine=='stampede3':
        data_dir = '/scratch/06536/qezlou/Goku/processed_data/'
    elif machine=='ucr':
        data_dir = '/rhome/mqezl001/bigdata/HETDEX/data/'
    elif machine=='pc':
        data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/'

    # Save the config file to the save directory
    json.dump({
        'train_subdir': train_subdir,
        'num_latents': num_latents,
        'w_type': w_type,
        'norm_type': norm_type
    }, open(op.join(data_dir, train_subdir, 'config.json'), 'w'))

    if loss_type=='poisson':
        get_counts = True
    else:
        get_counts = False
    z = np.round(z, 1)
    emu = emus_multifid.HmfNativeBins(data_dir=data_dir,
                                      z=z,
                                      num_inducing=num_inducing, 
                                      num_latents=num_latents,
                                      norm_type=norm_type,
                                      noise_floor=noise_floor,
                                      get_counts= get_counts,
                                      logging_level='DEBUG')
    if ind_test is None:
        ind_train = None
        model_file=f'hmf_emu_combined_z{z}_inducing_{int(num_inducing)}_latents_{int(num_latents)}_all.pkl'

    else:
        ind_train = np.delete(np.arange(emu.Y[1].shape[0]), [ind_test])
        model_file=f'hmf_emu_combined_z{z}_inducing_{int(num_inducing)}_latents_{int(num_latents)}_leave{ind_test}.pkl'

    emu.logger.info(f'will save on {model_file}')
    
    emu.train(ind_train,
            train_subdir=train_subdir, 
            opt_params={'max_iters':12_000, 'initial_lr':5e-3, 'iter_save':12_000}, 
            model_file=model_file,
            composite_kernel=['matern32', 'matern52', 'matern32', 'matern52'],
            w_type=w_type,
            loss_type=loss_type
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HMF LOOCV')
    parser.add_argument('--ind_test', default=None, type=int, help='')
    parser.add_argument('--z', default=2.5, type=float, help='Redshift')
    parser.add_argument('--machine', default='stampede3', type=str, help='Machine name')
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')

    args = parser.parse_args()
    # load the config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    args = parser.parse_args()
    run_it(args.ind_test, z=args.z, train_subdir=config['train_subdir'], machine=args.machine, num_latents=config['num_latents'], w_type=config['w_type'], norm_type=config['norm_type'], noise_floor=config.get('noise_floor', 0.0), loss_type=config.get('loss_type', 'gaussian'))
