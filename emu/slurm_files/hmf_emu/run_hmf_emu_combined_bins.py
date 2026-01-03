import argparse
import numpy as np
from gal_goku import emus_multifid
import json
import os.path as op


def run_it(ind_test, z, train_subdir, machine='stampede3',
           norm_type='subtract_mean', noise_floor=0.0,
           flow_max_iters=20000, flow_initial_lr=1e-3, flow_scheduler_type='plateau',
           flow_scheduler_gamma=0.5, flow_scheduler_patience=150, flow_scheduler_min_lr=1e-6,
           flow_early_stopping_patience=400, flow_early_stopping_min_delta=1e-4,
           flow_batch_size=128, flow_num_bijectors=6, flow_hidden_units=(256, 256), 
           flow_log_every=500, flow_num_samples=512):


    if machine=='stampede3':
        data_dir = '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
    elif machine=='vista':
        data_dir = '/scratch/06536/qezlou/goku/processed_data/xi_on_grid/'
    elif machine=='ucr':
        data_dir = '/rhome/mqezl001/bigdata/HETDEX/data/xi_bins/'
    elif machine=='pc':
        data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
    else:
        raise ValueError('machine not recognized')

    # Save the config file to the save directory
    json.dump({
        'train_subdir': train_subdir,
        'norm_type': norm_type,
        'flow_max_iters': flow_max_iters,
        'flow_initial_lr': flow_initial_lr,
        'flow_scheduler_type': flow_scheduler_type,
        'flow_scheduler_gamma': flow_scheduler_gamma,
        'flow_scheduler_patience': flow_scheduler_patience,
        'flow_scheduler_min_lr': flow_scheduler_min_lr,
        'early_stopping_patience': flow_early_stopping_patience,
        'early_stopping_min_delta': flow_early_stopping_min_delta,
        'flow_batch_size': flow_batch_size,
        'flow_num_bijectors': flow_num_bijectors,
        'flow_hidden_units': list(flow_hidden_units),
        'flow_log_every': flow_log_every,
        'flow_num_samples': flow_num_samples,
        'noise_floor': noise_floor
    }, open(op.join(data_dir, train_subdir, 'config.json'), 'w'))
    

    emu = emus_multifid.HmfNativeBins(data_dir=data_dir,
                                     z=z,
                                     norm_type=norm_type,
                                     noise_floor=noise_floor,
                                     logging_level='DEBUG')
    if ind_test is None:
        ind_train = None
        model_file=f'hmf_emu_combined_z{z}_all'

    else:
        ind_train = np.delete(np.arange(emu.Y[1].shape[0]), [ind_test])
        model_file=f'xi_emu_combined_z{z}_leave{ind_test}'
    
    emu.logger.info(f'will save on {model_file}')
    
    emu.train(ind_train,
              train_subdir=train_subdir, 
              opt_params={
                'max_iters': flow_max_iters,
                'initial_lr': flow_initial_lr,
                'lr_scheduler_type': flow_scheduler_type,
                'lr_scheduler_gamma': flow_scheduler_gamma,
                'lr_scheduler_patience': flow_scheduler_patience,
                'lr_scheduler_min_lr': flow_scheduler_min_lr,
                'early_stopping_patience': flow_early_stopping_patience,
                'early_stopping_min_delta': flow_early_stopping_min_delta,
                'batch_size': flow_batch_size,
                'num_bijectors': flow_num_bijectors,
                'hidden_units': flow_hidden_units,
                'log_every': flow_log_every,
                'num_samples': flow_num_samples
                }, 
            model_file=model_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Xi LOOCV')
    parser.add_argument('--ind_test', default=None, type=int, help='')
    parser.add_argument('--z', default=2.5, type=float, help='Redshift')
    parser.add_argument('--machine', default='stampede3', type=str, help='Machine name')
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')

    args = parser.parse_args()
    # load the config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    args = parser.parse_args()
    run_it(args.ind_test, z=args.z, train_subdir=config['train_subdir'], 
           machine=args.machine,
           norm_type=config['norm_type'], 
           noise_floor=config.get('noise_floor', 0.0),
           flow_max_iters=config.get('flow_max_iters', 20000),
           flow_initial_lr=config.get('flow_initial_lr', 1e-3),
           flow_scheduler_type=config.get('flow_scheduler_type', 'plateau'),
           flow_scheduler_gamma=config.get('flow_scheduler_gamma', 0.5),
           flow_scheduler_patience=config.get('flow_scheduler_patience', 150),
           flow_scheduler_min_lr=config.get('flow_scheduler_min_lr', 1e-6),
           flow_early_stopping_patience=config.get('flow_early_stopping_patience', 400),
           flow_early_stopping_min_delta=config.get('flow_early_stopping_min_delta', 1e-4),
           flow_batch_size=config.get('flow_batch_size', 128),
           flow_num_bijectors=config.get('flow_num_bijectors', 6),
           flow_hidden_units=tuple(config.get('flow_hidden_units', (256, 256))),
           flow_log_every=config.get('flow_log_every', 500),
           flow_num_samples=config.get('flow_num_samples', 512))
