import argparse
import numpy as np
import importlib
from gal_goku import emus_multifid


def run_it(ind_test, z, machine='stampede3'):
    
    num_inducing=500
    num_latents=14
    w_type='fixed_independent'  # 'diagonal' or 'pca'
    #w_type='diagonal'

    train_subdir = 'HMF/train_independent_l14_full_comp_kernel/'
    #train_subdir = 'HMF/train_l14_full_comp_kernel/'

    if machine=='stampede3':
        data_dir = '/scratch/06536/qezlou/Goku/processed_data/'
    elif machine=='ucr':
        data_dir = '/rhome/mqezl001/bigdata/HETDEX/data/'
    elif machine=='pc':
        data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/'

    z = np.round(z, 1)
    emu = emus_multifid.HmfNativeBins(data_dir=data_dir,
                                      z=z,
                                      num_inducing=num_inducing, 
                                      num_latents=num_latents,
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
            opt_params={'max_iters':12_000, 'initial_lr':5e-3, 'iter_save':500}, 
            model_file=model_file,
            composite_kernel=['matern32', 'matern52', 'matern32', 'matern52'],
            w_type=w_type
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HMF LOOCV')
    parser.add_argument('--ind_test', default=None, type=int, help='')
    parser.add_argument('--z', default=2.5, type=float, help='Redshift')
    parser.add_argument('--machine', default='stampede3', type=str, help='Machine name')

    args = parser.parse_args()
    run_it(args.ind_test, z=args.z, machine=args.machine)
