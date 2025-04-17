import argparse
import numpy as np
import importlib
from gal_goku import emus_multifid
importlib.reload(emus_multifid)


def run_it(ind_test, num_inducing=500, num_latents=40):
    #data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
    #train_subdir = 'train_hetero'
    data_dir = '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
    train_subdir = 'train_combined'
    
    emu = emus_multifid.XiNativeBinsFullDimReduc(data_dir=data_dir,
                                                num_inducing=num_inducing, num_latents=num_latents,
                                                logging_level='DEBUG')
    if ind_test is None:
        ind_train = None
        model_file=f'xi_emu_combined_inducing_{int(num_inducing)}_latents_{int(num_latents)}_leave{ind_test}_all.pkl'

    else:
        ind_train = np.delete(np.arange(emu.Y[1].shape[0]), [ind_test])
        model_file=f'xi_emu_combined_inducing_{int(num_inducing)}_latents_{int(num_latents)}_leave{ind_test}.pkl'
    
    emu.logger.info(f'will save on {model_file}')
    
    emu.train(ind_train,
            train_subdir=train_subdir, 
            opt_params={'max_iters':22_000, 'initial_lr':5e-3}, 
            model_file=model_file
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--ind_test', default=None, type=int, help='')
    

    args = parser.parse_args()
    run_it(args.ind_test)