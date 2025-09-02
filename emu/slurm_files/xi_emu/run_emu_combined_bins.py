import argparse
import numpy as np
import importlib
from gal_goku import emus_multifid
importlib.reload(emus_multifid)


def run_it(ind_test, use_rho, num_inducing=500, num_latents=40, remove_sims=None):
    #data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/xi_on_grid/'
    #train_subdir = 'train_hetero'
    data_dir = '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
    #train_subdir = 'train_combined_less_massive'
<<<<<<< HEAD
    train_subdir = 'train_remove_bad_l2_sims'
=======
    train_subdir = 'train_remove_bad_l2_sims_test_kl_reg'
    
>>>>>>> a4dfa92 (remove-bad-sims & KL_multiplier)
    emu = emus_multifid.XiNativeBinsFullDimReduc(data_dir=data_dir,
                                                num_inducing=num_inducing, 
                                                num_latents=num_latents,
                                                use_rho=bool(use_rho),
                                                #remove_sims=remove_sims,
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
            opt_params={'max_iters':38_000, 'initial_lr':5e-3, 'kl_multiplier': 0.1}, 
            model_file=model_file
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get the correlation function of the galaxies in the PIGs')
    parser.add_argument('--ind_test', default=None, type=int, help='')
    parser.add_argument('--use_rho', default=1, type=int, help='')
    parser.add_argument('--remove_sims', default=None, type=int, nargs='+', help='')

    args = parser.parse_args()
    run_it(ind_test=args.ind_test, use_rho=args.use_rho, remove_sims=args.remove_sims)
