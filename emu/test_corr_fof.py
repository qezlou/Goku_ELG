import numpy as np
import h5py
import get_corr
corr = get_corr.Corr()

sim_tag = 'cosmo_10p_Box1000_Part3000_0074'

basedir = '/scratch/06536/qezlou/Goku/FOF/HF'
pig_dir = f'{basedir}/{sim_tag}/output/PIG_003/'
save_file = f'/work2/06536/qezlou/Goku/corr_funcs_fof/fof_{sim_tag}.hdf5'

r_edges = np.logspace(0, np.log10(200), 100)
corr_fof = corr.get_corr_fof(pig_dir, r_edges)

if corr.rank ==0:
    corr.logger.info(f'save_file = {save_file}')
    with h5py.File(save_file, 'w') as f:
        f['r'] = ( r_edges[:-1] + r_edges[1:] ) / 2
        f['corr'] = corr_fof

