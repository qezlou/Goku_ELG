import numpy as np
import h5py
import get_corr
from mpi4py import MPI
import os.path as op

comm_size = MPI.COMM_WORLD.Get_size()
corr = get_corr.Corr(ranks_for_nbkit=comm_size)

sim_tag = 'cosmo_10p_Box250_Part750_0000'

basedir = '/scratch/06536/qezlou/Goku/FOF/L2'
pig_dir = f'{basedir}/{sim_tag}/output/PIG_003/'
save_dir = '/scratch/06536/qezlou/Goku/processed_data/false_positive/'
assert op.exists(save_dir)
save_file = op.join(save_dir,f'wp_fof_uniform_fp_15percent_{sim_tag}.hdf5')
assert not op.exists(save_file), f'file exists: {save_file}'

r_edges = np.logspace(-1.5, np.log10(2), 8)
r_edges = np.append(r_edges, np.logspace(np.log10(2), np.log10(30), 10)[1:])
r_edges = np.append(r_edges, np.logspace(np.log10(30), np.log10(80), 50)[1:])


corr_fof = corr.get_corr_fof(pig_dir, r_edges, mode='projected', false_positive_ratio=0.15)
corr_fof.run()

if corr.rank ==0:
    corr.logger.info(f'save_file = {save_file}')
    with h5py.File(save_file, 'w') as f:
        f['r'] = np.array([r_edges[i]+r_edges[i+1] for i in range(r_edges.size-1)])
        f['corr'] = corr_fof.corr['corr']

