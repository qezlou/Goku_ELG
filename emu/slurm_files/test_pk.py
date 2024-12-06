import numpy as np
import h5py
import get_corr

corr = get_corr.Corr(ranks_for_nbkit=96, logging_level='DEBUG')

sim_fidelity='L2'
sim_num = '0316'

pig_dir = f'/scratch/06536/qezlou/Goku/FOF/{sim_fidelity}/compressed_10p_Box250_Part750_{sim_num}/output/PIG_003'

seeds = np.unique(np.random.randint(0, 1000_000, size=100))
all_pks, k = corr.get_power(pig_dir, seeds=seeds, mesh_res=1)

savefile = f'/work2/06536/qezlou/Goku/power/Zheng07_seeds_compressed_10p_Box250_Part750_{sim_num}_mesh_res_1.0.hdf5'

if corr.rank == 0:
    with h5py.File(savefile, 'w') as fw:
        fw['pk']= all_pks
        fw['k'] = k