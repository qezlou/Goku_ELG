
import numpy as np
from matplotlib import pyplot as plt
from gal_goku_sims import hmf as halo_mass_func
base_dir = '/scratch/06536/qezlou/Goku/FOF/L2'

hmf = halo_mass_func.Hmf()


pigs = hmf.get_pig_dirs(base_dir, z=0.0, narrow=False)

h, b = hmf.get_fof_hmf(pigs['pig_dirs'][0], vol=250.0**3, param=pigs['params'][0], bins=np.arange(11.1, 14.5, 0.1), merge=False)

fig, ax = plt.subplots(1,1, figsize=(6,5))

mbins = 0.5 * (b[1:] + b[:-1])
ax.plot(mbins, h)
ax.set_yscale('log')
fig.savefig('test_hmf.png')