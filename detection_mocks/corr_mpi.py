from matplotlib import pyplot as plt
import numpy as np
import nbodykit

from nbodykit.lab import *
from nbodykit import CurrentMPIComm
from nbodykit import style, setup_logging
plt.style.use('Notebook.mystyle')
comm = CurrentMPIComm.get()
rank = comm.Get_rank()
size = comm.Get_size()

def generate_lognormal_mock(nbar=5e-4, BoxSize=1000, Nmesh=256, bias=2, seed=4):
    """Generate mock lognormal catalog"""
    redshift = 0.55
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 2.0

    cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=Nmesh, bias=bias, seed=seed)

    return cat

def plot_corr(fig, ax, cat):
    r_edges = np.arange(50,250,2)
    r_bins = 0.5*(r_edges[1:] + r_edges[:-1]) 
    corr = SimulationBox2PCF(data1=cat, mode='1d', edges=r_edges, nthread=12)
    corr.run()
    print(corr.corr)
    #ax.plot(corr.corr['r'], corr.corr['corr'])
    print(f"rank={rank}   |  corr = {corr.corr['r']}  | corr = {corr.corr['corr']}")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
cat = generate_lognormal_mock(BoxSize=1000, seed=4)
plot_corr(fig, ax, cat)