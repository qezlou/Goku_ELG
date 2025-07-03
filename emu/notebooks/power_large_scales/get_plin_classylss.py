"""
Running the classylss to compute the linear power at z=99 for any arbitrary cosmology
"""
import h5py
from gal_goku import gal


# Get cosmo params
import importlib
from gal_goku import summary_stats
importlib.reload(summary_stats)

data_dir='/scratch/06536/qezlou/Goku/processed_data'
prop = summary_stats.Propagator(data_dir=data_dir, z=2.5, fid='HF')
cosmo_pars = prop.params[15]
sim_tag = prop.sim_tags[15]

gal_base =gal.GalBase()
box= 1_000
npart=3_000
k, pk = gal_base.get_init_linear_power(box=box, npart=npart, cosmo_pars=cosmo_pars, k=None)


with h5py.File(f'{data_dir}/power_class_{sim_tag}.h5','w') as fw:
    fw['k'] = k
    fw['pk'] = pk
