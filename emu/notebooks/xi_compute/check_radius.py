import numpy as np
import h5py
import importlib
from gal_goku_sims import xi
importlib.reload(xi)
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
srank = comm.Get_rank()
ssize = comm.Get_size()

base_dir = '/scratch/06536/qezlou/Goku/FOF/L2/'
z = 2.5
narrow = False

corr= xi.Corr()
pigs = corr.get_pig_dirs(base_dir, z=z, narrow=narrow)
#corr.load_halo_cat(pigs['pig_dirs'][0], cosmo=corr.get_cosmo(pigs['params'][0]))

from nbodykit.lab import BigFileCatalog, HaloCatalog

cat = BigFileCatalog(pigs['pig_dirs'][10], dataset='FOFGroups')

redshift = 1/cat.attrs['Time'] - 1
cat['Mass'] *= 1e10
cosmo = corr.get_cosmo(pigs['params'][10])
halos = HaloCatalog(cat, 
                    cosmo=cosmo, 
                    redshift=redshift,
                    mdef='200m',
                    mass='Mass',
                    position='MassCenterPosition',
                    velocity='MassCenterVelocity')
halos.attrs['BoxSize'] /= 1000
halos['Position'] /= 1000

mass = halos['Mass'].compute()
pos = halos['Position'].compute()
radius = halos['Radius'].compute() * (1 + redshift)  # in Mpc/h
print(f'masss = {np.min(mass)/1e10} - {np.max(mass)/1e10} 1e10Msun/h')
print(f'pos = {np.min(pos)} - {np.max(pos)}')
print(f'radius = {np.min(radius)} - {np.max(radius)}')


ex_rad_fac = 2
# Gather data to rank 0 for exclusion
all_pos = self.nbkit_comm.gather(halos['Position'].compute(), root=0)
all_mass = self.nbkit_comm.gather(halos['Mass'].compute(), root=0)
# The returned units in nbodykit.transform.HaloRadius is proper Mpc/h
all_r200 = self.nbkit_comm.gather(ex_rad_fac*halos['Radius'].compute()*(1+redshift), root=0)
self.logger.info(f'min r200: {np.min(halos["Radius"].compute()*(1+redshift))*ex_rad_fac} Mpc/h | max r200: {np.max(halos["Radius"].compute()*(1+redshift))*ex_rad_fac} Mpc/h')

if self.nbkit_rank == 0:
    pos_concat = np.concatenate(all_pos)
    mass_concat = np.concatenate(all_mass)
    r200_concat = np.concatenate(all_r200)
    keep_mask_all = self.halo_exclusion(pos_concat, mass_concat, r200_concat, boxsize=halos.attrs['BoxSize'])
else:
    keep_mask_all = None

# Broadcast the final keep mask
keep_mask_all = self.nbkit_comm.bcast(keep_mask_all, root=0)

# Apply the mask on this rank's halos
## Fist, find the cumulative halo count to this rank
n_halos_local = len(halos)
counts = self.nbkit_comm.allgather(n_halos_local)
offset = sum(counts[:self.nbkit_rank])
keep_mask = keep_mask_all[offset:offset + len(halos)]