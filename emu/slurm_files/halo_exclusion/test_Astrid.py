"""
Test using the SubhaloRankGr for filtering centrals and compute the correlation function
"""
import bigfile
import numpy as np
from nbodykit.lab import SimulationBox2PCF, ArrayCatalog
import h5py


save_dir = '/work2/06536/qezlou/astrid_sfrh/FOF/SubGroups'
bf = bigfile.File(save_dir)
# Get the available blocks
print("Available blocks:", bf.blocks)

# Open one of the blocks to examine its structure

with bf['SubhaloRankInGr'] as f:
    print("Attributes:", f.attrs)
    print("Size:", f.size)
    if f.size > 0:
        print("Fraction of centrals", 1 - np.where(f[:])[0].size / f.size)
        mask_cen =  f[:] == 0
with bf['SubhaloMass'] as f:
    print(f.attrs, f.size, f[:].shape)
    mass = np.log10(f[:]) + 10
    mask_mass = (mass >= 11)

with bf['SubhaloPos'] as f:
    print(f.attrs, f.size, f[:].shape)
    pos = f[:]/1000.


r_edges = np.logspace(np.log10(0.01), np.log10(0.1), 4)
r_edges = np.append(r_edges, np.logspace(np.log10(0.1), np.log10(2), 15)[1:])
r_edges = np.append(r_edges, np.logspace(np.log10(2), np.log10(60), 15)[1:])
r_edges = np.append(r_edges, np.linspace(60, 80, 20)[1:])

def get_corr(cat, savefile):
    corr = SimulationBox2PCF(data1=cat, mode='1d', edges=r_edges,  position='Position', BoxSize=250)
    corr.run()
    result = corr.corr['corr'][:]
    mbins =  np.array([(r_edges[i]+r_edges[i+1])/2 for i in range(r_edges.size-1)])

    with h5py.File(savefile, 'w') as f:
        f.create_dataset('mbins', data=mbins)
        f.create_dataset('result', data=result)



cat_orig = ArrayCatalog({'Position': pos[mask_mass]})
get_corr(cat_orig, 'corr_all.h5')
cat_cen = ArrayCatalog({'Position': pos[mask_cen & mask_mass]})
get_corr(cat_cen, 'corr_cen.h5')
