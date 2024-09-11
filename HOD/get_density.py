"""
A module to calculate the density field of the simulation box. It will be used for assembly bias HOD introduced in Hadzhiyska et al. 2023.
"""
# Take care of path import on TACC, helps to import mpi4py
import sys
sys.path.insert(0, '/home1/06536/qezlou/miniconda3/envs/goku/lib/python3.8/site-packages')
path_to_exclude = '/opt/apps/intel19/impi19_0/python3/3.9.7/lib/python3.9/site-packages/'
#sys.path.remove()

if path_to_exclude in sys.path:
    sys.path.remove(path_to_exclude)
########


import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import argparse
import h5py
import numpy as np
#from mpi4py import MPI
from astropy.cosmology import Planck15 as cosmo
from nbodykit.lab import *
from nbodykit import CurrentMPIComm
#CurrentMPIComm.set(MPI.COMM_WORLD)
class Density:
    """ A parallel code to get the density field in Gadget simulations"""
    def __init__(self, snaps, Nmesh, savedir, savefile=None, boxsize=None, z=None,
                 parttype='PartType1', ls_vec=[0,0,1], zspace=True, momentumz=False):
        """
        snaps : The path to the simulation snapshots in the format like : ./output/snapdie_029/snap_029*
        savedir : The path to the directory to store the results in. 
                  It saves the intermdeiate results and the final map.
        savefile: Optional, the file to save the full map on.
        Nmesh : The grid size
        zspace : Whether the density be on redshift space or not
        momentumz : If True, calculate the weighted velocity along z direction. The 
                    recorded field is (1+ delta)*Vpec_z, we have also saved (1+delta) as density
                    feild.
        ls_vec : list, len of 3
                 Unit line-of-sight vector
        """
        # Files' info
        self.snaps = snaps
        self.savedir = savedir
        self.savefile = savefile
        # The method parameters
        self.Nmesh = Nmesh
        self.boxsize = boxsize
        self.z = z
        self.zspace = zspace
        self.momentumz = momentumz
        self.parttype = parttype
        self.ls_vec = ls_vec
        if self.parttype =='PartType0':
            self.typestr='Gas'
        if self.parttype =='PartType1':
            self.typestr='DM'
        # MPI
        self.comm = CurrentMPIComm.get()
    
    def _apply_RSD(self, coord, vel):
        """Apply redshift space distortion along the slightline"""
        # Old Dask used in nbodykit does not accept elemnt-wise assignment, 
        # so we need to project V_pec along ls_vec
        print('Debug : BoxSize', self.boxsize, ' ls_vec: ', self.ls_vec, ' coord type :', type(coord), ' vel type :', type(vel), flush=True)
        coord = (coord + vel*self.ls_vec*1000*cosmo.h/cosmo.H(self.z).value)%self.boxsize
        return coord
    
    def get_mpgadget_cat(self):
        """
        Retrun a particle catalog for MP-Gadget type simulations, i.e. BigFile format
        """
        from nbodykit.lab import BigFileCatalog
        ptype_dict={'PartType0':'0', 'PartType1':'1'}
        cat = BigFileCatalog(self.snaps, dataset=ptype_dict[self.parttype], header='Header')
        self.z = 1/cat._attrs['Time'][0] - 1
        self.boxsize = cat._attrs['BoxSize'][0]
        return cat
    
    def Gadget(self):
        """
        Generate the density feild on a grid for Gadget simulations, e.g. Iluustris.
        """ 
   
        position_str = 'Position'
        vel_str = 'Velocity'
        cat = self.get_mpgadget_cat()
        if self.zspace:
            cat[position_str] = self._apply_RSD(coord=cat[position_str], vel=cat[vel_str])

        print('Rank ', self.comm.rank, ' cat,size= ', cat.size, flush=True)

        mesh = cat.to_mesh(Nmesh=self.Nmesh, position=position_str, compensated=True)
        dens = mesh.compute()
        if self.momentumz :
            # Average line-of-sight velocity in each voxel, the Gadget/Arepo units are in
            # sqrt(a)*km/s units
            cat['Vz'] = cat['Velocities'][:,2]/np.sqrt(1+self.z)
            mesh_momen = cat.to_mesh(Nmesh=self.Nmesh, position='Coordinates', value='Vz', compensated=True)
            pz = mesh_momen.compute()
        L = np.arange(0, self.Nmesh, 1)
        # Write each ranks' results on a file
        with h5py.File(self.savedir+str(self.comm.rank)+"_densfield.hdf5",'w') as f :
            f[self.typestr+'/dens'] = dens[:]
            if self.momentumz :
                f[self.typestr+'/pz'] = pz[:]
            f[self.typestr+'/x'] = L[dens.slices[0]]
            f[self.typestr+'/y'] = L[dens.slices[1]]
            f[self.typestr+'/z'] = L[dens.slices[2]]
            f[self.typestr+'/num_parts'] = cat.size
        if self.savefile is not None:
            self.comm.Barrier()
            if self.comm.rank==0:
                print('Saving the results', flush=True)
                self.make_full_mesh()
            self.comm.Barrier()
    
    def make_full_mesh(self):
        """ Loop over the saved hdf5 files for each rank to constrcut the full mesh and save it 
        """
        dens = np.empty((self.Nmesh, self.Nmesh, self.Nmesh))
        pz = np.empty((self.Nmesh, self.Nmesh, self.Nmesh))
        num_parts=0
        for i in range(self.comm.Get_size()) :
            print('file '+str(i)+' started!')
            with h5py.File(self.savedir+str(i)+'_densfield.hdf5','r') as f:
                x = slice(f[self.typestr+'/x'][0], f[self.typestr+'/x'][-1]+1)
                y = slice(f[self.typestr+'/y'][0], f[self.typestr+'/y'][-1]+1) 
                z = slice(f[self.typestr+'/z'][0], f[self.typestr+'/z'][-1]+1)
                dens[x,y,z] = f[self.typestr+'/dens'][:]
                if self.momentumz :
                    pz[x,y,z] = f[self.typestr+'/pz'][:]
                num_parts += f[self.typestr+'/num_parts'][()]
        with h5py.File(self.savefile, 'w') as f_w:
            f_w[self.typestr+'/dens'] = dens
            f_w[self.typestr+'/momentumz'] = pz
            f_w[self.typestr+'/num_parts']=num_parts



def runit(snaps, savedir, savefile, Nmesh, boxsize, z):
    
    dens = Density(snaps=snaps, savedir=savedir, 
                   savefile=savefile, Nmesh=Nmesh, 
                   zspace=False, momentumz=False,
                   boxsize= boxsize, z=z)
    dens.Gadget()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--snaps', type=str, required=True, help='adress to snapshots in globe pattern, like "./snap_0*"')
    parser.add_argument('--savedir', type=str, required=True, help='the dir to save the results of each rank in')
    parser.add_argument('--savefile', type=str, required=False, default=None, help='The file name to save the full density map')
    parser.add_argument('--Nmesh', type=int, required=True, help='Number of mesh cells along each axis')
    parser.add_argument('--boxsize', type=float, required=False, default=None, help='boxsize in cMpc/h, only if simtype="Gadget_old"')
    parser.add_argument('--z', type=float, required=False, help='For MDPL2 converted format we need to pass the redshift')

    
    args = parser.parse_args()
    runit(snaps=args.snaps, savedir=args.savedir, 
          savefile=args.savefile, Nmesh=args.Nmesh, 
          boxsize=args.boxsize, 
          z=args.z)
      

