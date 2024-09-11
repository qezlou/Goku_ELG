"""
For each smoothing scale sigma, this script computes the Q quantity and it takes
~ 15 miniues with 4 nodes and 7 tasks per node. The script is memory efficient.It is run
on Frontera.
"""

import sys
## To load h5py properly on Frontera
path_to_exclude = '/opt/apps/intel19/impi19_0/python3/3.7.0/lib/python3.7/site-packages'
path_to_include = '/home1/06536/qezlou/miniconda3/envs/py38/lib/python3.8/site-packages/'

sys.path.insert(0, path_to_include)
sys.path.remove(path_to_exclude)
from datetime import datetime
import h5py
from scipy.ndimage import gaussian_filter as gf
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from mpi4py_helper import distribute_array

def get_density(sigma=1, N=1_000):
    """Get the smoothed density field.
    Parameters:
    -----------
    sigma: float
        The smoothing scale of the density field.
    N: int
        The number of grid points in each direction.
    Returns:
        The smoothed density field.
    """
    import h5py
    logger(f'computing density ')
    with h5py.File('/scratch1/06536/qezlou/Goku/full_density.hdf5','r') as f:
        dens = f['DM/dens'][:]
        dens = gf(dens, sigma=sigma, mode='wrap')
    logger(f'Density is computed ')
    return dens[:]

def get_phi_hat(sigma, L=1, N=1_000):
    """Get the potential in Fourier space.
    Parmeters:
    ----------
    sigma: float
        The smoothing scale of the density field.
    L: float
        The size of the domain.
    N: int
        The number of grid points in each direction.
    Returns:
    --------
    Phi_hat: array
        The potential in Fourier space.
    kx, ky, kz: array
        The wave numbers in each direction.
    """
    logger(f'computing Phi_hat ')
    # Generate grid
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    z = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    F = get_density(sigma=sigma, N=N)

    # Compute the FFT of the source term
    logger(f'computing FFT ' )
    F_hat = fft.fftn(F)
    del F
    logger(f'FFT is computed ')

    # Wave numbers in each direction
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')

    # Compute the FFT of the potential using the Poisson equation in Fourier space
    denominator = -(kx**2 + ky**2 + kz**2)
    denominator[0, 0, 0] = 1  # Avoid division by zero for the zero frequency component

    Phi_hat = F_hat / denominator
    Phi_hat[0, 0, 0] = 0  # Set the mean of the solution to zero to enforce uniqueness
    logger(f'Phi_hat is computed ')
    return Phi_hat, kx, ky, kz

def get_tidal_tensor(sigma=1, N=1_000):
    """Get the tidal tensor in real space.
    Parameters:
    -----------
    sigma: float
        The smoothing scale of the density field.
    L: float
        The size of the domain.
    N: int
        The number of grid points in each direction.
    Returns:
    --------
    Saves the tidal tensor components on hdf5 files. This approach
    is used to avoid memory overflow.
    """
    logger(f'Computing Tidal Tensor ')
    Phi_hat, kx, ky, kz = get_phi_hat(sigma=sigma, N=N)
    # Compute the second mixed partial derivatives in Fourier space
    Txx_hat = -(kx * kx) * Phi_hat
    Txy_hat = -(kx * ky) * Phi_hat
    Txz_hat = -(kx * kz) * Phi_hat
    Tyy_hat = -(ky * ky) * Phi_hat
    Tyz_hat = -(ky * kz) * Phi_hat
    Tzz_hat = -(kz * kz) * Phi_hat
    del Phi_hat, kx, ky, kz
    logger(f'saving Tidal Tensor')
    
    # Let's save the FT of the tidal tensor components on hdf5 files
    # to avoid memory overflow
    with h5py.File('Txx_hat.hdf5','w') as fw:
        fw['Txx_hat'] = Txx_hat
    with h5py.File('Txy_hat.hdf5','w') as fw:
        fw['Txy_hat'] = Txy_hat
    with h5py.File('Txz_hat.hdf5','w') as fw:
        fw['Txz_hat'] = Txz_hat
    with h5py.File('Tyy_hat.hdf5','w') as fw:
        fw['Tyy_hat'] = Tyy_hat
    with h5py.File('Tyz_hat.hdf5','w') as fw:
        fw['Tyz_hat'] = Tyz_hat
    with h5py.File('Tzz_hat.hdf5','w') as fw:
        fw['Tzz_hat'] = Tzz_hat
    del Txx_hat, Txy_hat, Txz_hat, Tyy_hat, Tyz_hat, Tzz_hat
    
    # Now we can compute the tidal tensor in real space
    with h5py.File('Txx_hat.hdf5','r') as fr:
        Txx_hat = fr['Txx_hat'][:]
        Txx = fft.ifftn(Txx_hat).real
        with h5py.File('Txx.hdf5','w') as fw:
            fw['Txx'] = Txx
        del Txx_hat, Txx
    
    with h5py.File('Txy_hat.hdf5','r') as fr:
        Txy_hat = fr['Txy_hat'][:]
        Txy = fft.ifftn(Txy_hat).real
        with h5py.File('Txy.hdf5','w') as fw:
            fw['Txy'] = Txy
        del Txy_hat, Txy
    
    with h5py.File('Txz_hat.hdf5','r') as fr:
        Txz_hat = fr['Txz_hat'][:]
        Txz = fft.ifftn(Txz_hat).real
        with h5py.File('Txz.hdf5','w') as fw:
            fw['Txz'] = Txz
        del Txz_hat, Txz
    
    with h5py.File('Tyy_hat.hdf5','r') as fr:
        Tyy_hat = fr['Tyy_hat'][:]
        Tyy = fft.ifftn(Tyy_hat).real
        with h5py.File('Tyy.hdf5','w') as fw:
            fw['Tyy'] = Tyy
        del Tyy_hat, Tyy

    with h5py.File('Tyz_hat.hdf5','r') as fr:
        Tyz_hat = fr['Tyz_hat'][:]
        Tyz = fft.ifftn(Tyz_hat).real
        with h5py.File('Tyz.hdf5','w') as fw:
            fw['Tyz'] = Tyz
        del Tyz_hat, Tyz

    with h5py.File('Tzz_hat.hdf5','r') as fr:
        Tzz_hat = fr['Tzz_hat'][:]
        Tzz = fft.ifftn(Tzz_hat).real
        with h5py.File('Tzz.hdf5','w') as fw:
            fw['Tzz'] = Tzz
        del Tzz_hat, Tzz

    logger(f'tidal tensor is saved')
    
def get_q(sigma=1, L=1, N=1_000):
    """Compute the scalar quantity q from the tidal tensor.
    Parameters:
    -----------
    sigma: float
        The smoothing scale of the density field.
    L: float
        The size of the domain.
    N: int
        The number of grid points in each direction.
    Returns:
    --------
    Saves the q array on an hdf5 file.
    """
    chunks_x = distribute_array(MPI, comm, np.arange(N))
    if rank == 0:
        # Compute the tidal tensor, fft is quick
        # so we can afford to do it on a single process
        # It takes ~ 15 minutes to compute the tidal tensor
        logger(f' Getting tidal tensors ')
        get_tidal_tensor(sigma=sigma, N=N)
        logger(f' Done computing tidal tensors ')
    comm.Barrier()
    logger(f' Cpmputing Q ')
    Txx = h5py.File('Txx.hdf5','r')['Txx']
    Txy = h5py.File('Txy.hdf5','r')['Txy']
    Txz = h5py.File('Txz.hdf5','r')['Txz']
    Tyy = h5py.File('Tyy.hdf5','r')['Tyy']
    Tyz = h5py.File('Tyz.hdf5','r')['Tyz']
    Tzz = h5py.File('Tzz.hdf5','r')['Tzz']
    # Initialize the scalar quantity q
    q = np.zeros(shape=(N,N,N))

    # Step 1: Create the 4D array for tensors
    T = np.zeros((len(chunks_x), N, N, 3, 3))
    # This part is MPI, I checked it improves the speed by
    # a factor of number of tasks
    T[..., 0, 0] = Txx[chunks_x, :, :]
    T[..., 0, 1] = Txy[chunks_x, :, :]
    T[..., 0, 2] = Txz[chunks_x, :, :]
    T[..., 1, 0] = Txy[chunks_x, :, :]
    T[..., 1, 1] = Tyy[chunks_x, :, :]
    T[..., 1, 2] = Tyz[chunks_x, :, :]
    T[..., 2, 0] = Txz[chunks_x, :, :]
    T[..., 2, 1] = Tyz[chunks_x, :, :]
    T[..., 2, 2] = Tzz[chunks_x, :, :]

    # Step 2: Compute the eigenvalues in a vectorized manner
    eigenvalues = np.linalg.eigvalsh(T)  # Shape will be (chunks_x, N, N, 3)
    del T
    # Step 3: Compute the scalar quantity q
    # Differences of eigenvalues
    lambda_diff = eigenvalues[..., np.newaxis, :] - eigenvalues[..., np.newaxis]
    del eigenvalues
    # Keep only the upper triangle of differences without the diagonal
    lambda_diff = lambda_diff[..., np.triu_indices(3, k=1)[0], np.triu_indices(3, k=1)[1]]
    # Compute q
    q = np.zeros(shape=(N,N,N))
    q[chunks_x,:,:] = 0.5 * np.sum(lambda_diff ** 2, axis=-1)  # Summing squares of differences
    del lambda_diff

    # q now has the shape (N_x, N, N) 
    comm.Barrier()
    logger(f' Q is computed ')
    q = np.ascontiguousarray(q, np.float32)
    comm.Barrier()
    logger(f'run Allreduce')
    comm.Allreduce(MPI.IN_PLACE, q, op=MPI.SUM)
    comm.Barrier()
    logger(f'Saving')
    if rank ==0:
        with h5py.File(f'q_mpi_sigma{np.round(sigma,2)}.hdf5','w') as fw:
            fw['q']  = q
    else:
        del q
    comm.Barrier()

def logger(message):
    """convinient tool for logging the code progress"""
    print(f'{str(datetime.datetime.now())}'
          f'rank = {rank}'
          f'| {message}', flush=True)


# Get the Q quantity for different smoothing scales
for sigma in [1.5,2,2.5]:
    logger(f'Starting sigma = {sigma}')
    get_q(sigma=sigma)

