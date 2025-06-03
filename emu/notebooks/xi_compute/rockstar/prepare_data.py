## Get ASCII file to raed with rockstar
import bigfile
from nbodykit.lab import BigFileCatalog
from os import path as op
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

def bigfile_to_ascii(part_dir, save_dir):
    """
    Convert a BigFile particle catalog to an ASCII file.
    Parameters
    ----------
    part_dir : str
        Path to the BigFile particle catalog.
    ascii_file : str
        Path to the output ASCII file.
    """
    # Rockstar expects positions in Mpc/h and velocities in km/s
    # Make sure the units are correct; apply conversion if needed
    cat = BigFileCatalog(part_dir, dataset='1')
    print(f"Rank {rank} | cat.csize = {cat.csize}, cat.size = {cat.size}", flush=True)
    pos = cat['Position'].compute()/1000  # Convert from kpc to Mpc
    vel = cat['Velocity'].compute()
    mass = cat.attrs['MassTable'][1]*1e10  # Convert from Msun/h to Msun
    ids = cat['ID'].compute()
    boxsize = cat.attrs['BoxSize'][0]/1000  # Convert from kpc to Mpc
    

    # Write rockstar.cfg
    config_file = op.join(save_dir, 'rockstar.cfg')
    ascii_file = op.join(save_dir, f'rockstar_ascii{rank}.txt')
    if rank == 0:
        with open(config_file, 'w') as f:
            f.write(f"""
            INBASE = {ascii_file}
            OUTBASE = ./output
            PARALLEL_IO = 1
            NUM_BLOCKS = 24
            FORCE_RES = 0.05
            PARTICLE_MASS = {mass}e10
            BOX_SIZE = {boxsize}
            NUM_SNAPS = 1
            SNAP_FORMAT = 3  # ascii
            FULL_PARTICLE_CHUNKS = 1
            OVERLAP_LENGTH = 0.1
            BINARY_OUTPUT = 1
                    """)
    
    # Write to ASCII file
    with open(ascii_file, 'w') as f:
        for i in range(pos.shape[0]):
            line = f"{pos[i,0]} {pos[i,1]} {pos[i,2]} {vel[i,0]} {vel[i,1]} {vel[i,2]} {mass} {ids[i]}\n"
            f.write(line)


part_dir = '/scratch/06536/qezlou/Goku/FOF/L2/compressed_10p_Box250_Part750_0098/output/PART_003'
save_dir ='/scratch/06536/qezlou/Goku/FOF/rock_halos'
bigfile_to_ascii(part_dir, save_dir)