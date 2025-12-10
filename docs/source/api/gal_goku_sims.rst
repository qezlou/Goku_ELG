gal_goku_sims Package
=====================

This package handles simulation data processing and computation of summary statistics.

Main Modules
------------

hmf Module
~~~~~~~~~~

.. automodule:: gal_goku_sims.hmf
   :members:
   :undoc-members:
   :show-inheritance:

Halo Mass Function computations from simulation catalogs.

**Main Classes:**

HMF
^^^

Computes the halo mass function from simulation halo catalogs.

**Key functionality:**

- Read halo catalogs from simulations
- Compute mass functions with proper binning
- Handle multiple cosmologies and redshifts
- Export results in standardized formats

**Typical Usage:**

.. code-block:: python

   from gal_goku_sims import hmf
   
   # Initialize HMF computer
   hmf_calc = hmf.HMF(
       catalog_path='path/to/halos.hdf5',
       box_size=1000.0,  # Mpc/h
       redshift=2.0
   )
   
   # Compute mass function
   masses, phi = hmf_calc.compute()

xi Module
~~~~~~~~~

.. automodule:: gal_goku_sims.xi
   :members:
   :undoc-members:
   :show-inheritance:

Correlation function computations from simulation data.

**Main Classes:**

HaloXi
^^^^^^

Computes halo-halo correlation functions from simulation catalogs.

**Key functionality:**

- Compute 2-point correlation functions
- Support for mass-threshold samples
- Jackknife error estimation
- Parallel computation with MPI

**Typical Usage:**

.. code-block:: python

   from gal_goku_sims import xi
   
   # Initialize correlation function computer
   xi_calc = xi.HaloXi(
       catalog_path='path/to/halos.hdf5',
       box_size=1000.0,  # Mpc/h
       mass_threshold=1e12  # Msun/h
   )
   
   # Compute correlation function
   r, xi_r = xi_calc.compute()

mpi_helper Module
~~~~~~~~~~~~~~~~~

.. automodule:: gal_goku_sims.mpi_helper
   :members:
   :undoc-members:
   :show-inheritance:

MPI utilities for parallel processing of simulation data.

**Key Functions:**

- Process distribution across MPI ranks
- Collective operations for data gathering
- Efficient parallel I/O
- Error handling in MPI context

**Typical Usage:**

.. code-block:: python

   from gal_goku_sims import mpi_helper
   from mpi4py import MPI
   
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   
   # Distribute work across ranks
   my_tasks = mpi_helper.distribute_tasks(
       total_tasks=100,
       rank=rank,
       size=size
   )

Data Formats
------------

Halo Catalogs
~~~~~~~~~~~~~

Halo catalogs are expected in HDF5 format with the following structure:

.. code-block:: text

   halos.hdf5
   ├── mass          # Halo masses [Msun/h]
   ├── pos           # Positions [Mpc/h], shape (N, 3)
   ├── vel           # Velocities [km/s], shape (N, 3)
   └── metadata
       ├── box_size  # Box size [Mpc/h]
       ├── redshift  # Redshift
       └── cosmology # Cosmological parameters

Correlation Functions
~~~~~~~~~~~~~~~~~~~~~

Correlation function outputs are saved in HDF5 format:

.. code-block:: text

   xi.hdf5
   ├── r             # Separation bins [Mpc/h]
   ├── xi            # Correlation function values
   ├── xi_err        # Errors (if computed)
   └── metadata
       ├── mass_threshold  # Mass threshold [Msun/h]
       ├── redshift        # Redshift
       └── n_pairs         # Number of pairs per bin

Performance Considerations
--------------------------

MPI Parallelization
~~~~~~~~~~~~~~~~~~~

For large datasets, use MPI parallelization:

.. code-block:: bash

   mpirun -np 16 python compute_correlations.py

Memory Management
~~~~~~~~~~~~~~~~~

When working with large catalogs:

1. Use chunked reading with HDF5
2. Process data in batches
3. Clear memory explicitly with ``del`` statements
4. Monitor memory usage with ``memory_profiler``

Optimization Tips
~~~~~~~~~~~~~~~~~

- Use pre-computed pair counts when possible
- Cache frequently accessed data
- Vectorize operations with NumPy
- Profile code to identify bottlenecks
