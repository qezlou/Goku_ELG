Quick Reference
===============

This page provides quick reference information for common tasks.

Command Line Reference
----------------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build HTML documentation
   cd docs
   make html
   
   # Build PDF documentation
   make latexpdf
   
   # Clean build files
   make clean
   
   # Rebuild everything
   make clean html

Using the Emulator
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Activate environment
   conda activate gal_goku
   
   # Run Python scripts
   python my_script.py
   
   # Run with MPI
   mpirun -np 16 python mpi_script.py

API Quick Reference
-------------------

Initialize Emulators
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gal_goku import emus
   
   # HMF emulator
   hmf_emu = emus.Hmf(
       data_dir='path/to/data',
       y_log=True,
       fid='L2'
   )
   
   # Make predictions
   predictions, variances = hmf_emu.predict(X_test)

Compute Linear Power
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gal_goku import gal
   
   # Initialize
   gal_model = gal.GalBase()
   
   # Compute P(k)
   cosmo_params = [0.3, 0.05, 0.7, 2.1e-9, 0.96,
                   -1.0, 0.0, 3.046, 0.0, 0.06]
   k, P_lin = gal_model.get_init_linear_power(
       cosmo_params,
       redshifts=[2.0]
   )

Parameter Reference
-------------------

Cosmological Parameters
~~~~~~~~~~~~~~~~~~~~~~~

Order: [Omega_m, Omega_b, h, A_s, n_s, w0, wa, N_ur, alpha_s, m_nu]

+----------+-------------+------------------------+
| Index    | Parameter   | Typical Range          |
+==========+=============+========================+
| 0        | Omega_m     | 0.24 - 0.40            |
+----------+-------------+------------------------+
| 1        | Omega_b     | 0.04 - 0.06            |
+----------+-------------+------------------------+
| 2        | h           | 0.60 - 0.80            |
+----------+-------------+------------------------+
| 3        | A_s         | 1.8e-9 - 2.4e-9        |
+----------+-------------+------------------------+
| 4        | n_s         | 0.92 - 1.00            |
+----------+-------------+------------------------+
| 5        | w0          | -1.2 - -0.8            |
+----------+-------------+------------------------+
| 6        | wa          | -0.5 - 0.5             |
+----------+-------------+------------------------+
| 7        | N_ur        | 2.5 - 3.5              |
+----------+-------------+------------------------+
| 8        | alpha_s     | -0.02 - 0.02           |
+----------+-------------+------------------------+
| 9        | m_nu [eV]   | 0.0 - 0.15             |
+----------+-------------+------------------------+

Fiducial Cosmology (Planck-like)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fiducial = [0.3111, 0.0490, 0.6766, 2.1e-9, 0.9665,
               -1.0, 0.0, 3.046, 0.0, 0.06]

Common Patterns
---------------

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Leave-one-out
   hmf_emu.loo_train_pred(savefile='loo_results.h5')
   
   # Leave-bunch-out
   X_test, Y_test, Y_pred, var = hmf_emu.leave_bunch_out(n_out=5)

Batch Predictions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Generate test cosmologies
   n_samples = 100
   X_test = np.random.uniform(
       low=[0.24, 0.04, 0.60, 1.8e-9, 0.92, -1.2, -0.5, 2.5, -0.02, 0.0],
       high=[0.40, 0.06, 0.80, 2.4e-9, 1.00, -0.8, 0.5, 3.5, 0.02, 0.15],
       size=(n_samples, 10)
   )
   
   # Predict
   predictions, variances = hmf_emu.predict(X_test)

Error Handling
--------------

Common Errors and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'gal_goku'**

Solution:

.. code-block:: bash

   cd src/gal_goku
   pip install -e .

**FileNotFoundError: data directory not found**

Solution: Update the ``data_dir`` path to point to your data.

**MemoryError during prediction**

Solution: Process in smaller batches:

.. code-block:: python

   batch_size = 10
   for i in range(0, len(X_test), batch_size):
       batch = X_test[i:i+batch_size]
       pred, var = hmf_emu.predict(batch)
       # Process results

Plotting Templates
------------------

Basic Line Plot
~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 6))
   plt.plot(x, y, 'b-', linewidth=2, label='Data')
   plt.xlabel('X axis', fontsize=14)
   plt.ylabel('Y axis', fontsize=14)
   plt.title('Title', fontsize=16)
   plt.legend(fontsize=12)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('output.png', dpi=150)

Log-Log Plot
~~~~~~~~~~~~

.. code-block:: python

   plt.figure(figsize=(10, 6))
   plt.loglog(x, y, 'r-', linewidth=2)
   plt.xlabel(r'$k$ [$h$/Mpc]', fontsize=14)
   plt.ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.savefig('power_spectrum.png', dpi=150)

Plot with Error Bands
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   plt.figure(figsize=(10, 6))
   plt.plot(x, y, 'b-', linewidth=2, label='Mean')
   plt.fill_between(x, y - err, y + err,
                     alpha=0.3, color='blue',
                     label='±1σ')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('with_errors.png', dpi=150)

File Formats
------------

HDF5 Structure
~~~~~~~~~~~~~~

Reading HDF5 files:

.. code-block:: python

   import h5py
   
   with h5py.File('data.h5', 'r') as f:
       # List all keys
       print(list(f.keys()))
       
       # Read datasets
       data = f['dataset_name'][:]
       
       # Read attributes
       metadata = f.attrs['metadata']

Writing HDF5 files:

.. code-block:: python

   with h5py.File('output.h5', 'w') as f:
       # Create dataset
       f.create_dataset('data', data=my_array)
       
       # Add attributes
       f.attrs['description'] = 'My data'

Useful Links
------------

- **GitHub Repository**: https://github.com/qezlou/private-gal-emu
- **GPflow Documentation**: https://gpflow.github.io/
- **NumPy Documentation**: https://numpy.org/doc/
- **SciPy Documentation**: https://docs.scipy.org/
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/

Environment Variables
---------------------

Useful environment variables:

.. code-block:: bash

   # OpenMP threads (for parallel computations)
   export OMP_NUM_THREADS=16
   
   # MKL threads (if using Intel MKL)
   export MKL_NUM_THREADS=16
   
   # Disable TensorFlow warnings (if using GPflow)
   export TF_CPP_MIN_LOG_LEVEL=2

Performance Tips
----------------

1. **Use vectorized operations** instead of loops
2. **Pre-allocate arrays** when possible
3. **Use appropriate data types** (float32 vs float64)
4. **Profile your code** to find bottlenecks
5. **Cache expensive computations**

.. code-block:: python

   # Example: Vectorized vs loop
   
   # Slow
   result = []
   for x in X:
       result.append(expensive_function(x))
   
   # Fast
   result = np.vectorize(expensive_function)(X)
   # or even better, if possible:
   result = expensive_function(X)  # if it handles arrays

Keyboard Shortcuts
------------------

Jupyter Notebook
~~~~~~~~~~~~~~~~

- ``Shift + Enter``: Run cell
- ``Ctrl + Enter``: Run cell, stay in cell
- ``Alt + Enter``: Run cell, insert below
- ``A``: Insert cell above
- ``B``: Insert cell below
- ``DD``: Delete cell
- ``M``: Change to Markdown
- ``Y``: Change to Code

IPython
~~~~~~~

- ``?function``: Get help
- ``??function``: Get source code
- ``%timeit``: Time execution
- ``%prun``: Profile code
- ``%matplotlib inline``: Inline plots

Troubleshooting Checklist
--------------------------

Before asking for help, check:

1. ☐ Is the correct conda environment activated?
2. ☐ Are all packages installed?
3. ☐ Are file paths correct?
4. ☐ Are parameter values in valid ranges?
5. ☐ Is there enough memory available?
6. ☐ Have you checked the error message carefully?
7. ☐ Have you tried the examples in the tutorials?
8. ☐ Have you checked the GitHub issues?
