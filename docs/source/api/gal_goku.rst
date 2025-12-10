gal_goku Package
================

This is the main package containing the emulator framework and galaxy clustering models.

Main Modules
------------

gal Module
~~~~~~~~~~

.. automodule:: gal_goku.gal
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

GalBase Class
^^^^^^^^^^^^^

.. autoclass:: gal_goku.gal.GalBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   The base class for galaxy clustering computations.

   **Key Methods:**

   .. automethod:: get_init_linear_power
   .. automethod:: configure_logging

LargeScaleGal Class
^^^^^^^^^^^^^^^^^^^

.. autoclass:: gal_goku.gal.LargeScaleGal
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Computes galaxy correlation functions at large scales (r > 40 Mpc/h).

emus Module
~~~~~~~~~~~

.. automodule:: gal_goku.emus
   :members:
   :undoc-members:
   :show-inheritance:

BaseStatEmu Class
^^^^^^^^^^^^^^^^^

.. autoclass:: gal_goku.emus.BaseStatEmu
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Base emulator class for summary statistics.

   **Key Methods:**

   .. automethod:: predict
   .. automethod:: loo_train_pred
   .. automethod:: train_pred_all_sims
   .. automethod:: leave_bunch_out

Hmf Class
^^^^^^^^^

.. autoclass:: gal_goku.emus.Hmf
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Halo Mass Function emulator.

   **Parameters:**
   
   - **data_dir** (str): Directory containing the training data
   - **y_log** (bool): Whether to use log-space for predictions
   - **fid** (str): Fiducial cosmology identifier
   - **multi_bin** (bool): Use multi-bin mode
   - **logging_level** (str): Logging verbosity level
   - **narrow** (bool): Use narrow parameter range
   - **no_merge** (bool): Don't merge data from multiple sources

emu_cosmo Module
~~~~~~~~~~~~~~~~

.. automodule:: gal_goku.emu_cosmo
   :members:
   :undoc-members:
   :show-inheritance:

Cosmological emulator utilities.

halo_tools Module
~~~~~~~~~~~~~~~~~

.. automodule:: gal_goku.halo_tools
   :members:
   :undoc-members:
   :show-inheritance:

Tools for halo manipulations, mass conversions, and halo model calculations.

init_power Module
~~~~~~~~~~~~~~~~~

.. automodule:: gal_goku.init_power
   :members:
   :undoc-members:
   :show-inheritance:

Initial conditions and linear power spectrum calculations using ClassyLSS.

summary_stats Module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: gal_goku.summary_stats
   :members:
   :undoc-members:
   :show-inheritance:

Summary statistics computations from simulation data.

single_fid Module
~~~~~~~~~~~~~~~~~

.. automodule:: gal_goku.single_fid
   :members:
   :undoc-members:
   :show-inheritance:

Single-fidelity Gaussian Process emulator implementations.

multi_fid Module
~~~~~~~~~~~~~~~~

.. automodule:: gal_goku.multi_fid
   :members:
   :undoc-members:
   :show-inheritance:

Multi-fidelity Gaussian Process emulator implementations.

utils Module
~~~~~~~~~~~~

.. automodule:: gal_goku.utils
   :members:
   :undoc-members:
   :show-inheritance:

Utility functions for data processing and manipulation.

plot Module
~~~~~~~~~~~

.. automodule:: gal_goku.plot
   :members:
   :undoc-members:
   :show-inheritance:

Plotting utilities for visualization of results.

plot_gal Module
~~~~~~~~~~~~~~~

.. automodule:: gal_goku.plot_gal
   :members:
   :undoc-members:
   :show-inheritance:

Specialized plotting functions for galaxy clustering visualizations.
