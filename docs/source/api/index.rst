API Reference
=============

This section provides detailed API documentation for all modules in Goku-ELG.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   gal_goku
   gal_goku_sims

Package Overview
----------------

Goku-ELG consists of two main packages:

1. **gal_goku**: The main emulator package containing:
   
   - Galaxy clustering models
   - Emulator classes (HMF, correlation functions)
   - Multi-fidelity Gaussian Process implementations
   - Utility functions

2. **gal_goku_sims**: Simulation data processing package containing:
   
   - Halo mass function computations
   - Correlation function calculations
   - MPI utilities for parallel processing

Key Modules
-----------

gal_goku.gal
~~~~~~~~~~~~

Main module for galaxy clustering computations.

.. currentmodule:: gal_goku.gal

.. autosummary::
   :toctree: generated/
   
   GalBase
   LargeScaleGal

gal_goku.emus
~~~~~~~~~~~~~

Emulator classes for different summary statistics.

.. currentmodule:: gal_goku.emus

.. autosummary::
   :toctree: generated/
   
   BaseStatEmu
   Hmf

gal_goku.halo_tools
~~~~~~~~~~~~~~~~~~~

Utilities for halo manipulations and calculations.

gal_goku.init_power
~~~~~~~~~~~~~~~~~~~

Initial conditions and linear power spectrum calculations.

gal_goku_sims.hmf
~~~~~~~~~~~~~~~~~

Halo mass function computations from simulation data.

gal_goku_sims.xi
~~~~~~~~~~~~~~~~

Correlation function calculations from simulation data.
