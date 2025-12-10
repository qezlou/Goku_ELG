Quick Start Guide
=================

This guide will help you get started with Goku-ELG quickly.

Basic Usage
-----------

Computing Galaxy Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a simple example to compute the galaxy-galaxy correlation function:

.. code-block:: python

   import numpy as np
   from gal_goku import gal
   
   # Initialize the galaxy clustering module
   galaxy_model = gal.GalBase(logging_level='INFO')
   
   # Define cosmological parameters
   # [Omega_m, Omega_b, h, A_s, n_s, w0, wa, N_ur, alpha_s, m_nu]
   cosmo_params = [0.3, 0.05, 0.7, 2.1e-9, 0.96, -1.0, 0.0, 3.046, 0.0, 0.06]
   
   # Get the initial linear power spectrum
   k, P_lin = galaxy_model.get_init_linear_power(
       cosmo_params, 
       redshifts=[2.0]
   )
   
   print(f"Computed power spectrum for {len(k)} k-modes")

Using the Halo Mass Function Emulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gal_goku import emus
   
   # Initialize the HMF emulator
   hmf_emu = emus.Hmf(
       data_dir='path/to/data',
       y_log=True,
       fid='L2',
       logging_level='INFO'
   )
   
   # Train the emulator (if not already trained)
   # This step may take a few minutes
   predictions, truth, mass_bins = hmf_emu.train_pred_all_sims()
   
   print(f"HMF predictions shape: {predictions.shape}")
   print(f"Mass bins: {mass_bins}")

Making Predictions
~~~~~~~~~~~~~~~~~~

Once the emulator is trained, you can make predictions for new cosmological parameters:

.. code-block:: python

   import numpy as np
   
   # Define test cosmological parameters
   # Shape: (n_samples, n_parameters)
   X_test = np.array([
       [0.31, 0.049, 0.68, 2.0e-9, 0.965, -1.0, 0.0, 3.046, 0.0, 0.06],
       [0.28, 0.047, 0.72, 2.2e-9, 0.955, -0.9, 0.0, 3.046, 0.0, 0.08],
   ])
   
   # Get predictions and uncertainties
   predictions, variances = hmf_emu.predict(X_test)
   
   print(f"Predictions: {predictions}")
   print(f"Variances: {variances}")

Working with Halo-Halo Correlations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gal_goku_sims import xi
   
   # Load halo-halo correlation function data
   # This assumes you have simulation data available
   halo_xi = xi.HaloXi(data_dir='path/to/halo/data')
   
   # Process and analyze correlations
   # (Specific methods depend on your data structure)

Complete Workflow Example
--------------------------

Here's a complete workflow from cosmology to galaxy clustering:

.. code-block:: python

   import numpy as np
   from gal_goku import gal, emus
   import matplotlib.pyplot as plt
   
   # Step 1: Set up cosmological parameters
   cosmo_params = {
       'omega_m': 0.3,
       'omega_b': 0.05,
       'h': 0.7,
       'A_s': 2.1e-9,
       'n_s': 0.96,
       'w0': -1.0,
       'wa': 0.0,
       'N_ur': 3.046,
       'alpha_s': 0.0,
       'm_nu': 0.06
   }
   
   # Convert to array format
   cosmo_array = [cosmo_params[key] for key in [
       'omega_m', 'omega_b', 'h', 'A_s', 'n_s', 
       'w0', 'wa', 'N_ur', 'alpha_s', 'm_nu'
   ]]
   
   # Step 2: Initialize galaxy model
   gal_model = gal.GalBase(logging_level='INFO')
   
   # Step 3: Get linear power spectrum
   k, P_lin = gal_model.get_init_linear_power(
       cosmo_array,
       redshifts=[2.0]
   )
   
   # Step 4: Plot the results
   plt.figure(figsize=(10, 6))
   plt.loglog(k, P_lin)
   plt.xlabel('k [h/Mpc]')
   plt.ylabel('P(k) [(Mpc/h)^3]')
   plt.title('Linear Matter Power Spectrum')
   plt.grid(True, alpha=0.3)
   plt.savefig('power_spectrum.png')
   print("Power spectrum saved to power_spectrum.png")

Understanding Parameters
-------------------------

Cosmological Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

The emulator uses 10 cosmological parameters:

+---------------+------------------+--------------------------------------+
| Parameter     | Symbol           | Description                          |
+===============+==================+======================================+
| omega_m       | :math:`\Omega_m` | Total matter density                 |
+---------------+------------------+--------------------------------------+
| omega_b       | :math:`\Omega_b` | Baryon density                       |
+---------------+------------------+--------------------------------------+
| h             | :math:`h`        | Hubble parameter (H0 = 100h km/s/Mpc)|
+---------------+------------------+--------------------------------------+
| A_s           | :math:`A_s`      | Scalar amplitude (×10⁻⁹)             |
+---------------+------------------+--------------------------------------+
| n_s           | :math:`n_s`      | Scalar spectral index                |
+---------------+------------------+--------------------------------------+
| w0            | :math:`w_0`      | Dark energy EoS (present)            |
+---------------+------------------+--------------------------------------+
| wa            | :math:`w_a`      | Dark energy EoS (evolution)          |
+---------------+------------------+--------------------------------------+
| N_ur          | :math:`N_{ur}`   | Effective number of relativistic     |
|               |                  | species                              |
+---------------+------------------+--------------------------------------+
| alpha_s       | :math:`\alpha_s` | Running of spectral index            |
+---------------+------------------+--------------------------------------+
| m_nu          | :math:`m_\nu`    | Sum of neutrino masses [eV]          |
+---------------+------------------+--------------------------------------+

Typical Ranges
~~~~~~~~~~~~~~

Here are typical parameter ranges for ΛCDM-like cosmologies:

.. code-block:: python

   parameter_ranges = {
       'omega_m': (0.24, 0.40),
       'omega_b': (0.04, 0.06),
       'h': (0.60, 0.80),
       'A_s': (1.8e-9, 2.4e-9),
       'n_s': (0.92, 1.00),
       'w0': (-1.2, -0.8),
       'wa': (-0.5, 0.5),
       'N_ur': (2.5, 3.5),
       'alpha_s': (-0.02, 0.02),
       'm_nu': (0.0, 0.15)
   }

Best Practices
--------------

1. **Start Simple**: Begin with the default/fiducial cosmology before exploring parameter space
2. **Validate**: Always check emulator predictions against known results
3. **Stay in Range**: Keep parameters within the trained range for reliable predictions
4. **Check Uncertainties**: Use the variance estimates to assess prediction confidence
5. **Logging**: Enable logging to track computation progress and debug issues

Common Patterns
---------------

Leave-One-Out Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gal_goku import emus
   
   hmf_emu = emus.Hmf(data_dir='path/to/data')
   
   # Perform leave-one-out cross-validation
   hmf_emu.loo_train_pred(savefile='loo_results.h5')

Batch Predictions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate many test cosmologies
   n_samples = 100
   X_test = np.random.uniform(low=[0.24, 0.04, 0.60, 1.8e-9, 0.92, 
                                    -1.2, -0.5, 2.5, -0.02, 0.0],
                               high=[0.40, 0.06, 0.80, 2.4e-9, 1.00,
                                     -0.8, 0.5, 3.5, 0.02, 0.15],
                               size=(n_samples, 10))
   
   # Make predictions for all samples
   predictions, variances = hmf_emu.predict(X_test)

Next Steps
----------

- Explore :doc:`tutorials/index` for detailed examples
- Check the :doc:`api/index` for complete API documentation
- See example notebooks in the ``emu/notebooks/`` directory
