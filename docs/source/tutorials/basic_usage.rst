Basic Usage Tutorial
====================

This tutorial covers the basics of using Goku-ELG for cosmological emulation.

Learning Objectives
-------------------

After completing this tutorial, you will be able to:

- Load and initialize the emulator
- Set up cosmological parameters
- Make predictions with the emulator
- Understand the output format
- Visualize results

Prerequisites
-------------

- Basic Python knowledge
- Familiarity with NumPy arrays
- Understanding of cosmological parameters (helpful but not required)

Installation Check
------------------

First, verify that Goku-ELG is properly installed:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Import main modules
   from gal_goku import gal, emus
   from gal_goku_sims import hmf
   
   print("All imports successful!")

Part 1: Understanding Cosmological Parameters
----------------------------------------------

Goku-ELG uses 10 cosmological parameters. Let's define a fiducial cosmology:

.. code-block:: python

   # Define fiducial cosmology (similar to Planck 2018)
   fiducial_cosmo = {
       'omega_m': 0.3111,      # Total matter density
       'omega_b': 0.0490,      # Baryon density
       'h': 0.6766,            # Hubble parameter
       'A_s': 2.1e-9,          # Scalar amplitude
       'n_s': 0.9665,          # Scalar spectral index
       'w0': -1.0,             # Dark energy EoS
       'wa': 0.0,              # Dark energy evolution
       'N_ur': 3.046,          # Effective neutrino number
       'alpha_s': 0.0,         # Running of spectral index
       'm_nu': 0.06            # Neutrino mass [eV]
   }
   
   # Convert to array format (required by emulator)
   param_names = ['omega_m', 'omega_b', 'h', 'A_s', 'n_s', 
                  'w0', 'wa', 'N_ur', 'alpha_s', 'm_nu']
   cosmo_array = np.array([fiducial_cosmo[p] for p in param_names])
   
   print("Cosmological parameters:")
   for name, value in zip(param_names, cosmo_array):
       print(f"  {name:10s} = {value:.6f}")

Part 2: Computing Linear Power Spectrum
----------------------------------------

The first step is computing the linear matter power spectrum:

.. code-block:: python

   # Initialize the galaxy model
   gal_model = gal.GalBase(logging_level='INFO')
   
   # Compute linear power spectrum at z=2
   redshift = 2.0
   k, P_lin = gal_model.get_init_linear_power(
       cosmo_array,
       redshifts=[redshift]
   )
   
   print(f"Computed P(k) for {len(k)} k-modes")
   print(f"k range: {k.min():.4f} to {k.max():.4f} h/Mpc")
   print(f"P(k) range: {P_lin.min():.2e} to {P_lin.max():.2e} (Mpc/h)^3")

Visualizing the Power Spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   plt.figure(figsize=(10, 6))
   plt.loglog(k, P_lin, 'b-', linewidth=2, label=f'z={redshift}')
   plt.xlabel(r'$k$ [$h$/Mpc]', fontsize=14)
   plt.ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]', fontsize=14)
   plt.title('Linear Matter Power Spectrum', fontsize=16)
   plt.grid(True, alpha=0.3)
   plt.legend(fontsize=12)
   plt.tight_layout()
   plt.savefig('linear_power_spectrum.png', dpi=150)
   print("Plot saved as 'linear_power_spectrum.png'")

Part 3: Working with the HMF Emulator
--------------------------------------

Now let's use the Halo Mass Function emulator:

.. code-block:: python

   # NOTE: You need to have the training data available
   # Replace 'path/to/data' with the actual path to your data directory
   
   data_dir = 'path/to/data'  # Update this!
   
   # Initialize the HMF emulator
   hmf_emu = emus.Hmf(
       data_dir=data_dir,
       y_log=True,          # Use log-space
       fid='L2',            # Fiducial simulation
       logging_level='INFO'
   )
   
   print("HMF emulator initialized")
   print(f"Number of training simulations: {len(hmf_emu.labels)}")

Making Predictions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare test cosmologies
   # Let's test 3 different cosmologies with varying omega_m
   X_test = np.array([
       [0.28, 0.049, 0.68, 2.1e-9, 0.97, -1.0, 0.0, 3.046, 0.0, 0.06],
       [0.31, 0.049, 0.68, 2.1e-9, 0.97, -1.0, 0.0, 3.046, 0.0, 0.06],
       [0.34, 0.049, 0.68, 2.1e-9, 0.97, -1.0, 0.0, 3.046, 0.0, 0.06],
   ])
   
   # Make predictions
   predictions, variances = hmf_emu.predict(X_test)
   
   # If using log-space, convert back
   hmf_pred = 10**predictions
   
   print(f"Predictions shape: {predictions.shape}")
   print(f"Mass bins shape: {hmf_emu.mbins.shape}")

Visualizing HMF Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   plt.figure(figsize=(10, 6))
   
   colors = ['blue', 'green', 'red']
   omega_m_values = [0.28, 0.31, 0.34]
   
   for i, (pred, om) in enumerate(zip(hmf_pred, omega_m_values)):
       plt.semilogy(hmf_emu.mbins, pred, 
                    color=colors[i], linewidth=2,
                    label=f'$\Omega_m = {om}$')
   
   plt.xlabel(r'$\log_{10}(M / M_\odot h^{-1})$', fontsize=14)
   plt.ylabel(r'$\phi(M)$ [dex$^{-1}$ (Mpc/$h$)$^{-3}$]', fontsize=14)
   plt.title('Halo Mass Function Predictions', fontsize=16)
   plt.grid(True, alpha=0.3)
   plt.legend(fontsize=12)
   plt.tight_layout()
   plt.savefig('hmf_predictions.png', dpi=150)
   print("Plot saved as 'hmf_predictions.png'")

Part 4: Understanding Uncertainties
------------------------------------

The emulator provides uncertainty estimates:

.. code-block:: python

   # Plot predictions with error bars
   plt.figure(figsize=(10, 6))
   
   i = 1  # Middle cosmology
   pred = hmf_pred[i]
   std = np.sqrt(variances[i])
   
   # Convert std from log-space to linear space
   pred_upper = 10**(predictions[i] + std)
   pred_lower = 10**(predictions[i] - std)
   
   plt.semilogy(hmf_emu.mbins, pred, 'b-', linewidth=2, 
                label='Prediction')
   plt.fill_between(hmf_emu.mbins, pred_lower, pred_upper,
                     alpha=0.3, color='blue',
                     label='±1σ uncertainty')
   
   plt.xlabel(r'$\log_{10}(M / M_\odot h^{-1})$', fontsize=14)
   plt.ylabel(r'$\phi(M)$ [dex$^{-1}$ (Mpc/$h$)$^{-3}$]', fontsize=14)
   plt.title('HMF with Uncertainties', fontsize=16)
   plt.grid(True, alpha=0.3)
   plt.legend(fontsize=12)
   plt.tight_layout()
   plt.savefig('hmf_with_uncertainties.png', dpi=150)
   print("Plot saved as 'hmf_with_uncertainties.png'")

Part 5: Batch Processing
-------------------------

For many cosmologies, use batch processing:

.. code-block:: python

   # Generate random cosmologies
   n_samples = 50
   np.random.seed(42)  # For reproducibility
   
   # Sample uniformly in parameter space
   X_batch = np.random.uniform(
       low=[0.24, 0.04, 0.60, 1.8e-9, 0.92, -1.2, -0.5, 2.5, -0.02, 0.0],
       high=[0.40, 0.06, 0.80, 2.4e-9, 1.00, -0.8, 0.5, 3.5, 0.02, 0.15],
       size=(n_samples, 10)
   )
   
   # Make predictions for all
   batch_predictions, batch_variances = hmf_emu.predict(X_batch)
   
   print(f"Made predictions for {n_samples} cosmologies")
   print(f"Output shape: {batch_predictions.shape}")

Summary Statistics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute summary statistics
   mean_hmf = 10**batch_predictions.mean(axis=0)
   std_hmf = 10**batch_predictions.std(axis=0)
   
   plt.figure(figsize=(10, 6))
   plt.semilogy(hmf_emu.mbins, mean_hmf, 'k-', linewidth=2,
                label='Mean over 50 cosmologies')
   plt.fill_between(hmf_emu.mbins, 
                     mean_hmf - std_hmf,
                     mean_hmf + std_hmf,
                     alpha=0.3, color='gray',
                     label='Standard deviation')
   
   plt.xlabel(r'$\log_{10}(M / M_\odot h^{-1})$', fontsize=14)
   plt.ylabel(r'$\phi(M)$ [dex$^{-1}$ (Mpc/$h$)$^{-3}$]', fontsize=14)
   plt.title('HMF Statistics over Parameter Space', fontsize=16)
   plt.grid(True, alpha=0.3)
   plt.legend(fontsize=12)
   plt.tight_layout()
   plt.savefig('hmf_statistics.png', dpi=150)
   print("Plot saved as 'hmf_statistics.png'")

Exercises
---------

1. **Parameter Exploration**: Modify the fiducial cosmology and see how the power spectrum changes.

2. **Redshift Evolution**: Compute power spectra at multiple redshifts (e.g., z=0, 1, 2, 3) and plot them together.

3. **Parameter Sensitivity**: Create a grid of omega_m values and plot how the HMF changes.

4. **Custom Ranges**: Modify the parameter ranges in the batch processing example to focus on a specific region of parameter space.

Next Steps
----------

- Move on to :doc:`hmf_emulation` for more detailed HMF analysis
- Check :doc:`galaxy_clustering` to compute observable galaxy statistics
- Explore :doc:`advanced_topics` for sophisticated applications

Additional Resources
--------------------

- See the :doc:`../api/index` for complete API documentation
- Check out the example notebooks in ``emu/notebooks/``
- Read the :doc:`../introduction` for scientific background
