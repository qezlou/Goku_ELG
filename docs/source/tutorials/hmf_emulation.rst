HMF Emulation Tutorial
======================

This tutorial focuses on Halo Mass Function (HMF) emulation, a key component of Goku-ELG.

Learning Objectives
-------------------

- Understand the halo mass function and its importance
- Load and prepare HMF training data
- Train single-fidelity and multi-fidelity emulators
- Perform cross-validation
- Interpret and validate results

What is the Halo Mass Function?
--------------------------------

The halo mass function :math:`\phi(M)` describes the number density of dark matter halos as a function of their mass:

.. math::

   \frac{dn}{d\log_{10}M} = \phi(M) = \ln(10) \cdot M \cdot \frac{dn}{dM}

It's a fundamental quantity in cosmology that depends on:

- Cosmological parameters (:math:`\Omega_m`, :math:`\sigma_8`, etc.)
- Redshift
- The power spectrum

Setting Up
----------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from gal_goku import emus
   import h5py
   
   # Set plotting style
   plt.style.use('seaborn-v0_8-darkgrid')

Loading Training Data
---------------------

.. code-block:: python

   # Initialize the HMF emulator with training data
   data_dir = 'path/to/hmf/data'  # Update this path
   
   hmf_emu = emus.Hmf(
       data_dir=data_dir,
       y_log=True,              # Work in log-space
       fid='L2',                # Fiducial simulation
       multi_bin=False,         # Single-bin mode
       logging_level='INFO',
       narrow=False,            # Use full parameter range
       no_merge=True            # Don't merge datasets
   )
   
   print(f"Loaded {len(hmf_emu.labels)} training simulations")
   print(f"Mass bins: {len(hmf_emu.mbins)}")
   print(f"Mass range: {hmf_emu.mbins.min():.2f} to {hmf_emu.mbins.max():.2f}")

Training the Emulator
---------------------

Simple Training
~~~~~~~~~~~~~~~

.. code-block:: python

   # Train on all simulations and compare with truth
   predictions, truth, mass_bins = hmf_emu.train_pred_all_sims()
   
   print(f"Training complete!")
   print(f"Predictions shape: {predictions.shape}")
   print(f"Truth shape: {truth.shape}")

Visualizing Training Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot a few examples
   n_examples = 4
   fig, axes = plt.subplots(2, 2, figsize=(14, 10))
   axes = axes.flatten()
   
   for i in range(n_examples):
       ax = axes[i]
       
       # Plot truth and prediction
       ax.plot(mass_bins, truth[i], 'ko-', label='Truth', alpha=0.6)
       ax.plot(mass_bins, predictions[i], 'r^-', label='Prediction', alpha=0.8)
       
       # Calculate residual
       residual = np.abs((predictions[i] - truth[i]) / truth[i]) * 100
       
       ax.set_xlabel(r'$\log_{10}(M / M_\odot h^{-1})$')
       ax.set_ylabel(r'$\log_{10}[\phi(M)]$')
       ax.set_title(f'Simulation {i}: Mean error = {residual.mean():.2f}%')
       ax.legend()
       ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig('hmf_training_examples.png', dpi=150)
   print("Saved training examples plot")

Cross-Validation
----------------

Leave-One-Out Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Perform leave-one-out cross-validation
   savefile = 'hmf_loo_results.h5'
   hmf_emu.loo_train_pred(savefile=savefile, narrow=0)
   
   print(f"LOO results saved to {savefile}")

Analyzing LOO Results
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load and analyze LOO results
   with h5py.File(savefile, 'r') as f:
       loo_predictions = f['predictions'][:]
       loo_truth = f['truth'][:]
       loo_variances = f['variances'][:]
   
   # Compute errors
   absolute_errors = loo_predictions - loo_truth
   relative_errors = (absolute_errors / loo_truth) * 100
   
   print(f"Mean absolute error: {np.mean(np.abs(absolute_errors)):.4f}")
   print(f"Mean relative error: {np.mean(np.abs(relative_errors)):.2f}%")
   print(f"Max relative error: {np.max(np.abs(relative_errors)):.2f}%")

Visualizing LOO Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, axes = plt.subplots(1, 2, figsize=(14, 5))
   
   # Left: Scatter plot of predictions vs truth
   ax = axes[0]
   ax.scatter(loo_truth.flatten(), loo_predictions.flatten(),
              alpha=0.3, s=5)
   
   # Perfect prediction line
   lims = [loo_truth.min(), loo_truth.max()]
   ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
   
   ax.set_xlabel('True log(HMF)', fontsize=12)
   ax.set_ylabel('Predicted log(HMF)', fontsize=12)
   ax.set_title('Leave-One-Out: Predictions vs Truth', fontsize=14)
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Right: Residual distribution
   ax = axes[1]
   ax.hist(relative_errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
   ax.axvline(0, color='red', linestyle='--', linewidth=2)
   ax.set_xlabel('Relative Error [%]', fontsize=12)
   ax.set_ylabel('Frequency', fontsize=12)
   ax.set_title('Distribution of Relative Errors', fontsize=14)
   ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig('loo_performance.png', dpi=150)
   print("Saved LOO performance plot")

Leave-Bunch-Out Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Leave out 5 simulations at a time
   n_out = 5
   X_test, Y_test, Y_pred, var_pred = hmf_emu.leave_bunch_out(n_out=n_out)
   
   print(f"Left out {n_out} simulations")
   print(f"Test set shape: {Y_test.shape}")
   
   # Compute errors
   lbo_errors = np.abs((Y_pred - Y_test) / Y_test) * 100
   print(f"Mean LBO error: {lbo_errors.mean():.2f}%")

Making Predictions
------------------

Single Cosmology
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define a test cosmology
   test_cosmo = np.array([[
       0.30,      # omega_m
       0.048,     # omega_b
       0.70,      # h
       2.0e-9,    # A_s
       0.96,      # n_s
       -1.0,      # w0
       0.0,       # wa
       3.046,     # N_ur
       0.0,       # alpha_s
       0.06       # m_nu
   ]])
   
   # Predict
   pred, var = hmf_emu.predict(test_cosmo)
   
   # Convert from log-space
   hmf_pred = 10**pred[0]
   
   # Plot
   plt.figure(figsize=(10, 6))
   plt.semilogy(mass_bins, hmf_pred, 'b-', linewidth=2)
   plt.xlabel(r'$\log_{10}(M / M_\odot h^{-1})$', fontsize=14)
   plt.ylabel(r'$\phi(M)$ [dex$^{-1}$ (Mpc/$h$)$^{-3}$]', fontsize=14)
   plt.title('HMF Prediction for Test Cosmology', fontsize=16)
   plt.grid(True, alpha=0.3)
   plt.savefig('hmf_test_prediction.png', dpi=150)

Parameter Dependence Study
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Study how HMF changes with omega_m
   omega_m_values = np.linspace(0.26, 0.36, 11)
   
   hmf_vs_omega_m = []
   
   for om in omega_m_values:
       test_cosmo = np.array([[om, 0.048, 0.70, 2.0e-9, 0.96,
                               -1.0, 0.0, 3.046, 0.0, 0.06]])
       pred, _ = hmf_emu.predict(test_cosmo)
       hmf_vs_omega_m.append(10**pred[0])
   
   hmf_vs_omega_m = np.array(hmf_vs_omega_m)
   
   # Plot results
   plt.figure(figsize=(12, 6))
   
   # Select a few mass bins to plot
   mass_indices = [5, 10, 15, 20]
   
   for idx in mass_indices:
       mass = mass_bins[idx]
       plt.plot(omega_m_values, hmf_vs_omega_m[:, idx],
                'o-', linewidth=2, label=f'log(M) = {mass:.1f}')
   
   plt.xlabel(r'$\Omega_m$', fontsize=14)
   plt.ylabel(r'$\phi(M)$ [dex$^{-1}$ (Mpc/$h$)$^{-3}$]', fontsize=14)
   plt.title('HMF Dependence on Matter Density', fontsize=16)
   plt.legend(fontsize=11)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('hmf_vs_omega_m.png', dpi=150)

Advanced: Multi-Fidelity Emulation
-----------------------------------

Using Multiple Fidelities
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize multi-fidelity emulator
   hmf_emu_mf = emus.Hmf(
       data_dir=data_dir,
       y_log=True,
       fid='L2',
       multi_bin=True,          # Enable multi-bin mode
       logging_level='INFO'
   )
   
   print("Multi-fidelity emulator initialized")

The multi-fidelity approach combines:

1. **High-fidelity**: Large-box, high-resolution simulations (expensive)
2. **Low-fidelity**: Smaller-box or lower-resolution simulations (cheap)

This improves accuracy while reducing computational cost.

Best Practices
--------------

1. **Always use log-space** for HMF to handle the large dynamic range
2. **Check cross-validation** before using for science
3. **Monitor uncertainties** - large uncertainties indicate extrapolation
4. **Validate against known results** when possible
5. **Use appropriate mass range** - avoid extrapolating beyond training data

Common Issues
-------------

Issue: High Errors at Low Masses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Low number of halos in small mass bins leads to noisy measurements.

**Solution**: 
- Increase simulation volume
- Use wider mass bins
- Apply smoothing

Issue: Poor Extrapolation
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Testing cosmologies far from training set.

**Solution**:
- Check if test parameters are within training range
- Add more training simulations
- Use informative priors

Exercises
---------

1. **Redshift Dependence**: Modify the tutorial to study HMF at different redshifts.

2. **Parameter Sensitivity**: Create a 2D plot showing HMF sensitivity to both omega_m and sigma_8.

3. **Custom Validation**: Implement k-fold cross-validation instead of leave-one-out.

4. **Compare with Theory**: Compare emulator predictions with theoretical HMF (e.g., Tinker et al. 2008).

Next Steps
----------

- Move on to :doc:`galaxy_clustering` to convert HMF to observable quantities
- Explore :doc:`advanced_topics` for sophisticated GP techniques
- Check the notebooks in ``emu/notebooks/hmf_emu/`` for more examples
