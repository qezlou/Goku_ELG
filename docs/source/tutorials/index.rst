Tutorials
=========

This section contains tutorials and examples for using Goku-ELG.

.. toctree::
   :maxdepth: 2

   basic_usage
   hmf_emulation
   galaxy_clustering
   advanced_topics

Available Notebooks
-------------------

The ``emu/notebooks/`` directory contains many Jupyter notebooks demonstrating various features:

Halo Mass Function
~~~~~~~~~~~~~~~~~~

- **hmf.ipynb**: Basic HMF computations
- **hmf_emu.ipynb**: Training HMF emulators
- **hmf_emu_multifid.ipynb**: Multi-fidelity HMF emulation
- **hmf_comp_kernel.ipynb**: Comparing different GP kernels

Correlation Functions
~~~~~~~~~~~~~~~~~~~~~

- **xi_on_grid.ipynb**: Computing correlation functions on a grid
- **fft_corr.ipynb**: FFT-based correlation function computations
- **xi_dim_reduc_gp.ipynb**: Dimensionality reduction for correlation functions

Power Spectrum
~~~~~~~~~~~~~~

- **power.ipynb**: Linear power spectrum calculations
- **p_m.ipynb**: Matter power spectrum
- **pk.ipynb**: Power spectrum emulation

Galaxy Clustering
~~~~~~~~~~~~~~~~~

- **corr_for_pigs.ipynb**: Galaxy correlation functions
- **single_fid.ipynb**: Single-fidelity emulation
- **false_positive.ipynb**: Validation and testing

Advanced Topics
~~~~~~~~~~~~~~~

- **play_gpflow.ipynb**: Exploring GPflow features
- **linear_SVGP.ipynb**: Stochastic Variational GPs
- **additive_gp.ipynb**: Additive Gaussian Processes
- **pca_weighted.ipynb**: Weighted PCA for dimensionality reduction

Tutorial Structure
------------------

Each tutorial covers:

1. **Objective**: What you'll learn
2. **Prerequisites**: Required knowledge and packages
3. **Step-by-step guide**: Detailed instructions with code
4. **Results**: Expected outputs and interpretation
5. **Exercises**: Practice problems (where applicable)

Getting Started
---------------

We recommend following the tutorials in this order:

1. :doc:`basic_usage` - Start here if you're new to Goku-ELG
2. :doc:`hmf_emulation` - Learn about HMF emulation
3. :doc:`galaxy_clustering` - Compute galaxy clustering statistics
4. :doc:`advanced_topics` - Dive into advanced features

Running the Notebooks
---------------------

To run the example notebooks:

.. code-block:: bash

   cd emu/notebooks
   jupyter notebook

Then navigate to the notebook you want to explore.

Contributing Tutorials
----------------------

We welcome tutorial contributions! If you've developed a useful workflow or example:

1. Fork the repository
2. Add your notebook to ``emu/notebooks/``
3. Document it clearly with markdown cells
4. Submit a pull request

Questions?
----------

If you have questions about the tutorials:

- Open an issue on GitHub
- Check the :doc:`../api/index` for detailed API documentation
- Contact the developers (see :doc:`../citation`)
