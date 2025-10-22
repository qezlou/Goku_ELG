Goku-ELG Documentation
======================

.. image:: https://img.shields.io/badge/status-under--development-orange
   :alt: Under Development

.. image:: https://img.shields.io/badge/built%20with-GPflow-2ea44f
   :target: https://gpflow.github.io/
   :alt: Built with GPflow

.. image:: https://img.shields.io/badge/arXiv-preprint%20coming%20soon-blue
   :target: https://arxiv.org/
   :alt: arXiv Preprint Coming Soon

**Goku-ELG** is a percent-level accurate cosmological surrogate model for emission-line galaxies (ELGs), replacing expensive N-body simulations for modeling galaxy clustering.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   quick_reference
   tutorials/index
   api/index
   citation
   license

Overview
--------

The Problem
~~~~~~~~~~~

Modeling cosmological observations with high fidelity typically requires computationally expensive forward simulations of large-scale structure.
Because these simulations are slow and costly, researchers often resort to simplified or approximate models, sacrificing physical accuracy and precision.

Our Solution
~~~~~~~~~~~~

Using a **Bayesian experimental design** strategy, we carefully select a limited number of simulation runs within a **10-dimensional cosmological parameter space**.
We then train a **multi-fidelity Gaussian Process (GP)** surrogate on these simulation results to emulate the observed clustering signal.

This emulator achieves **percent-level cross-validation accuracy**, enabling **fast and reliable inference** through **Markov Chain Monte Carlo (MCMC)** sampling without the need for repeated, expensive N-body simulations.

Key Features
------------

- **High Accuracy**: Percent-level cross-validation accuracy for galaxy clustering statistics
- **Multi-fidelity GP**: Advanced Gaussian Process modeling combining multiple simulation fidelities
- **10D Parameter Space**: Comprehensive coverage of cosmological parameters
- **Fast Inference**: Enables rapid MCMC sampling without repeated N-body simulations
- **Built on GOKU Suite**: Leverages the GOKU simulation suite for training

Interactive Demo
----------------

Try our interactive galaxy clustering demo: `Goku-ELG Interactive Demo <https://qezlou.github.io/gal-clustering-viz/change_one/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
