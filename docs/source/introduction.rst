Introduction
============

What is Goku-ELG?
-----------------

**Goku-ELG** is a cosmological emulator for emission-line galaxies (ELGs), built using the GOKU simulation suite. 
It provides a fast and accurate surrogate model for predicting galaxy clustering statistics across a wide range of cosmological parameters.

The Science Behind It
---------------------

For Astrophysicists
~~~~~~~~~~~~~~~~~~~

Resolving emission-line galaxies in N-body simulations requires high resolution, while achieving high-fidelity clustering 
statistics demands large volumes. We address both challenges **efficiently** by using machine learning models trained on 
the GOKU simulation suite (`Yang et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025PhRvD.111h3529Y/abstract>`_).

Our method employs a **multi-fidelity Gaussian Process model** (`Kennedy & O'Hagan 2000 <https://academic.oup.com/biomet/article-abstract/87/1/1/221217?redirectedFrom=PDF>`_, 
`Ho et al. 2021 <https://arxiv.org/abs/2105.01081>`_), along with elements of the `Stochastic Variational GP <https://arxiv.org/pdf/1411.2005>`_ 
framework, extended to support datasets with varying uncertainty levels through a modified likelihood.

Technical Approach
~~~~~~~~~~~~~~~~~~

The emulator combines several advanced techniques:

1. **Bayesian Experimental Design**: Strategic selection of simulation runs in the parameter space
2. **Multi-fidelity Gaussian Processes**: Leveraging simulations at different resolutions and volumes
3. **Modified Stochastic Variational Framework**: Handling datasets with heterogeneous uncertainties
4. **Halo Model Framework**: Converting halo statistics to observable galaxy clustering

Key Summary Statistics
----------------------

The emulator provides predictions for:

- **Halo Mass Function (HMF)**: :math:`\phi(M) dM` - the abundance of dark matter halos as a function of mass
- **Halo-Halo Correlation Function**: :math:`\xi_{hh}(r, M_{th1}, M_{th2})` - spatial clustering of halos
- **Galaxy-Galaxy Correlation Function**: :math:`\xi_{gg}(r)` - observable galaxy clustering
- **Projected Correlation Function**: :math:`w_p(r_p)` - projected galaxy clustering

Computational Pipeline
----------------------

The computational flow follows this sequence:

.. code-block:: text

   HMF Emulator → ϕ(M) dM
                   ↓
   Halo-Halo ξ Emulator → ξ_hh(M_th1, M_th2)
                           ↓
   Hankel Transform → P_hh(k, M_th1, M_th2)
                      ↓
   HOD Model → P_gg(k)
               ↓
   Inverse Hankel Transform → ξ_gg(r)

Cosmological Parameters
-----------------------

The emulator covers a 10-dimensional parameter space:

1. :math:`\Omega_m` - Total matter density
2. :math:`\Omega_b` - Baryon density
3. :math:`h` - Hubble parameter
4. :math:`A_s` - Scalar amplitude
5. :math:`n_s` - Scalar spectral index
6. :math:`w_0` - Dark energy equation of state (present)
7. :math:`w_a` - Dark energy equation of state (evolution)
8. :math:`N_{ur}` - Effective number of ultra-relativistic species
9. :math:`\alpha_s` - Running of spectral index
10. :math:`m_\nu` - Neutrino mass

Performance Validation
----------------------

Our emulator achieves:

- **< 1% accuracy** in cross-validation tests for galaxy-galaxy clustering
- **Percent-level precision** for halo mass function predictions
- **Robust extrapolation** within the trained parameter space
- **Fast evaluation**: Orders of magnitude faster than running N-body simulations

Applications
------------

Goku-ELG is designed for:

- **Cosmological parameter inference** from galaxy surveys
- **Mock catalog generation** for survey planning
- **Fisher forecasts** for upcoming surveys
- **Bayesian inference** with MCMC or nested sampling
- **Rapid prototyping** of galaxy clustering models

Target Surveys
~~~~~~~~~~~~~~

This emulator is particularly suited for:

- HETDEX (Hobby-Eberly Telescope Dark Energy Experiment)
- DESI (Dark Energy Spectroscopic Instrument)
- Euclid
- Roman Space Telescope
- Other emission-line galaxy surveys
