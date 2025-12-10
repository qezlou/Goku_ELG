Citation
========

If you use Goku-ELG in your research, please cite our paper:

BibTeX Entry
------------

.. code-block:: bibtex

   @article{qezlou2025goku,
     title={Goku-ELG: A Cosmological Emulator for Emission-Line Galaxies},
     author={Qezlou, Mahdi and Yang, Yanhui and Bird, Simeon and Ho, Ming-Feng},
     journal={In preparation},
     year={2025}
   }

Related Papers
--------------

The GOKU Simulation Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~

Our emulator is built on the GOKU simulation suite. If you use the simulations or discuss them, 
please also cite:

.. code-block:: bibtex

   @article{yang2025goku,
     title={GOKU: A Fast and Accurate Cosmological Simulation Suite},
     author={Yang, Yanhui and others},
     journal={Physical Review D},
     volume={111},
     pages={083529},
     year={2025},
     doi={10.1103/PhysRevD.111.083529}
   }

Multi-Fidelity Gaussian Processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-fidelity GP methodology we use is based on:

**Kennedy & O'Hagan (2000)**

.. code-block:: bibtex

   @article{kennedy2000predicting,
     title={Predicting the output from a complex computer code when fast approximations are available},
     author={Kennedy, Marc C and O'Hagan, Anthony},
     journal={Biometrika},
     volume={87},
     number={1},
     pages={1--13},
     year={2000},
     publisher={Oxford University Press}
   }

**Ho et al. (2021)**

.. code-block:: bibtex

   @article{ho2021multifidelity,
     title={Multi-fidelity Gaussian Process Modeling for Chemical Property Prediction},
     author={Ho, Ming-Feng and others},
     journal={arXiv preprint arXiv:2105.01081},
     year={2021}
   }

Stochastic Variational Gaussian Processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the stochastic variational framework:

.. code-block:: bibtex

   @inproceedings{hensman2015scalable,
     title={Scalable Variational Gaussian Process Classification},
     author={Hensman, James and Matthews, Alexander G and Ghahramani, Zoubin},
     booktitle={International Conference on Artificial Intelligence and Statistics},
     pages={351--360},
     year={2015}
   }

Acknowledgments
---------------

This work was supported by:

- [Funding sources to be added]
- Computational resources from [institutions to be added]

The development of Goku-ELG would not have been possible without:

- The GPflow team for their excellent Gaussian Process library
- The scikit-learn developers
- The broader Python scientific computing community

Software Acknowledgments
------------------------

Goku-ELG builds upon several open-source packages:

- **GPflow**: Gaussian Process library
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **scikit-learn**: Machine learning tools
- **matplotlib**: Visualization
- **h5py**: HDF5 file handling
- **ClassyLSS**: Linear power spectrum calculations

Contact
-------

For questions, bug reports, or feature requests:

- **Email**: mahdi.qezlou@email.ucr.edu
- **GitHub**: https://github.com/qezlou/private-gal-emu
- **Issues**: https://github.com/qezlou/private-gal-emu/issues

Community
---------

We welcome contributions from the community! Please see our GitHub repository for:

- Contributing guidelines
- Issue tracker
- Discussion forums
- Code of conduct

Stay Updated
------------

- Check our GitHub repository for updates
- Watch the repository to get notifications of new releases
- Follow announcements on [social media/mailing list to be added]

License Information
-------------------

See :doc:`license` for details on usage and redistribution rights.
