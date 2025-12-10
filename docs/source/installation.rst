Installation
============

Requirements
------------

Python Version
~~~~~~~~~~~~~~

Goku-ELG requires Python 3.8 or higher. Python 3.12 is recommended for optimal performance.

Dependencies
~~~~~~~~~~~~

Core dependencies:

- numpy
- scipy >= 1.15.1
- matplotlib
- scikit-learn
- h5py
- emukit
- GPflow (for Gaussian Process models)

Installation Steps
------------------

Step 1: Create a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using a dedicated conda environment:

.. code-block:: bash

   conda create -n gal_goku python=3.12 numpy scipy=1.15.1 matplotlib
   conda activate gal_goku

Step 2: Install Additional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the required Python packages:

.. code-block:: bash

   python -m pip install emukit
   python -m pip install scikit-learn
   python -m pip install h5py

Step 3: Install Goku-ELG
~~~~~~~~~~~~~~~~~~~~~~~~~

There are two main packages to install:

**Option A: Install both packages**

.. code-block:: bash

   # Navigate to the gal_goku package
   cd src/gal_goku
   python -m pip install -e .
   
   # Navigate to the gal_goku_sims package
   cd ../gal_goku_sims
   python -m pip install -e .

**Option B: Install from the root directory**

If you have a setup script in the root:

.. code-block:: bash

   python -m pip install -e .

Special Dependencies
--------------------

ClassyLSS (Optional but Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For computing linear power spectra, the package uses `classylss`. The installation script 
will automatically clone and set up the required repositories:

1. `classylss <https://github.com/sbird/classylss>`_
2. `class_public <https://github.com/lesgourg/class_public>`_

These will be cloned automatically during installation of the ``gal_goku`` package.

Manual Installation
~~~~~~~~~~~~~~~~~~~

If you need to install classylss manually:

.. code-block:: bash

   # Clone repositories
   git clone https://github.com/sbird/classylss.git
   git clone https://github.com/lesgourg/class_public.git
   
   # Copy external data
   cp -r class_public/external/bbn classylss/classylss/data/
   
   # Install classylss
   cd classylss
   python -m pip install -e .

Verification
------------

To verify your installation, try importing the packages:

.. code-block:: python

   import gal_goku
   from gal_goku import emus, gal
   from gal_goku_sims import hmf, xi
   
   print("Installation successful!")

Running Tests
~~~~~~~~~~~~~

You can verify the installation by running a simple test:

.. code-block:: python

   import numpy as np
   from gal_goku import emus
   
   # This should run without errors if installation is correct
   print("Goku-ELG is ready to use!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**

If you encounter import errors, make sure you've activated the correct conda environment:

.. code-block:: bash

   conda activate gal_goku

**Missing Dependencies**

If specific dependencies are missing, install them individually:

.. code-block:: bash

   pip install <package-name>

**ClassyLSS Issues**

ClassyLSS requires a C compiler. On Linux, ensure you have gcc installed:

.. code-block:: bash

   sudo apt-get install gcc  # Ubuntu/Debian
   sudo yum install gcc       # CentOS/RHEL

Development Installation
------------------------

For developers who want to contribute:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/qezlou/private-gal-emu.git
   cd private-gal-emu
   
   # Create development environment
   conda create -n gal_goku_dev python=3.12
   conda activate gal_goku_dev
   
   # Install in editable mode with development dependencies
   cd src/gal_goku
   pip install -e ".[dev]"

GPU Support (Optional)
----------------------

For accelerated Gaussian Process training with GPUs, install TensorFlow with GPU support:

.. code-block:: bash

   pip install tensorflow-gpu
   
   # Verify GPU is available
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Next Steps
----------

After installation, check out the :doc:`quickstart` guide to learn how to use Goku-ELG.
