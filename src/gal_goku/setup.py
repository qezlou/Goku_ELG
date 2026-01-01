import setuptools

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup function
setuptools.setup(
    name="gal_goku",
    version="0.0.0",
    author="Mahdi Qezlou, Yanhui Yang, Simeon Bird",
    author_email="mahdi.qezlou@email.ucr.edu",
    description="Galaxy Emulator based on Goku suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qezlou/private-HETDEX-cosmo",
    project_urls={
        "Bug Tracker": "https://github.com/qezlou/private-HETDEX-cosmo",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        "scipy",
        "scikit-learn",
        "matplotlib",
        "h5py",
        "mcfit",
        "cython",
        "gpflow",
        "tensorflow~=2.19.0",
        "tensorflow-probability~=0.25.0",
        "colossus",
        "camb",
        "configobj",
        "mfgpflow @ git+https://github.com/qezlou/multi_fidelity_gpflow.git@learnable-heteroscedasticity",
        "classylss @ git+https://github.com/sbird/classylss.git",
    ],
)
