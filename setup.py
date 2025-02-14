import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gal_goku",
    version="0.0.0",
    author="Mahdi Qezlou, Yanhui Yang, Simeon Bird",
    author_email="mahdi.qezlou@email.ucr.edu",
    description="Galaxy Emulator based on Goku suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahdiqezlou/lali",
    project_urls={
        "Bug Tracker": "https://github.com/mahdiqezlou/lali",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[
    "scipy==1.15.1",
    "numpy",
    "h5py",
    ],
)
