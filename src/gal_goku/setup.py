import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import os
import shutil

class CustomInstallCommand(install):
    """Customized install command - clones repos, copies files, and installs classylss."""

    def run(self):
        self.do_custom_steps()
        super().run()

    def do_custom_steps(self):
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        print(f'Parent directory: {parent_dir}', flush=True)

        # Clone classylss repository
        classylss_repo = os.path.join(parent_dir, 'classylss')
        if not os.path.exists(classylss_repo):
            print('Cloning classylss repository...', flush=True)
            subprocess.check_call(['git', 'clone', 'https://github.com/sbird/classylss.git', classylss_repo])
        else:
            print(f'{classylss_repo} already exists, skipping clone.', flush=True)

        # Clone class_public repository
        class_public_repo = os.path.join(parent_dir, 'class_public')
        if not os.path.exists(class_public_repo):
            print('Cloning class_public repository...', flush=True)
            subprocess.check_call(['git', 'clone', 'https://github.com/lesgourg/class_public.git', class_public_repo])
        else:
            print(f'{class_public_repo} already exists, skipping clone.', flush=True)

        # Copy ./class_public/external/ to ./classylss/classylss/data/
        src_dir = os.path.join(class_public_repo, 'external')
        dest_dir = os.path.join(classylss_repo, 'classylss', 'data')

        if os.path.exists(src_dir):
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            s = os.path.join(src_dir, 'bbn')
            d = os.path.join(dest_dir, 'bbn')
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
            print('File bbn copied successfully.', flush=True)
        else:
            print(f'Source directory {src_dir} does not exist.', flush=True)

        # Install classylss
        subprocess.check_call(['python', '-m', 'pip', 'install', classylss_repo])

        # Install multi_fidelity_gpflow
        subprocess.check_call(['python', '-m', 'pip', 'install', 'git+https://github.com/qezlou/multi_fidelity_gpflow.git'])


class CustomDevelopCommand(develop):
    """Custom develop command - same as custom install."""

    def run(self):
        self.do_custom_steps()
        super().run()

    def do_custom_steps(self):
        CustomInstallCommand.do_custom_steps(self)

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
        "configobj"
    ],
    cmdclass={
        'install': CustomInstallCommand,  # This will automatically trigger during installation
        'develop': CustomDevelopCommand,  # This will automatically trigger during development mode
    },
)
