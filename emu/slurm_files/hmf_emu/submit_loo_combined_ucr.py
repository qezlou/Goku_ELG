import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -J H-{i}
#SBATCH -p intel
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date

source /rhome/mqezl001/bigdata/HETDEX/.gal_env/bin/activate
which python
python run_hmf_emu_combined_bins.py --ind_test {i} --z 2.5 --machine ucr --config pca_w_m32_m52_m32_m52_learn_hetero.json
"""

# Loop from 34 to 36
for i in np.arange(19, 20):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')
