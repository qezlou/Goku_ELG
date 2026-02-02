import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -A AST25019
#SBATCH -J H1m-{i}
#SBATCH -p spr
#SBATCH -N 1
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date
source /scratch/06536/qezlou/Goku/packs/.gal_env/bin/activate
python run_hmf_emu_combined_bins.py --ind_test {i} --z 1.0 --machine stampede3 --config pca_w_m32_m52_m32_m52_learn_hetero.json
"""

# Loop from 0 to num_chunks
for i in np.arange(0, 24):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')
