import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -J HNmg{i}
#SBATCH -p skx
#SBATCH -N 2
#SBATCH --ntasks-per-node 48
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date

ibrun python run_xi_train.py --fid 'HF' --narrow 1 --numchunks 15 --chunk {i}
"""

# Loop from 0 to 20 and create modified files
for i in np.arange(15):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')