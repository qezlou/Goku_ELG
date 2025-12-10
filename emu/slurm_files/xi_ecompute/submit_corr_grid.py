import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -A AST25019
#SBATCH -J w9.0-{i}
#SBATCH -p icx
#SBATCH -N 1
#SBATCH --ntasks-per-node 80
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date
export LD_PRELOAD=$CONDA_PREFIX/lib/libssl.so:$CONDA_PREFIX/lib/libcrypto.so
ibrun python run_xi_train.py --fid 'L2' --narrow 1 --numchunks 23 --chunk {i} --stat_type 'xi_hh' --z 9.0
"""

# Loop from 0 to 20 and create modified files
for i in range(15, 23):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')
