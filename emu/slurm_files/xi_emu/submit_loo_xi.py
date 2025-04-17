import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -J c{i}
#SBATCH -p spr
#SBATCH -N 4
#SBATCH --ntasks-per-node 14
#SBATCH --cpus-per-task 8
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date
export LD_LIBRARY_PATH=/work2/06536/qezlou/stampede3/miniconda3/envs/py3.12/lib:$LD_LIBRARY_PATH
ibrun python run_loo_xi.py --num_chunks 2 --chunk {i}
"""

# Loop from 0 to num_chunks
for i in np.arange(0,2):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')