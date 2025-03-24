import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -J c{i}
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node 36
#SBATCH --cpus-per-task 1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date
ibrun -n 36 python run_loo_xi.py --num_chunks 1 --chunk {i}
"""

# Loop from 0 to num_chunks
for i in np.arange(1):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')