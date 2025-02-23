import os

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -J mg{i}
#SBATCH -p skx
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date

ibrun python run_xi_train.py --fid 'L2' --narrow 0 --numchunks 20 --chunk {i}
"""

# Loop from 0 to 20 and create modified files
for i in range(0,6):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')
