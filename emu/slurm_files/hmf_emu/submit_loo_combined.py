import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -A AST25019
#SBATCH -J comb{i}
#SBATCH -p skx
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date
export LD_LIBRARY_PATH=/work2/06536/qezlou/stampede3/miniconda3/envs/py3.12/lib:$LD_LIBRARY_PATH
python run_hmf_emu_combined_bins.py --ind_test {i} --z 2.5
"""

# Loop from 0 to num_chunks
for i in np.arange(30, 34):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')
