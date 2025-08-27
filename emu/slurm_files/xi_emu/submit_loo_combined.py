import os
import numpy as np

# Define the template for the modified lines
template = """#!/bin/bash
#SBATCH -J comb{i}
#SBATCH -p spr
#SBATCH -N 1
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date
export LD_LIBRARY_PATH=/work2/06536/qezlou/stampede3/miniconda3/envs/py3.12/lib:$LD_LIBRARY_PATH
python run_emu_combined_bins.py --ind_test {i} --remove_sims 26 522 524
"""

# Loop from 0 to num_chunks
for i in np.arange(20, 33):
    print(i)
    filename = f"job_script_{i}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=i))
    os.system(f'sbatch job_script_{i}.sh')
    os.remove(f'job_script_{i}.sh')
