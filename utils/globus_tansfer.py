import os
import numpy as np
import os.path as op
import argparse

box = {'HF':'1000', 'L1':'1000', 'L2':'250'}
part = {'HF':'3000', 'L1':'750', 'L2':'750'}


def run_globus(fidelity, start, end):
    #Ranch
    source_end_point = 'e6d7586e-c815-4f11-9a90-37d1747989c1'
    # Stampede3
    destination_end_point = '1e9ddd41-fe4b-406f-95ff-f3d79f9cb523'

    source_path = '/stornext/ranch_01/ranch/projects/AST21005/Goku_emulator_sims/Goku_narrow_sims/'
    destination_path = f'/scratch/06536/qezlou/Goku/FOF/{fidelity}/narrow/'

    pattern = f'Box{box[fidelity]}_Part{part[fidelity]}'
    print(f'pattern is {pattern}')

    if start >= 0:
        # Create a file of the tar to transfer
        file_list = op.join(destination_path, f'transfer_list_{fidelity}_{round(start)}-{round(end)}.txt')
        with open(op.join(file_list),'w') as f:
            for i in range(start, end):
                s = op.join(source_path, f'compressed_10p_{pattern}_{str(i).rjust(4,"0")}.tar')
                d = op.join(destination_path, f'compressed_10p_{pattern}_{str(i).rjust(4,"0")}.tar')
                f.write(f'{s} {d} \n')
    else:
        # If start is -1, it means we want to distribute all files across 3 Globus jobs
        # Create a file of the tar to transfer
        file_list = op.join(destination_path, f'transfer_list_{fidelity}')
        # Find all the .tar files and put the list on a temp file
        os.system(f'globus ls "{source_end_point}":{source_path} | grep "{pattern}" | grep ".tar" > {file_list}.temp')
    
        with open(f'{file_list}.temp', 'r') as fr:
            all_lines =fr.readlines()
        

        for j in range(3):
            with open(f'{file_list}_{j}.txt', 'w') as fw:
                files = distribute_files(all_lines, j)
                for line in files:
                    line = line.strip('\n')
                    s = op.join(source_path, line)
                    d = op.join(destination_path, line)
                    fw.write(f'{s} {d} \n')

            os.system(f'globus transfer {source_end_point} {destination_end_point} --batch {file_list}_{j}.txt')

def distribute_files(fnames, job, size=int(3)):
    """Distribute a list of files among available ranks
    fnames : a list of file names
    Returns : A list of files for each job
    """
    num_files = len(fnames)
    files_per_rank = int(num_files/size)
    #a list of file names for each job
    fnames_jobs = fnames[job*files_per_rank : (job+1)*files_per_rank]
    # Some ranks get 1 more snaphot file
    remained = int(num_files - files_per_rank*size)
    if job in range(1,remained+1):
        fnames_jobs.append(fnames[files_per_rank*size + job-1 ])
    return fnames_jobs

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fidelity', type=str, help='e.g HF')
    parser.add_argument('--start', default=-1, type=int, help='e.g. 200')
    parser.add_argument('--end', default=-1, type=int, help='e.g 300')
    args = parser.parse_args()
    run_globus(args.fidelity, args.start, args.end)

