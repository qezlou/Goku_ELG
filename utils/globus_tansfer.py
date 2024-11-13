import os
import os.path as op
import argparse

fidelity = 'L2'

box = {'HF':'1000', 'L1':'1000', 'L2':'250'}
part = {'HF':'3000', 'L1':'750', 'L2':'750'}


def run_globus(fidelity, start, end):
    #Ranch
    source_end_point = 'e6d7586e-c815-4f11-9a90-37d1747989c1'
    # Stampede3
    destination_end_point = '1e9ddd41-fe4b-406f-95ff-f3d79f9cb523'

    source_path = '/stornext/ranch_01/ranch/projects/AST21005/Goku_emulator_sims/Goku_sims/'
    destination_path = f'/scratch/06536/qezlou/Goku/FOF/{fidelity}'

    # Create a file of the tar to transfer
    file_list = op.join(destination_path, f'transfer_list_{round(start)}-{round(end)}.txt')



    with open(op.join(file_list),'w') as f:
        for i in range(start, end):
            s = op.join(source_path, f'compressed_10p_Box{box[fidelity]}_Part{part[fidelity]}_{str(i).rjust(4,'0')}.tar')
            d = op.join(destination_path, f'compressed_10p_Box{box[fidelity]}_Part{part[fidelity]}_{str(i).rjust(4,'0')}.tar')
            f.write(f'{s} {d} \n')

    os.system(f'globus transfer {source_end_point} {destination_end_point} --batch {file_list}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fidelity', type=str, help='e.g HF')
    parser.add_argument('--start', type=int, help='e.g. 200')
    parser.add_argument('--end', type=int, help='e.g 300')
    args = parser.parse_args()
    run_globus(args.fidelity, args.start, args.end)

