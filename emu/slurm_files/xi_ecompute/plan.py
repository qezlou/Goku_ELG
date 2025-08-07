import os

all_fids = ['HF', 'L2']
narrow = [True, False]
all_zs = [0.5, 0.0]

# Print table header
print(f"{'Fid':<5} {'z':<4} {'Narrow':<8} {'# Sims':<8} {'# xi files':<12}")
print('-' * 44)

for z in all_zs:
    print('-' * 22)
    for fid in all_fids:
        for narrow_flag in narrow:
            if not narrow_flag:
                snap_dir = f'/scratch/06536/qezlou/Goku/FOF/{fid}'
                narrow_str = 'W'
            else:
                snap_dir = f'/scratch/06536/qezlou/Goku/FOF/{fid}/narrow'
                narrow_str = 'N'
            
            # Find all the sims on scratch
            sims = [f for f in os.listdir(snap_dir) if f.startswith('compressed')]

            # Find computed xi files
            if not narrow_flag:
                xi_Dir = f'/scratch/06536/qezlou/Goku/processed_data/xi_bins/{fid}'
            else:
                xi_Dir = f'/scratch/06536/qezlou/Goku/processed_data/xi_bins/{fid}/narrow'
            xi_files = [f for f in os.listdir(xi_Dir) if f.startswith('compressed') and f.endswith(f'z{z}.hdf5')]

            # Print row
            print(f"{fid:<5} {z:<4} {narrow_str:<8} {len(sims):<8} {len(xi_files):<12}")
