import os
import shutil

# Set the base directory and the target save directory
base_dir = '/scratch/06536/qezlou/Goku/FOF/HF'
svedir = '/scratch/06536/qezlou/Goku/processed_data/power_matter/HF/'  # Change this to your desired destination directory

# Ensure the save directory exists
os.makedirs(svedir, exist_ok=True)

# Loop over all subdirectories in base_dir
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    
    if os.path.isdir(subdir_path):
        target_file = os.path.join(subdir_path, 'output', 'powerspectrum-0.2857.txt')
        
        if os.path.isfile(target_file):
            dest_file = os.path.join(svedir, f'power_matter_{subdir}.txt')
            shutil.copy(target_file, dest_file)
            print(f'Copied {target_file} to {dest_file}')
        else:
            print(f'File not found in {subdir_path}')

print('Done.')