import os
import glob

def hcange_ngrid():
    """
    Goes through all subdirectories to change the value of Ngrid in _genic_params.ini files for faster 
    computation.
    """
    # Set the value you want to change to
    new_value = "Ngrid = 750"

    # Find all matching subdirectories
    for dir_name in glob.glob("compressed_10p_Box250_Part750_*"):
        ini_path = os.path.join(dir_name, "_genic_params.ini")
        if os.path.isfile(ini_path):
            with open(ini_path, 'r') as file:
                lines = file.readlines()

            with open(ini_path, 'w') as file:
                for line in lines:
                    if line.strip().startswith("Ngrid"):
                        file.write(new_value + '\n')
                    else:
                        file.write(line)

    print("Update completed.")



all_files = []
# Find all matching subdirectories in sorted order for determinism
files = sorted(glob.glob("compressed_10p_Box250_Part750_*"))
for dir_name in files:
    # Skip directories that are not valid or already processed
    if os.path.isdir(dir_name) and dir_name != 'compressed_10p_Box250_Part750_0000' and dir_name != 'compressed_10p_Box250_Part750_0001' and dir_name != 'compressed_10p_Box250_Part750_0002':  # Exclude files like .tar
        all_files.append(dir_name)
print(f"Found {len(all_files)} directories matching the pattern.")

for i, dir_name in enumerate(all_files[24:48]):
    src_path = os.path.abspath("submit_gen_ic")
    dst_path = os.path.join(dir_name, "submit_gen_ic")
    os.system(f"cp {src_path} {dst_path}")
    os.chdir(dir_name)
    os.system("sbatch submit_gen_ic")
    os.chdir("..")