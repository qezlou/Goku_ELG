#!/bin/bash

num_jobs=3  # Number of parallel jobs
tar_files=(*.tar)
total_files=${#tar_files[@]}

if [ "$total_files" -eq 0 ]; then
    echo "No .tar files found in the current directory."
    exit 1
fi

# Build a string that redefines the tar_files array in the screen session.
# This simple approach works as long as your file names don't contain spaces.
tar_files_list=""
for file in "${tar_files[@]}"; do
    tar_files_list+="'${file}' "
done

# Function to process files inside a screen session
extract_files() {
    local job_id=$1
    local screen_name="untar-job-$job_id"

    # Start a new screen session and run the extraction.
    # We embed the array definition and other needed variables into the command.
    screen -S "$screen_name" -dm bash -c "tar_files=(${tar_files_list});
total_files=\${#tar_files[@]};
for ((j=${job_id}; j<total_files; j+=${num_jobs})); do
    echo \"Extracting \${tar_files[j]} (Job ${job_id})...\";
    tar -xvf \"\${tar_files[j]}\";
done;
echo \"Extraction completed in ${screen_name}\";
exec bash"  # Keeps the screen session open after completion.
}

# Start jobs in separate screen sessions
for ((i = 0; i < num_jobs; i++)); do
    extract_files $i
done

echo "Extraction started in $num_jobs screen sessions."
echo "Use 'screen -ls' to list running sessions."
echo "Attach to a session with: screen -r untar-job-<ID>"
echo "For example: screen -r untar-job-0"
