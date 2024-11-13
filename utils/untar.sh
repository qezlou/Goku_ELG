prefix=$1  # First argument: file name prefix e.g. cosmo_10p_Box250_Part750_
lower=$2   # Second argument: lower bound
upper=$3   # Third argument: upper bound

batch_size=10  # Define batch size
current_lower=$lower

while [ $current_lower -le $upper ]; do
    current_upper=$((current_lower + batch_size - 1))
    if [ $current_upper -gt $upper ]; then
        current_upper=$upper  # Cap the upper bound to the final value
    fi

    # Name the screen session based on current batch
    screen_name="L2-untar-$current_lower-$current_upper"
    
    # Run the untar script inside a new screen session
    screen -dmS "$screen_name" bash -c "
    for i in \$(seq $current_lower $current_upper); do
        formatted_num=\$(printf \"%04d\" \"\$i\")
        tar -xvf \"${prefix}\${formatted_num}.tar\"
    done
    "

    # Move to the next batch
    current_lower=$((current_upper + 1))
done
