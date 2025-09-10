#!/bin/bash

# Define base arguments
base_args="--lattice_cache=results/lat_cache_seed5 --lang_dir=results/lang --number_of_epochs_seed=5"
base_command="uv run train_cfmam.py hparams/cfmam.yaml"

# Define arrays for hyperparameters to explore
num_paths_train_values=(2 4 8 16 2)
num_paths_test_values=(2 2 2 2 2)
learning_rates=(0.001 0.01 0.1)
step_sizes=(0.01 0.05 0.1)
nbest_scales=(0.5 1.0 1.5)
acoustic_scales=(1.0 1.5 2.0)
use_embeddings_values=(true false)
use_frozen_feats_values=(false)
num_exact_log_p_values=(1 2)

# Initialize trial counter
trial_counter=1

# Loop through all combinations of hyperparameters
for num_paths_train in "${num_paths_train_values[@]}"; do
  for num_paths_test in "${num_paths_test_values[@]}"; do
      for lr in "${learning_rates[@]}"; do
        for step_size in "${step_sizes[@]}"; do
          for nbest_scale in "${nbest_scales[@]}"; do
            for acoustic_scale in "${acoustic_scales[@]}"; do
              for use_embeddings in "${use_embeddings_values[@]}"; do
                for use_frozen_feats in "${use_frozen_feats_values[@]}"; do
                  for num_exact_log_p in "${num_exact_log_p_values[@]}"; do
                    # Construct the trial ID
                    trial_id="alpha30${trial_counter}"
                    
                    # Construct the command for the current configuration
                    cmd="$base_command $base_args --trial_id=$trial_id --num_paths_train=$num_paths_train --num_paths_test=$num_paths_test --lr_cfm=$lr --step_size=$step_size --nbest_scale=$nbest_scale --ac_scale=$acoustic_scale --use_embeddings=$use_embeddings --use_frozen_feats=$use_frozen_feats --num_exact_log_p=$num_exact_log_p"
                    
                    # Log the command being run
                    echo "Running: $cmd"
                    
                    # Run the command and save the output to a log file
                    log_file="logs/${trial_id}_train${num_paths_train}_test${num_paths_test}_lr${lr}_ss${step_size}_nbest${nbest_scale}_acscale${acoustic_scale}_embed${use_embeddings}_frozen${use_frozen_feats}_logp${num_exact_log_p}.log"
                    mkdir -p logs
                    $cmd > $log_file 2>&1
                    
                    # Check the exit status of the command
                    if [ $? -eq 0 ]; then
                      echo "Trial $trial_id completed successfully."
                    else
                      echo "Trial $trial_id failed. Check log file: $log_file"
                    fi
                    
                    # Increment the trial counter
                    trial_counter=$((trial_counter + 1))
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All hyperparameter trials completed."
