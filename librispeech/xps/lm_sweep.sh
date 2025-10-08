#!/bin/bash

# Define the base YAML configuration file
BASE_HPARAMS_FILE="hparams/lmdfm.yaml"

# --- Grid Search Parameters ---
# Define the values for learning rate (lr_adam)
LR_ADAM_VALUES=(10 1 0.1)

# Define the values for n_warmup_steps
N_WARMUP_STEPS_VALUES=(100000 250000 500000)

# Define the values for d_model
D_MODEL_VALUES=(512 768 1024)

# Define the values for transformer_dropout
TRANSFORMER_DROPOUT_VALUES=(0.05 0.1 0.2)
# --- End Grid Search Parameters ---

# Output directory for experiment results (for organization, though not directly used by --trial_id)
OUTPUT_ROOT="grid_search_direct_args"
mkdir -p "$OUTPUT_ROOT" # Still good practice to have a top-level dir if you store other things

# GRID PARAMS
NUM_EPOCH=10
WANDB_TAGS=lm_sweep

echo "Starting grid search..."

# Loop over learning rate values
for LR in "${LR_ADAM_VALUES[@]}"; do
    # Loop over n_warmup_steps values
    for NWS in "${N_WARMUP_STEPS_VALUES[@]}"; do
        # Loop over d_model values
        for DM in "${D_MODEL_VALUES[@]}"; do
            # Loop over transformer_dropout values
            for TD in "${TRANSFORMER_DROPOUT_VALUES[@]}"; do

                # Create a unique trial ID for this combination of hyperparameters
                # This will define the output folder for the results in your SpeechBrain setup
                TRIAL_ID="lmdfm_lr${LR}_nws${NWS}_dm${DM}_td${TD}"

                echo "Running experiment with TRIAL_ID: $TRIAL_ID"
                echo "  lr_adam: $LR"
                echo "  n_warmup_steps: $NWS"
                echo "  d_model: $DM"
                echo "  transformer_dropout: $TD"

                # The actual command to run your training script, now passing HPs directly
                # Make sure 'uv' and 'train_lmdfm.py' are accessible in your PATH or specify full paths
                uv run python train_lmdfm.py "$BASE_HPARAMS_FILE" \
                   --trial_id="$TRIAL_ID" \
                   --lr_adam="$LR" \
                   --n_warmup_steps="$NWS" \
                   --d_model="$DM" \
                   --transformer_dropout="$TD" \
                   --number_of_epochs=$NUM_EPOCH \
                   --train_only=true

                # Optional: Add a separator for clarity in logs
                echo "----------------------------------------------------"
            done
        done
    done
done

echo "Grid search completed."
