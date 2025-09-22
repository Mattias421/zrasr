#!/bin/bash

# Layer ranges per GPU
gpu0=(0.001 0.0001)
gpu1=(0.00003)
gpu2=(0.00006 0.00006)

run_on_gpu() {
  gpu_id=$1
  runs=("${!2}")
  for lr in "${runs[@]}"; do
    WANDB_TAGS=lr_sweep2 CUDA_VISIBLE_DEVICES=$gpu_id uv run train_dfm.py hparams/dfm.yaml --number_of_epochs=10 --trial_id=lr2_$lr --lr_adam=$lr
  done
}

# Launch jobs on each GPU
run_on_gpu 0 gpu0[@] &
run_on_gpu 1 gpu1[@] &
run_on_gpu 2 gpu2[@] &

# Wait for all to finish
wait
echo "All training jobs finished."
