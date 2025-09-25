#!/bin/bash

# Layer ranges per GPU
gpu0=(2000 2500)
gpu1=(3000)
gpu2=(1000 1500)

run_on_gpu() {
  gpu_id=$1
  runs=("${!2}")
  for lr in "${runs[@]}"; do
    WANDB_TAGS=t_wrm_swp CUDA_VISIBLE_DEVICES=$gpu_id uv run train_dfm.py hparams/dfm.yaml --number_of_epochs=20 --trial_id=t_wrm_$lr --t_warmup=$lr
  done
}

# Launch jobs on each GPU
run_on_gpu 0 gpu0[@] &
run_on_gpu 1 gpu1[@] &
run_on_gpu 2 gpu2[@] &

# Wait for all to finish
wait
echo "All training jobs finished."
