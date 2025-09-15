#!/bin/bash

# Layer ranges per GPU
gpu0_layers=(2 3 4 5)
gpu1_layers=(6 7 8 9)
gpu2_layers=(10 11 12)

run_on_gpu() {
  gpu_id=$1
  layers=("${!2}")
  for layer in "${layers[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_id uv run train_dfm.py hparams/dfm.yaml --trial_id=codebook_$layer --codebook_layer=$layer 
  done
}

# Launch jobs on each GPU
run_on_gpu 0 gpu0_layers[@] &
run_on_gpu 1 gpu1_layers[@] &
run_on_gpu 2 gpu2_layers[@] &

# Wait for all to finish
wait
echo "All training jobs finished."
