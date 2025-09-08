WANDB_TAGS=("beta")
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta --lattice_cache=results/lat_cache_beta 
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta2 --lattice_cache=results/lat_cache_beta --lr_cfm=0.01
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta3 --lattice_cache=results/lat_cache_beta --lr_cfm=0.0001
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta4 --lattice_cache=results/lat_cache_beta --lr_cfm=0.1
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta5 --lattice_cache=results/lat_cache_beta --num_exact_logp=1
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta6 --lattice_cache=results/lat_cache_beta --num_exact_logp=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta7 --lattice_cache=results/lat_cache_beta --num_exact_logp=3
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta8 --lattice_cache=results/lat_cache_beta --num_exact_logp=4
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta9 --lattice_cache=results/lat_cache_beta --batchsize=6
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta10 --lattice_cache=results/lat_cache_beta --num_paths_train=2 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta11 --lattice_cache=results/lat_cache_beta --num_paths_train=1 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta12 --lattice_cache=results/lat_cache_beta --num_paths_train=4 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta13 --lattice_cache=results/lat_cache_beta --num_paths_train=16 --num_paths_test=16
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta14 --lattice_cache=results/lat_cache_beta --num_paths_train=4 --num_paths_test=16
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta15 --lattice_cache=results/lat_cache_beta --num_paths_train=16 --num_paths_test=32

