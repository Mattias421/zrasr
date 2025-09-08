WANDB_TAGS=("beta")
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H --lattice_cache=results/lat_cache_beta 
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H2 --lattice_cache=results/lat_cache_beta --lr_cfm=0.01
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H3 --lattice_cache=results/lat_cache_beta --lr_cfm=0.0001
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H4 --lattice_cache=results/lat_cache_beta --lr_cfm=0.1
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H5 --lattice_cache=results/lat_cache_beta --num_exact_logp=1
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H6 --lattice_cache=results/lat_cache_beta --num_exact_logp=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H7 --lattice_cache=results/lat_cache_beta --num_exact_logp=3
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H8 --lattice_cache=results/lat_cache_beta --num_exact_logp=4
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H9 --lattice_cache=results/lat_cache_beta --batchsize=6
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H10 --lattice_cache=results/lat_cache_beta --num_paths_train=2 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H11 --lattice_cache=results/lat_cache_beta --num_paths_train=1 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H12 --lattice_cache=results/lat_cache_beta --num_paths_train=4 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H13 --lattice_cache=results/lat_cache_beta --num_paths_train=16 --num_paths_test=16
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H14 --lattice_cache=results/lat_cache_beta --num_paths_train=4 --num_paths_test=16
uv run train_cfmam.py hparams/cfmam.yaml --trial_id=beta_10H15 --lattice_cache=results/lat_cache_beta --num_paths_train=16 --num_paths_test=32

uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H20 --lattice_cache=results/lat_cache_beta 
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H202 --lattice_cache=results/lat_cache_beta --lr_cfm=0.01
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H203 --lattice_cache=results/lat_cache_beta --lr_cfm=0.0001
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H204 --lattice_cache=results/lat_cache_beta --lr_cfm=0.1
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H205 --lattice_cache=results/lat_cache_beta --num_exact_logp=1
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H206 --lattice_cache=results/lat_cache_beta --num_exact_logp=2
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H207 --lattice_cache=results/lat_cache_beta --num_exact_logp=3
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H208 --lattice_cache=results/lat_cache_beta --num_exact_logp=4
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H209 --lattice_cache=results/lat_cache_beta --batchsize=6
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H2010 --lattice_cache=results/lat_cache_beta --num_paths_train=2 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H2011 --lattice_cache=results/lat_cache_beta --num_paths_train=1 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H2012 --lattice_cache=results/lat_cache_beta --num_paths_train=4 --num_paths_test=2
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H2013 --lattice_cache=results/lat_cache_beta --num_paths_train=16 --num_paths_test=16
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H2014 --lattice_cache=results/lat_cache_beta --num_paths_train=4 --num_paths_test=16
uv run train_cfmam.py hparams/cfmam.yaml --use_embeddings=true --trial_id=beta_10H2015 --lattice_cache=results/lat_cache_beta --num_paths_train=16 --num_paths_test=32
