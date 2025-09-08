# uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=1 --seed=2222 --trial_id=cfmam
# uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=1 --seed=3333 --trial_id=cfmam
#
# uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=0 --seed=2222 --trial_id=seed_model
# uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=0 --seed=3333 --trial_id=seed_model

# uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=0 --seed=1111 --trial_id=seed_model_bs2 --batch_size=2
uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=1 --seed=1111 --trial_id=cfmam_bs2 --batch_size=2


uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=0 --seed=1111 --trial_id=seed_model_bs2_N16 --batch_size=2 --num_paths=16
uv run train_cfmam.py hparams/cfmam.yaml --number_of_epochs_seed=10 --number_of_epochs_reward=1 --seed=1111 --trial_id=cfmam_bs2_N16 --batch_size=2 --num_paths=16
