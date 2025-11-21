import os
import sys
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from run_lstm_q import run_with_config

# find one LSTM file
data_dir = "dataset"
files = [f for f in os.listdir(data_dir) if "LSTM_data_" in f and f.endswith(".pt")]
if not files:
    print("No LSTM files found in dataset folder; aborting dry-run")
    sys.exit(1)
file_name = files[0]
print("Using file:", file_name)

# tiny hyperparameter grid for a fast dry-run
test_config = {
    "dataset_path": data_dir,
    "file_pattern": file_name,
    "file_extension": ".pt",
    "freq": "quarter",
    "param_grid": {
        "hidden_dim": [16],
        "num_layers": [1],
        "dropout_rate": [0.0],
        "lr": [0.001],
        "batch_size": [8],
        "num_epochs": [1],
        "weight": [20],
        "weight_decay": [0.0],
    },
    "k_folds": 2,
    "checkpoint_dir": "checkpoints_lstm/",
    "test_year": {"13-19": 2019, "default": 2018},
}

# Run on CPU to keep memory/compute low
run_with_config(test_config, seed=1, device=torch.device("cpu"))
