import os
import json
import yaml
import numpy as np
from datetime import datetime


def create_experiment_dir(name):
    # Create timestamped directory for storing results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{name}_{timestamp}"
    os.makedirs(path, exist_ok=True)
    return path


def save_metrics(metrics, path):
    # Convert numpy types to native Python types for JSON serialization
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            clean_metrics[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v

    with open(f"{path}/metrics.json", "w") as f:
        json.dump(clean_metrics, f, indent=4)


def save_config(cfg, path):
    # Save config to YAML file for reproducibility
    with open(f"{path}/config.yaml", "w") as f:
        yaml.dump(cfg, f)