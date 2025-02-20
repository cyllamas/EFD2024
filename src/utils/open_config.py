import os
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "parameters.yml")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "train_config.yaml")

def load_config():
    """Loads parameters from a YAML configuration file"""
    with open(CONFIG_PATH, "r") as file:
        params = yaml.safe_load(file)
    return params, PROJECT_ROOT

def load_train_config():
    """Loads parameters from a YAML model configuration file"""
    with open(MODEL_CONFIG_PATH, "r") as file:
        params = yaml.safe_load(file)
    return params, PROJECT_ROOT