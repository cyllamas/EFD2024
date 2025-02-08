import os
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "parameters.yml")

def load_config():
    """Loads parameters from a YAML configuration file"""
    with open(CONFIG_PATH, "r") as file:
        params = yaml.safe_load(file)
    return params, PROJECT_ROOT
