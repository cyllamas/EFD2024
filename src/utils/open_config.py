import os
import yaml
import argparse


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

def get_default_params():
    """Parses command-line arguments with defaults loaded from a YAML file."""
    config, _ = load_config()
    train_config, _ = load_train_config()

    parser = argparse.ArgumentParser(description="Command-line arguments for EFD 2024 Application")
    parser.add_argument('--project_root', type=str, default=PROJECT_ROOT, help='Application Path')
    parser.add_argument('--food_drive_2023', type=str, default=config["files"]["food_drive_2023"], help='Path to 2023 food drive data')
    parser.add_argument('--food_drive_2024', type=str, default=config["files"]["food_drive_2024"], help='Path to 2024 food drive data')
    parser.add_argument('--edmonton_geo', type=str, default=config["files"]["edmonton_geo"], help='Path to Edmonton geo data')
    parser.add_argument('--train_config', type=str, default=config["files"]["train_config"], help='Path to the training config')
    parser.add_argument('--model_poly', type=str, default=config["files"]["model_poly"], help='Path to save the first trained model')
    parser.add_argument('--model_dt', type=str, default=config["files"]["model_dt"], help='Path to save the second trained model')
    parser.add_argument('--cleaned_data', type=str, default=config["files"]["cleaned_data"], help='Path to cleaned dataset')
    parser.add_argument('--evaluation_results', type=str, default=config["results"]["evaluation"], help='Path to evaluation results image')
    parser.add_argument('--gdrive_url', type=str, default=config["dvc"]["gdrive_url"], help='Google Drive URL for DVC storage')
    parser.add_argument('--gdrive_client_id', type=str, default=config["dvc"]["gdrive_client_id"], help='Google Drive client ID')
    parser.add_argument('--gdrive_client_secret', type=str, default=config["dvc"]["gdrive_client_secret"], help='Google Drive client secret')
    parser.add_argument('--mlflow_experiment_name', type=str, default=config["setup"]["mlflow_experiment_name"], help='MLflow experiment name')
    parser.add_argument('--mlflow_experiment_url', type=str, default=config["setup"]["mlflow_experiment_url"], help='MLflow experiment URL')
    parser.add_argument('--allow_data_file_track', type=str, default=config["setup"]["allow_data_file_track"], help='Enable DVC')
    parser.add_argument('--allow_ml_model_track', type=str, default=config["setup"]["allow_ml_model_track"], help='Enable MLFlow')
    parser.add_argument('--train_second_model', type=str, default=config["setup"]["train_second_model"], help='Enable Building Decision Tree')
    parser.add_argument('--copy_X', type=str, default=train_config["hyperparameters_poly"]["copy_X"], help='Whether to copy X for polynomial regression')
    parser.add_argument('--fit_intercept', type=str, default=train_config["hyperparameters_poly"]["fit_intercept"], help='Whether to fit intercept for polynomial regression')
    parser.add_argument('--n_jobs', type=int, default=train_config["hyperparameters_poly"]["n_jobs"], help='Number of jobs to run in parallel')
    parser.add_argument('--positive', type=str, default=train_config["hyperparameters_poly"]["positive"], help='Whether to constrain coefficients to be positive')
    parser.add_argument('--degree', type=int, default=train_config["model_specs_poly"]["degree"], help='Degree of the polynomial regression model')
    parser.add_argument('--model_type_poly', type=str, default=train_config["model_specs_poly"]["model_type"], choices=['Polynomial Regression'], help='Type of model to be used')
    parser.add_argument('--max_depth', type=str, default=train_config["hyperparameters_dt"]["max_depth"], help='Number of max_depth for Decision Trees')
    parser.add_argument('--min_samples_leaf', type=str, default=train_config["hyperparameters_dt"]["min_samples_leaf"], help='Number of min_samples_leaf for Decision Trees')
    parser.add_argument('--min_samples_split', type=int, default=train_config["hyperparameters_dt"]["min_samples_split"], help='Number of min_samples_split for Decision Trees')
    parser.add_argument('--random_state', type=str, default=train_config["hyperparameters_dt"]["random_state"], help='Number of random_state for Decision Trees')
    parser.add_argument('--model_type_dt', type=str, default=train_config["model_specs_dt"]["model_type"], choices=['Decision Tree Regressor'], help='Type of model to be used')

    parameters = parser.parse_args()

    return parameters
