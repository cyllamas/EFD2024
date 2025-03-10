import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import random
import string
import time
from utils.open_config import get_default_params
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from train import Trainer
from preprocess import DataLoader, DataCleaner, FeatureEngineer, DataMerger, DataSplitter

class Evaluator:
    """
    A class for evaluating a machine learning model's performance using various metrics and visualizations.
    """
    def __init__(self, model, X_train, X_test, y_train, y_test, model_name, preprocessor=None):
        """
        Initializes the Evaluator class.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def compute_metrics(self, y_true, y_pred, X):
        """
        Computes various evaluation metrics: MSE, RMSE, MAE, R2, and Adjusted R2.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1)
        return mse, rmse, mae, r2, adjusted_r2

    def print_metrics(self, flag):
        """
        Prints the evaluation metrics (MAE, MSE, RMSE, R², Adjusted R²) for both training and test sets.
        """
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_metrics = self.compute_metrics(self.y_train, y_train_pred, self.X_train)
        test_metrics = self.compute_metrics(self.y_test, y_test_pred, self.X_test)

        print(f"\n📊 Training Metrics ({self.model_name}):")
        print(f"   - MAE  : {train_metrics[2]:.4f}")
        print(f"   - MSE  : {train_metrics[0]:.4f}")
        print(f"   - RMSE : {train_metrics[1]:.4f}")
        print(f"   - R² Score: {train_metrics[3]:.4f}")
        print(f"   - Adjusted R²: {train_metrics[4]:.4f}")

        print(f"\n📊 Test Metrics ({self.model_name}):")
        print(f"   - MAE  : {test_metrics[2]:.4f}")
        print(f"   - MSE  : {test_metrics[0]:.4f}")
        print(f"   - RMSE : {test_metrics[1]:.4f}")
        print(f"   - R² Score: {test_metrics[3]:.4f}")
        print(f"   - Adjusted R²: {test_metrics[4]:.4f}")

        if flag == "On":
            mlflow.log_param("Model", self.model_name)
            mlflow.log_metric("Train MAE", train_metrics[2])
            mlflow.log_metric("Train MSE", train_metrics[0])
            mlflow.log_metric("Train RMSE", train_metrics[1])
            mlflow.log_metric("Train R²", train_metrics[3])
            mlflow.log_metric("Train Adjusted R²", train_metrics[4])

            mlflow.log_metric("Test MAE", test_metrics[2])
            mlflow.log_metric("Test MSE", test_metrics[0])
            mlflow.log_metric("Test RMSE", test_metrics[1])
            mlflow.log_metric("Test R²", test_metrics[3])
            mlflow.log_metric("Test Adjusted R²", test_metrics[4])

    def plot_evaluation(self, path, flag):
        """
        Plots several evaluation graphs, including:
        """
        y_test_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_test_pred

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        sns.residplot(x=y_test_pred, y=residuals, lowess=True, color='blue', ax=axs[0, 0])
        axs[0, 0].set_title('Residual Plot')
        axs[0, 0].set_xlabel('Predicted Values')
        axs[0, 0].set_ylabel('Residuals')

        axs[0, 1].scatter(self.y_test, y_test_pred, alpha=0.6)
        axs[0, 1].plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], '--', color='red')
        axs[0, 1].set_title('Actual vs Predicted Values')
        axs[0, 1].set_xlabel('Actual Values')
        axs[0, 1].set_ylabel('Predicted Values')

        sns.histplot(residuals, kde=True, bins=30, color='purple', ax=axs[1, 0])
        axs[1, 0].set_title('Residual Distribution')
        axs[1, 0].set_xlabel('Residuals')

        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        axs[1, 1].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        axs[1, 1].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        axs[1, 1].set_title('Learning Curve')
        axs[1, 1].set_xlabel('Training Size')
        axs[1, 1].set_ylabel('Mean Squared Error')
        axs[1, 1].legend(loc='best')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        if flag == "On":
            mlflow.log_artifact(path)

if __name__ == "__main__":
    try:
        print('===================== Model Development and Evaluation Started! =====================')
        params = get_default_params()
        df_efd_cleaned = pd.read_csv(os.path.join(params.project_root, params.cleaned_data))

        print('Performing Data Splitting Process ...')
        data_splitter = DataSplitter(df_efd_cleaned)
        X_train, y_train, X_test, y_test = data_splitter.split_data()

        print('Performing Feature Encoding and Standardization Process ...')
        trainer = Trainer(X_train, y_train, X_test, y_test)
        X_train_transformed, X_test_transformed = trainer.preprocess_data()

        if params.allow_ml_model_track == "On":
            if mlflow.active_run():
                mlflow.end_run()

            mlflow.sklearn.autolog(disable=True) # Manual input features due to error on training and testing df format after data transformation 
            run = mlflow.active_run()
            if run:
                print("Active run_id: {}".format(run.info.run_id))
                mlflow.end_run()

            exp_name = params.mlflow_experiment_name
            exp_url = params.mlflow_experiment_url
            mlflow.set_tracking_uri(exp_url) 
            mlflow.set_experiment(exp_name)
            experiment = mlflow.get_experiment_by_name(exp_name)

            if experiment is None:
                print("Experiment not found, creating a new one...")
                experiment = mlflow.create_experiment(exp_name)

            with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                print("Started new run with ID: {}".format(run.info.run_id))
                mlflow.log_params({
                    "model": params.model_type_poly,
                    "degree": params.degree,
                    "copy_X": params.copy_X,
                    "fit_intercept": params.fit_intercept,
                    "n_jobs": params.n_jobs,
                    "positive": params.positive
                })

                print('Performing Model Training Process ...')
                X_train_poly, X_test_poly = trainer.train_polynomial_regression(X_train_transformed, X_test_transformed)

                print('Performing Hyperparameter Tuning Process ...')
                best_model = trainer.hyperparameter_tuning_poly(X_train_poly)

                print('Performing Model Evaluation Process ...')
                evaluator = Evaluator(
                    model=best_model,
                    X_train=X_train_poly,
                    X_test=X_test_poly,
                    y_train=y_train,
                    y_test=y_test,
                    model_name="Polynomial Regression"
                )
                evaluator.print_metrics(params.allow_ml_model_track)
                print('Performing Model Evaluation Plot Process ...')
                evaluator.plot_evaluation(os.path.join(params.project_root, params.evaluation_results), params.allow_ml_model_track)

                print('Performing Save Config Process ...')
                trainer.save_train_config_poly(os.path.join(params.project_root, params.train_config), params.allow_ml_model_track)
                print('Performing Model Download Process ...')
                trainer.save_model_poly(os.path.join(params.project_root, params.model_poly), params.allow_ml_model_track)
                print(f'✅ Model saved: {os.path.join(params.project_root, params.model_poly)}')

            if params.train_second_model == 'On':
                if mlflow.active_run():
                    mlflow.end_run()

                mlflow.sklearn.autolog(disable=True) 
                run = mlflow.active_run()
                if run:
                    print("Active run_id: {}".format(run.info.run_id))
                    mlflow.end_run()

                exp_name = params.mlflow_experiment_name
                exp_url = params.mlflow_experiment_url
                mlflow.set_tracking_uri(exp_url) 
                mlflow.set_experiment(exp_name)
                experiment = mlflow.get_experiment_by_name(exp_name)

                if experiment is None:
                    print("Experiment not found, creating a new one...")
                    experiment = mlflow.create_experiment(exp_name)

                with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                    print("Started new run with ID: {}".format(run.info.run_id))
                    mlflow.log_params({
                        "model": params.model_type_dt,
                        "min_samples_leaf": params.min_samples_leaf,
                        "min_samples_leaf": params.min_samples_leaf,
                        "min_samples_split": params.min_samples_split,
                        "random_state": params.random_state
                    })

                    print('Performing Second Model Training Process ...')
                    trainer.train_decision_tree(X_train_transformed, X_test_transformed)

                    print('Performing Hyperparameter Tuning Process ...')
                    best_model = trainer.hyperparameter_tuning_dt(X_train_transformed)

                    print('Performing Model Evaluation Process ...')
                    evaluator_dt = Evaluator(
                        model=best_model,
                        X_train=X_train_transformed,
                        X_test=X_test_transformed,
                        y_train=y_train,
                        y_test=y_test,
                        model_name="Decision Trees"
                    )
                    evaluator_dt.print_metrics(params.allow_ml_model_track)
                    print('Performing Model Evaluation Plot Process ...')
                    evaluator_dt.plot_evaluation(os.path.join(params.project_root, params.evaluation_results), params.allow_ml_model_track)

                    print('Performing Save Config Process ...')
                    trainer.save_train_config_dt(os.path.join(params.project_root, params.train_config), params.allow_ml_model_track)
                    print('Performing Model Download Process ...')
                    trainer.save_model_dt(os.path.join(params.project_root, params.model_dt), params.allow_ml_model_track, params.max_depth, params.min_samples_leaf, params.min_samples_split, params.random_state)
                    print(f'✅ Second Model saved: {os.path.join(params.project_root, params.model_dt)}')
        else:
            print("MLflow tracking is disabled (allow_ml_model_track = 'Off'). Proceeding without MLflow tracking.")
            print('Performing Model Training Process ...')
            X_train_poly, X_test_poly = trainer.train_polynomial_regression(X_train_transformed, X_test_transformed)

            print('Performing Hyperparameter Tuning Process ...')
            best_model = trainer.hyperparameter_tuning_poly(X_train_poly)

            print('Performing Model Evaluation Process ...')
            evaluator = Evaluator(
                model=best_model,
                X_train=X_train_poly,
                X_test=X_test_poly,
                y_train=y_train,
                y_test=y_test,
                model_name="Polynomial Regression"
            )
            evaluator.print_metrics(params.allow_ml_model_track)
            print('Performing Model Evaluation Plot Process ...')
            evaluator.plot_evaluation(os.path.join(params.project_root, params.evaluation_results), params.allow_ml_model_track)

            print('Performing Save Config Process ...')
            trainer.save_train_config_poly(os.path.join(params.project_root, params.train_config), params.allow_ml_model_track)
            print('Performing Model Download Process ...')
            trainer.save_model_poly(os.path.join(params.project_root, params.model_poly), params.allow_ml_model_track)
            print(f'✅ Model saved: {os.path.join(params.project_root, params.model_poly)}')

            if params.train_second_model == 'On':
                print('Performing Second Model Training Process ...')
                trainer.train_decision_tree(X_train_transformed, X_test_transformed)

                print('Performing Hyperparameter Tuning Process ...')
                best_model = trainer.hyperparameter_tuning_dt(X_train_transformed)

                print('Performing Model Evaluation Process ...')
                evaluator_dt = Evaluator(
                    model=best_model,
                    X_train=X_train_transformed,
                    X_test=X_test_transformed,
                    y_train=y_train,
                    y_test=y_test,
                    model_name="Decision Trees"
                )
                evaluator_dt.print_metrics(params.allow_ml_model_track)
                print('Performing Model Evaluation Plot Process ...')
                evaluator_dt.plot_evaluation(os.path.join(params.project_root, params.evaluation_results), params.allow_ml_model_track)

                print('Performing Save Config Process ...')
                trainer.save_train_config_dt(os.path.join(params.project_root, params.train_config), params.allow_ml_model_track)
                print('Performing Model Download Process ...')
                trainer.save_model_dt(os.path.join(params.project_root, params.model_dt), params.allow_ml_model_track, params.max_depth, params.min_samples_leaf, params.min_samples_split, params.random_state)
                print(f'✅ Second Model saved: {os.path.join(params.project_root, params.model_dt)}')

        print('===================== Model Development and Evaluation completed! =====================')
    except Exception as e:
        print(f"An error occurred during model Development and evaluation: {e}")
