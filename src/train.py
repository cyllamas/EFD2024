import os
import yaml
import numpy as np
import pandas as pd
import pickle
import subprocess
import mlflow
import mlflow.sklearn
import tempfile
from utils.open_config import get_default_params
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from preprocess import DataLoader, DataCleaner, FeatureEngineer, DataMerger, DataSplitter

class Trainer:
    """
    A class to preprocess data, train a polynomial regression model, 
    perform hyperparameter tuning, evaluate the model, and save the model and configuration.
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the Trainer class with training and testing data.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_model = None
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean'))
                ]), ['Number of Doors', 'Number of Routes', 'Year',
                     'Total Volunteers', 'Donation Bags per Door', 'Time Spent']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Ward'])
            ]
        )

    def preprocess_data(self):
        """
        Preprocesses the training and test data using column transformations (imputation and one-hot encoding).
        """
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        X_test_transformed = self.preprocessor.transform(self.X_test)
        mlflow.log_param("num_features", len(self.X_train.columns))
        mlflow.log_param("cat_features", len(self.preprocessor.transformers_[1][1].categories_[0]))
           
        return X_train_transformed, X_test_transformed
    
    def train_polynomial_regression(self, X_train_transformed, X_test_transformed):
        """
        Trains a polynomial regression model using the transformed data.
        """
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train_transformed)
        X_test_poly = poly.transform(X_test_transformed)
        self.poly_reg = LinearRegression()
        self.poly_reg.fit(X_train_poly, self.y_train)
        return X_train_poly, X_test_poly

    def hyperparameter_tuning(self, X_train_poly):
        """
        Performs hyperparameter tuning on the polynomial regression model using grid search.
        """
        param_grid = {
            'fit_intercept': [True, False],
            'copy_X': [True, False]
        }
        grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
        for i, params in enumerate(grid_search.param_grid.values()):
            grid_search.fit(X_train_poly, self.y_train)
            score = grid_search.best_score_
            mlflow.log_metric("r2_score_hyperparameter_testing", score, step=i)
            mlflow.log_param("best_fit_intercept", grid_search.best_estimator_.fit_intercept)
            mlflow.log_param("best_copy_X", grid_search.best_estimator_.copy_X)

        self.best_model = grid_search.best_estimator_  
        return self.best_model  

    def save_train_config(self, config_path, flag):
        """
        Saves the best model's hyperparameters to a configuration file.
        """
        best_params = self.best_model.get_params()
        
        config_data = {
            'model_specs': {
                'model_type': 'Polynomial Regression',  
                'degree': 2  
            },
            'hyperparameters': {
                'copy_X': best_params.get('copy_X', True),  
                'fit_intercept': best_params.get('fit_intercept', True),  
                'n_jobs': best_params.get('n_jobs', None),  
                'positive': best_params.get('positive', False)  
            }
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        if flag == "On":
            mlflow.log_artifact(config_path)

    def save_model(self, model_path, flag):
        """
        Saves the trained model to a specified file path using pickle.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),    
                    ('scaler', StandardScaler())                   
                ]), ['Number of Doors', 'Number of Routes', 'Year', 
                    'Total Volunteers', 'Donation Bags per Door', 'Time Spent']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Ward'])  
            ]
        )

        polynomial_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  
            ('poly', PolynomialFeatures(degree=2)),  
            ('regressor', LinearRegression())  
        ])

        polynomial_pipeline.fit(self.X_train, self.y_train)

        with open(model_path, 'wb') as f:
            pickle.dump(polynomial_pipeline, f)

        input_data = pd.DataFrame({
            'Time Spent': [5.0],  # List format for single scalar value
            'Number of Doors': [100.0],  # Converted to float
            'Number of Routes': [10.0],  # Converted to float
            'Ward': ["Beaumont Ward"],
            'Year': [2025.0],  # Converted to float
            'Total Volunteers': [50.0],  # Converted to float
            'Donation Bags per Door': [3.0]  # Converted to float
        })

        try:
            if flag == "On":
                mlflow.sklearn.log_model(polynomial_pipeline, "polynomial_regression_model", input_example=input_data)
                mlflow.log_artifact(model_path)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        print('===================== Data Preparation Started! =====================')
        params = get_default_params()

        print('Performing Data Loading Process ...')
        data_loader = DataLoader(params)
        df_efd_2023, df_efd_2024, df_edm_geo = data_loader.load_data()

        print(f"Shape of 2023 Data: {df_efd_2023.shape}")
        print(f"Shape of 2024 Data: {df_efd_2024.shape}")
        print(f"Shape of Edmonton Geo Data: {df_edm_geo.shape}")

        print('Performing Data Cleaning Process ...')
        data_cleaner = DataCleaner(df_efd_2023, df_efd_2024)
        df_efd_2024_cleaned = data_cleaner.clean_2024_data()
        df_efd_2024_cleaned = data_cleaner.apply_concatenation(df_efd_2024_cleaned)
        df_efd_2024_cleaned = data_cleaner.rename_columns_2024(df_efd_2024_cleaned)
        df_efd_2023_cleaned = data_cleaner.clean_2023_data()

        print('Performing Data Merging Process ...')
        data_merger = DataMerger(df_efd_2024_cleaned, df_efd_2023_cleaned, df_edm_geo)
        df_efd_cleaned = data_merger.merge_cleaned_data()

        print('Performing Feature Engineering Process ...')
        feature_engineer = FeatureEngineer(df_efd_cleaned)
        df_efd_cleaned = feature_engineer.feature_engineering()

        print('Performing Export of Cleaned Dataset Process ...')
        data_loader.store_data(df_efd_cleaned, os.path.join(params.project_root, params.cleaned_data))
        print('===================== Data Preparation Completed! =====================')
    except Exception as e:
        print(f"An error occurred during data preparation: {e}") 