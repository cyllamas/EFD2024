import numpy as np
import pandas as pd
import os
import yaml
import subprocess
from utils.open_config import load_config

class DataLoader:
    def __init__(self, params, project_root):
        self.params = params
        self.project_root = project_root
        self.dvc_init()
        self.dvc_pull()

    def dvc_init(self):
        """
        Initializes DVC and sets up the remote storage for Google Drive.
        """
        try:
            subprocess.run(['dvc', 'remote', 'add', '-d', 'mygdrive', self.params['dvc']['gdrive_url']], check=True, cwd=self.project_root)
            print("DVC remote directory added successfully.")
            subprocess.run(['dvc', 'remote', 'modify', 'mygdrive', 'gdrive_client_id', self.params['dvc']['gdrive_client_id']], check=True, cwd=self.project_root)
            subprocess.run(['dvc', 'remote', 'modify', 'mygdrive', 'gdrive_client_secret', self.params['dvc']['gdrive_client_secret']], check=True, cwd=self.project_root)
            print("DVC remote modified with client credentials.")

        except subprocess.CalledProcessError as e:
            print(f"Error during DVC initialization or remote setup: {e}")

    def dvc_pull(self):
        """
        Runs 'dvc pull' to ensure required data files are up-to-date.
        """
        try:
            subprocess.run(['dvc', 'pull'], check=True, cwd=self.project_root)
            print("DVC pull completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running 'dvc pull': {e}")

    def load_data(self):
        """
        Loads data from CSV files for 2023 and 2024 Food Drive Data and City of Edmonton Geo data.
        """       
        raw_data_path_2023 = os.path.join(self.project_root, self.params["files"]["food_drive_2023"])
        raw_data_path_2024 = os.path.join(self.project_root, self.params["files"]["food_drive_2024"])
        raw_data_path_geo = os.path.join(self.project_root, self.params["files"]["edmonton_geo"])

        df_efd_2023 = pd.read_csv(raw_data_path_2023)
        df_efd_2024 = pd.read_csv(raw_data_path_2024, encoding='ISO-8859-1')
        df_edm_geo = pd.read_csv(raw_data_path_geo)

        return df_efd_2023, df_efd_2024, df_edm_geo

    def store_data(self, df, file_path):
        """
        Stores the dataframe to CSV and performs DVC push.
        """
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        try:
            subprocess.run(['dvc', 'add', file_path], check=True)
            subprocess.run(['git', 'add', os.path.join(self.project_root, 'data', 'processed.dvc')], check=True)
            subprocess.run(['dvc', 'config', 'core.autostage', 'true'], check=True)
            subprocess.run(['dvc', 'push'], check=True)
            print("Data pushed to DVC successfully!")
        except subprocess.CalledProcessError as e:
            print(f"DVC push failed: {e}") 
            
class DataCleaner:
    def __init__(self, df_efd_2023=None, df_efd_2024=None):
        """
        Initializes the DataCleaner with 2023 and/or 2024 data.
        """
        self.df_efd_2023 = df_efd_2023
        self.df_efd_2024 = df_efd_2024

    def clean_2024_data(self):
        """
        Cleans the 2024 Food Drive Data by removing unnecessary columns.
        """
        df_efd_2024_cleaned = self.df_efd_2024.copy()

        columns_to_drop = [
            'ID', 'Name', 'Start time', 'Completion time', 'Email', 'Email address\xa0',  
            'Did you complete more than 1 route?', 'Additional Routes completed (2 routes)',
            'Additional routes completed (3 routes)', 'Additional routes completed (3 routes)2',
            'Additional routes completed (More than 3 Routes)', 'Additional routes completed (More than 3 Routes)2',
            'Additional routes completed (More than 3 Routes)3'
        ]
        df_efd_2024_cleaned = df_efd_2024_cleaned.drop(columns=columns_to_drop, errors='ignore')
        df_efd_2024_cleaned = df_efd_2024_cleaned.drop_duplicates()
        df_efd_2024_cleaned['How many routes did you complete?'] = df_efd_2024_cleaned['How many routes did you complete?'].fillna(1)
        df_efd_2024_cleaned['Route Number/Name'] = 'Route ' + df_efd_2024_cleaned['Route Number/Name']

        time_spent_mapping = {
            '0 - 30 Minutes': 15,
            '30 - 60 Minutes': 45,
            '1 Hour - 1.5 Hours': 75,
            '2+ Hours': 90
        }

        df_efd_2024_cleaned['Time Spent Collecting Donations'] = df_efd_2024_cleaned['Time Spent Collecting Donations'].map(time_spent_mapping).fillna(0).astype(int)
        return df_efd_2024_cleaned

    def apply_concatenation(self, df_efd_2024_cleaned):
        """
        Applies concatenation of non-null values for the 'Ward' and cleans up other columns.
        """
        def concatenate_non_nulls(row):
            non_nulls = row.dropna().values  
            return ', '.join(non_nulls) if len(non_nulls) > 0 else None
        
        df_efd_2024_cleaned['Ward'] = df_efd_2024_cleaned[['Bonnie Doon Stake', 'Edmonton North Stake', 'Gateway Stake', 
                                                           'Riverbend Stake', 'Sherwood Park Stake', 'YSA Stake']].apply(concatenate_non_nulls, axis=1)
        columns_to_drop = ['Bonnie Doon Stake', 'Edmonton North Stake', 'Gateway Stake', 'Riverbend Stake', 
                           'Sherwood Park Stake', 'YSA Stake']
        df_efd_2024_cleaned = df_efd_2024_cleaned.drop(columns=columns_to_drop, errors='ignore')

        df_efd_2024_cleaned.loc[df_efd_2024_cleaned['Other Drop-off Locations'].notnull(), 'Drop Off Location'] = df_efd_2024_cleaned['Other Drop-off Locations']
        df_efd_2024_cleaned = df_efd_2024_cleaned.drop(columns=['Other Drop-off Locations'])
        df_efd_2024_cleaned['Year'] = 2024
        df_efd_2024_cleaned.columns = df_efd_2024_cleaned.columns.str.strip()

        return df_efd_2024_cleaned

    def rename_columns_2024(self, df_efd_2024_cleaned):
        """
        Renames columns in the 2024 Food Drive Data to standardize the names.
        """
        df_efd_2024_cleaned = df_efd_2024_cleaned.rename(columns={
            'How did you receive the form?': 'Form Method', 'Time Spent Collecting Donations': 'Time Spent', 
            'Route Number/Name': 'Route Information', '# of Adult Volunteers who participated in this route': 'Total Adult Volunteers',
            '# of Youth Volunteers who participated in this route': 'Total Youth Volunteers', '# of Doors in Route': 'Number of Doors',
            '# of Donation Bags Collected': 'Number of Donation Bags', 'How many routes did you complete?': 'Number of Routes',
            'Comments or Feedback': 'Comments'
        })

        return df_efd_2024_cleaned

    def clean_2023_data(self):
        """
        Cleans the 2023 Food Drive Data by renaming columns and adding default values.
        """
        df_efd_2023_cleaned = self.df_efd_2023.copy()
        df_efd_2023_cleaned['Route Information'] = 'Route Unknown'
        df_efd_2023_cleaned['Year'] = 2023
        df_efd_2023_cleaned = df_efd_2023_cleaned.rename(columns={
            'Location': 'Drop Off Location', '# of Adult Volunteers': 'Total Adult Volunteers', 
            '# of Youth Volunteers': 'Total Youth Volunteers', 'Doors in Route': 'Number of Doors', 
            'Donation Bags Collected': 'Number of Donation Bags', 'Routes Completed': 'Number of Routes'
        })
        return df_efd_2023_cleaned

class FeatureEngineer:
    def __init__(self, df_efd_cleaned=None):
        """
        Initializes the FeatureEngineer with the cleaned data.
        """
        self.df_efd_cleaned = df_efd_cleaned

    def feature_engineering(self):
        """
        Performs feature engineering on the cleaned data, adding new features for analysis.
        """
        df_efd_cleaned = self.df_efd_cleaned.copy()

        df_efd_cleaned['Total Volunteers'] = df_efd_cleaned['Total Adult Volunteers'] + df_efd_cleaned['Total Youth Volunteers']
        df_efd_cleaned['Donation Bags per Door'] = df_efd_cleaned['Number of Donation Bags'] / df_efd_cleaned['Number of Doors']
        df_efd_cleaned['Donation Bags per Route'] = df_efd_cleaned['Number of Donation Bags'] / df_efd_cleaned['Number of Routes']

        df_efd_cleaned['Donation Bags per Door'] = df_efd_cleaned['Donation Bags per Door'].replace([np.inf, -np.inf], 0)
        df_efd_cleaned[['Donation Bags per Door', 'Donation Bags per Route']] = df_efd_cleaned[['Donation Bags per Door', 'Donation Bags per Route']].fillna(0)

        return df_efd_cleaned

class DataMerger:
    def __init__(self, df_efd_2024_cleaned=None, df_efd_2023_cleaned=None, df_edm_geo=None):
        """
        Initializes the DataMerger with the cleaned data for 2024, 2023, and geographical data.
        """
        self.df_efd_2024_cleaned = df_efd_2024_cleaned
        self.df_efd_2023_cleaned = df_efd_2023_cleaned
        self.df_edm_geo = df_edm_geo

    def merge_cleaned_data(self):
        """
        Merges the cleaned data from 2023 and 2024 and integrates geographical data for Edmonton neighbourhoods.
        """
        common_columns = [
            'Drop Off Location', 'Stake', 'Route Information', 'Time Spent', 'Total Adult Volunteers', 
            'Total Youth Volunteers', 'Number of Doors', 'Number of Donation Bags', 'Number of Routes', 
            'Ward', 'Year'
        ]
        
        df_efd_cleaned = pd.concat([self.df_efd_2024_cleaned[common_columns], self.df_efd_2023_cleaned[common_columns]], ignore_index=True)
        df_efd_cleaned['Ward'] = df_efd_cleaned['Ward'].str.replace(' Stake', '', regex=False)
        df_efd_cleaned['Ward'] = df_efd_cleaned['Ward'].str.replace(' Ward', '', regex=False)

        df_efd_cleaned = df_efd_cleaned.merge(self.df_edm_geo[['Neighbourhood Name', 'Latitude', 'Longitude']],
                                              left_on='Ward', right_on='Neighbourhood Name', how='left')
        df_efd_cleaned.drop(columns=['Neighbourhood Name'], inplace=True)
        df_efd_cleaned['Ward'] = df_efd_cleaned['Ward'] + ' Ward'
        df_efd_cleaned = df_efd_cleaned[(df_efd_cleaned['Ward'] != 'Ward') & (~df_efd_cleaned['Ward'].isnull())]
        df_efd_cleaned = df_efd_cleaned.drop_duplicates()
        
        categorical_columns = ['Drop Off Location', 'Stake', 'Route Information', 'Ward']
        df_efd_cleaned[categorical_columns] = df_efd_cleaned[categorical_columns].astype('category')

        return df_efd_cleaned

class DataSplitter:
    def __init__(self, df_efd_cleaned=None):
        """
        Initializes the DataSplitter with the cleaned and engineered Food Drive data.
        """
        self.df_efd_cleaned = df_efd_cleaned

    def split_data(self):
        """
        Splits the cleaned data into training and testing datasets.
        """
        df_selected = self.df_efd_cleaned[['Time Spent', 'Number of Doors', 'Number of Donation Bags', 
                                           'Number of Routes', 'Ward', 'Year', 'Total Volunteers', 
                                           'Donation Bags per Door', 'Donation Bags per Route']]

        train_data_2023 = df_selected[df_selected['Year'] == 2023]
        test_data_2024 = df_selected[df_selected['Year'] == 2024]
        
        X_train = train_data_2023[['Time Spent', 'Number of Doors', 'Number of Routes', 'Ward', 'Year',
                                   'Total Volunteers', 'Donation Bags per Door']]
        y_train = train_data_2023['Number of Donation Bags']

        X_test = test_data_2024[['Time Spent', 'Number of Doors', 'Number of Routes', 'Ward', 'Year',
                                 'Total Volunteers', 'Donation Bags per Door']]
        y_test = test_data_2024['Number of Donation Bags']
        
        return X_train, y_train, X_test, y_test
