import pickle
import pandas as pd
import os
import yaml
from utils.open_config import get_default_params

class ModelPredictor:
    """
    A class to load a trained machine learning model and make predictions on donations based on input features.
    """
    def __init__(self, model_path):
        """
        Initializes the ModelPredictor class and loads a trained model from the specified file path.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file at {model_path} does not exist.")
        
        try:
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
        except Exception as e:
            raise RuntimeError(f"Error loading the model: {e}")
    
    def predict_donations(self, ward, time_spent, num_doors, num_routes, year, total_volunteers, bags_per_door):
        """
        Makes a prediction for the number of donations based on input features.
        """
        input_data = pd.DataFrame({
            'Time Spent': [time_spent],
            'Number of Doors': [num_doors],
            'Number of Routes': [num_routes],
            'Ward': [ward],
            'Year': [year],
            'Total Volunteers': [total_volunteers],
            'Donation Bags per Door': [bags_per_door]
        })
        
        try:
            prediction = self.model.predict(input_data)
            prediction_rounded = round(prediction[0])
            return f"{prediction_rounded}"
        except Exception as e:
            raise RuntimeError(f"Error making predictions: {e}")

if __name__ == "__main__":
    try:
        print('===================== Model Prediction Started! =====================')
        params = get_default_params()
        predictor = ModelPredictor(os.path.join(params.project_root, params.model_poly))

        ward = "Beaumont Ward"
        time_spent = 5.0
        num_doors = 100
        num_routes = 10
        year = 2024
        total_volunteers = 50
        bags_per_door = 3

        prediction = predictor.predict_donations(ward, time_spent, num_doors, num_routes, year, total_volunteers, bags_per_door)
        print(f"🛍️ Predicted Number of Donation Bags: {prediction}") 
        print('===================== Model Prediction Finished! =====================')
    except Exception as e:
        print(f"An error occurred: {e}")
