import pickle
import pandas as pd
import os
import yaml
from flask import Flask, jsonify, request
from src.utils.open_config import get_default_params

app = Flask(__name__)

def load_model(model_path):
    """Loads a trained machine learning model from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            print(f"Model loaded successfully from {model_path}")
            return model
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

def predict_donations(model, ward, time_spent, num_doors, num_routes, year, total_volunteers, bags_per_door):
    """Predicts the number of donation bags based on input features."""
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
        prediction = model.predict(input_data)
        return str(round(prediction[0]))
    except Exception as e:
        raise RuntimeError(f"Error making predictions: {e}")

params = get_default_params()
poly_model = load_model(os.path.join(params.project_root, params.model_poly))
dt_model = load_model(os.path.join(params.project_root, params.model_dt))

@app.route('/efd2024_home', methods=['GET'])
def home():
    app_info = {
        "name": "Edmonton Food Drive: Total Donation Bags Predictor",
        "description": "This API uses 2 models to predict the number of donation bags for a year.",
        "version": "v1.0",
        "author": {
            "name1": "Kendrick Moreno",
            "name2": "Roe Alincastre",
            "name3": "Catrina Llamas"
        },
        "endpoints": {
            "/efd2024_home": "Home Page",
            "/health_status": "Check the health status of the API",
            "/v1/predict": "This model version for regression prediction is a polynomial regression model",
            "/v2/predict": "This model version for regression prediction is a decision tree model",
        },
        "input_format": {
            "ward": "The name of the ward where the food drive is held (e.g., 'Beaumont Ward').",
            "time_spent": "The total time in hours volunteers spent on the drive (e.g., 5.0).",
            "num_doors": "The number of doors involved in the food drive (e.g., 100).",
            "num_routes": "The number of routes taken by volunteers (e.g., 10).",
            "year": "The year of the drive (e.g., 2024).",
            "total_volunteers": "The total number of volunteers involved (e.g., 50).",
            "bags_per_door": "The average number of donation bags per door (e.g., 3)."
        },
        "example_request": {
            "ward": "Beaumont Ward",
            "time_spent": 5.0,
            "num_doors": 100,
            "num_routes": 10,
            "year": 2024,
            "total_volunteers": 50,
            "bags_per_door": 3
        },
        "example_output": {
            "Predicted Number of Donation Bags": "281",
            "Status": "Success",
            "model": "Polynomial Regression"
        }
    }
    return jsonify(app_info)

@app.route('/health_status', methods=['GET'])
def health_check():
    """Provide health message of the API"""
    app_info = {
        "status": "OK",
        "models_loaded": {
            "poly_model": isinstance(poly_model, object),
            "dt_model": isinstance(dt_model, object)
        },
        "message": "EFD API is up and ready to receive request."
    }

    return jsonify(app_info)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict using the polynomial regression model."""
    data = request.get_json()
    try:
        prediction = predict_donations(
            poly_model,
            ward=data["ward"],
            time_spent=data["time_spent"],
            num_doors=data["num_doors"],
            num_routes=data["num_routes"],
            year=data["year"],
            total_volunteers=data["total_volunteers"],
            bags_per_door=data["bags_per_door"]
        )

        response = {
            "status": "Success", 
            "model": "Polynomial Regression", 
            "predicted number of donation bags": prediction
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """Predict using the decision tree model."""
    if not request.is_json:
        return jsonify({"error": "Request must contain JSON format data"}), 400
    
    data = request.get_json()
    try:
        prediction = predict_donations(
            dt_model,
            ward=data["ward"],
            time_spent=data["time_spent"],
            num_doors=data["num_doors"],
            num_routes=data["num_routes"],
            year=data["year"],
            total_volunteers=data["total_volunteers"],
            bags_per_door=data["bags_per_door"]
        )

        response = {
            "status": "Success", 
            "model": "Decision Tree", 
            "predicted number of donation bags": prediction
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=9000, debug=True)




