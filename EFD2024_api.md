# Edmonton Food Drive API - Documentation

![Edmonton Food Drive Logo](https://github.com/kmoreno013/MyProjects/blob/main/logo_efd.png?raw=true)

## Overview

The **Edmonton Food Drive API** is designed to provide predictions for the number of donation bags required during the food drive based on several input features. The API exposes two different machine learning models for predictions: a **Polynomial Regression** model and a **Decision Tree** model. This API allows users to retrieve predictions by providing the necessary features through two endpoints.

- **API Version**: v1.0
- **Author**: Kendrick Moreno, Roe Alincastre, Catrina Llamas

## Base URL
http://127.0.0.1:9000


## Endpoints

### 1. **Home Endpoint** (`/efd2024_home`)
- **Method**: `GET`
- **Description**: Provides information about the API, including the available endpoints and expected input/output formats.

#### Response Example:

```
{
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
    "/v1/predict": "Polynomial Regression model for prediction",
    "/v2/predict": "Decision Tree model for prediction"
  },
  "input_format": {
    "ward": "Ward where the food drive is held (e.g., 'Beaumont Ward')",
    "time_spent": "Total time in hours spent on the drive (e.g., 5.0)",
    "num_doors": "Number of doors involved (e.g., 100)",
    "num_routes": "Number of routes taken by volunteers (e.g., 10)",
    "year": "Year of the drive (e.g., 2024)",
    "total_volunteers": "Total number of volunteers (e.g., 50)",
    "bags_per_door": "Average number of donation bags per door (e.g., 3)"
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
```

### 2. **Health Status Endpoint** (/health_status)
- **Method:** `GET`
- **Description:** Provides a status message to check if the API is operational.

#### Response Example:

```
{
  "status": "OK",
  "models_loaded": {
    "poly_model": true,
    "dt_model": true
  },
  "message": "EFD API is up and ready to receive request."
}

```
### 3. **Prediction Endpoint v1** (/v1/predict)
**Method:** `POST`
**Description:** Predicts the number of donation bags using the Polynomial Regression model.

#### Request Body (JSON):
```
{
  "ward": "Beaumont Ward",
  "time_spent": 5.0,
  "num_doors": 100,
  "num_routes": 10,
  "year": 2024,
  "total_volunteers": 50,
  "bags_per_door": 3
}
```

#### Response Example:
```
{
  "status": "Success",
  "model": "Polynomial Regression",
  "predicted number of donation bags": "281"
}
```
### 4. *Prediction Endpoint v2* (/v2/predict)
- **Method:** `POST`
- **Description:** Predicts the number of donation bags using the Decision Tree model.

#### Request Body (JSON):
```
{
  "ward": "Beaumont Ward",
  "time_spent": 5.0,
  "num_doors": 100,
  "num_routes": 10,
  "year": 2024,
  "total_volunteers": 50,
  "bags_per_door": 3
}
```

#### Response Example:
```
{
  "status": "Success",
  "model": "Decision Tree",
  "predicted number of donation bags": "281"
}
```
## Dependencies
To run the Edmonton Food Drive API, you will need the following dependencies installed in your environment:
```
pip install pandas pyyaml Flask
```

## Usage Notes
* **Content-Type:** All requests to the API should be made with the Content-Type: application/json header.
* **Model Files:** The models used for predictions are loaded from the models/ directory. The expected models are:
  * polynomial_regression_model.pkl (Polynomial Regression model)
  * decision_tree_model.pkl (Decision Tree model)
* Predictions: The models predict the number of donation bags based on the following input features:
  * **ward:** The name of the ward where the food drive is held.
  * **time_spent:** The total time in hours volunteers spent on the drive.
  * **num_doors:** The number of doors involved in the food drive.
  * **num_routes:** The number of routes taken by volunteers.
  * **year:** The year of the drive.
  * **total_volunteers:** The total number of volunteers involved.
  * **bags_per_door:** The average number of donation bags per door.

## Example Request
* Home Endpoint: `GET http://127.0.0.1:9000/efd2024_home`
* Health Status: `GET http://127.0.0.1:9000/health_status`
* Prediction (Polynomial Regression): `POST http://127.0.0.1:9000/v1/predict -H "Content-Type: application/json" -d @configs/request.json`
* Prediction (Decision Tree): `POST http://127.0.0.1:9000/v2/predict -H "Content-Type: application/json" -d @configs/request.json`
