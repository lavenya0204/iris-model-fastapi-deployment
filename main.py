from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import List

# 1. Initialize FastAPI App
app = FastAPI(
    title="Iris Species Prediction API",
    description="An API to predict Iris flower species based on the given measurements."
)

# 2. Use the Pydantic BaseModel for Input Data
# This ensures that the input data from the API request is validated
# to match the features your model expects.
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initializing global variables that will hold the trained machine learning model and 
# the list of target/species names
model = None
target_names = None

# Define the path to your models directory
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, 'iris_logistic_regression_model.joblib')
TARGET_NAMES_PATH = os.path.join(MODELS_DIR, 'iris_target_names.joblib')

# 3. Load Model and Target Names on Application Startup
# The @app.on_event("startup") decorator ensures this function runs
# only once when your FastAPI application first starts.
@app.on_event("startup")
async def load_model_and_names():
    global model, target_names
    try:
        model = joblib.load(MODEL_PATH)
        target_names = joblib.load(TARGET_NAMES_PATH)
        print(f"Model loaded successfully from: {MODEL_PATH}")
        print(f"Target names loaded successfully from: {TARGET_NAMES_PATH}")
    except FileNotFoundError:
        print(f"Error: Model or target names file not found. "
              f"Please ensure '{MODEL_PATH}' and '{TARGET_NAMES_PATH}' exist. "
              "Run train_model.py first.")
        # raise RuntimeError
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")


# 4. Define API Endpoints

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Species Prediction API! Go to /docs for API documentation."}

# Prediction endpoint
@app.post("/predict")
def predict_species(features: IrisFeatures):
    """
    Args:
        features (IrisFeatures): An object containing the sepal_length,
                                 sepal_width, petal_length, and petal_width.
    Returns:
        dict: A dictionary containing the predicted species name.
    """
    if model is None or target_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again.")

    try:
        # Convert the Pydantic model features to a list or NumPy array
        input_data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]

        # Make prediction
        prediction_index = model.predict(input_data)[0]
        # Map prediction index to species name
        predicted_species = target_names[prediction_index]
        return {"predicted_species": predicted_species}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")