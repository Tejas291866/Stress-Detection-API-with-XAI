# api.py
from typing import List, Dict, Tuple, Optional # Add Optional here
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
import os
from typing import List, Dict, Tuple

# Import your custom modules
from data_loader import SWELLDataLoader
from feature_engineering import FeatureEngineer
from model import CNNLSTM
from xai_utils import explain_single_instance_shap # Your new XAI utility

# --- Configuration (MUST MATCH your training script's final configuration) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_CSV_PATH = 'train.csv' # Needed for scaler, label encoder, and SHAP background data
MODEL_LOAD_PATH = 'best_model.pth' 

TIMESTEPS = 17 
FEATURES_PER_STEP = 2 
TOTAL_FLATTENED_FEATURES = TIMESTEPS * FEATURES_PER_STEP # 17 * 2 = 34

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SWELL Stress Detection API with XAI",
    description="API for predicting stress levels from HRV features and explaining predictions using SHAP.",
    version="1.0.0"
)

# --- Global Variables for Model and Data Processing ---
# These will be loaded once when the API starts
model: Optional[CNNLSTM] = None
data_loader: Optional[SWELLDataLoader] = None
feature_engineer: Optional[FeatureEngineer] = None
background_data_raw: Optional[np.ndarray] = None
num_classes: int = 0
original_feature_names: List[str] = []

# --- Pydantic Model for Input Data ---
# This defines the expected structure of your input JSON
class HRVInput(BaseModel):
    # Use Field to add descriptions and examples for FastAPI's OpenAPI docs
    features: List[float] = Field(
        ..., # '...' means this field is required
        min_items=TOTAL_FLATTENED_FEATURES,
        max_items=TOTAL_FLATTENED_FEATURES,
        description=f"A list of {TOTAL_FLATTENED_FEATURES} HRV feature values, representing a flattened time series. "
                    f"Expected order corresponds to {TIMESTEPS} timesteps with {FEATURES_PER_STEP} features per step.",
        example=[0.5] * TOTAL_FLATTENED_FEATURES # Example with dummy values
    )

# --- Startup Event: Load Model and Data Processors ---
@app.on_event("startup")
async def load_resources():
    global model, data_loader, feature_engineer, background_data_raw, num_classes, original_feature_names

    print(f"Loading resources on {DEVICE}...")

    # Initialize data loader and load a small portion of training data
    # This is crucial for fitting the scaler and label encoder, and for SHAP background
    try:
        # Use a dummy test_path as it's not used for fitting, but required by SWELLDataLoader
        data_loader = SWELLDataLoader(train_path=TRAIN_CSV_PATH, test_path=TRAIN_CSV_PATH)
        
        # Load data; we only need X_train_raw and original_feature_names for setup
        # y_train_encoded is needed to fit the label encoder
        X_train_raw_full, _, y_train_encoded_full, _, original_feature_names = data_loader.load_data()
        
        # Store a subset of training data as background for SHAP
        # It's good practice to use a small, representative sample to reduce memory/computation
        background_data_raw = X_train_raw_full # Use full raw training data as background for now
        
        num_classes = len(data_loader.label_encoder.classes_)
        print(f"Number of classes loaded: {num_classes}")
        print(f"Original feature names loaded: {original_feature_names[:5]}...") # Print first few

    except Exception as e:
        raise RuntimeError(f"Failed to load and preprocess data for API startup: {e}")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer(n_components=None) # Use same PCA setting as training

    # Initialize and load the model
    try:
        model = CNNLSTM(input_channels=FEATURES_PER_STEP, timesteps=TIMESTEPS, num_classes=num_classes).to(DEVICE)
        if os.path.exists(MODEL_LOAD_PATH):
            model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
            model.eval() # Set to evaluation mode
            print(f"Model loaded from {MODEL_LOAD_PATH}")
        else:
            raise FileNotFoundError(f"Model file '{MODEL_LOAD_PATH}' not found. Cannot start API.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model for API startup: {e}")

    print("Resources loaded successfully. API is ready.")


# --- API Endpoint ---
@app.post("/predict_and_explain")
async def predict_and_explain(input_data: HRVInput):
    """
    Receives a single flattened HRV feature vector, predicts the stress class,
    and provides SHAP-based explanations for the prediction.
    """
    if model is None or data_loader is None or feature_engineer is None or background_data_raw is None:
        raise HTTPException(status_code=503, detail="Model or data resources not loaded yet. Please try again.")

    try:
        # Convert input list to numpy array
        instance_raw_features = np.array(input_data.features, dtype=np.float64).reshape(1, -1)

        # Call the XAI utility function
        predicted_class_name, explanations = explain_single_instance_shap(
            model=model,
            instance_raw_features=instance_raw_features,
            data_loader=data_loader,
            feature_engineer=feature_engineer,
            timesteps=TIMESTEPS,
            features_per_step=FEATURES_PER_STEP,
            device=DEVICE,
            background_data_raw=background_data_raw,
            top_n_features=10 # You can make this configurable via query parameter if needed
        )

        return {
            "predicted_class": predicted_class_name,
            "explanations": explanations,
            "message": "Prediction and explanation generated successfully."
        }

    except Exception as e:
        print(f"Error during prediction or explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Root Endpoint (Optional) ---
@app.get("/")
async def root():
    return {"message": "SWELL Stress Detection API is running. Use /predict_and_explain POST endpoint."}