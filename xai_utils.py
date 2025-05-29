# xai_utils.py

import torch
import torch.nn as nn
import numpy as np
import shap
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os

# Import your custom modules
# Make sure model.py, data_loader.py, and feature_engineering.py are in the same directory
from model import CNNLSTM
from data_loader import SWELLDataLoader 
from feature_engineering import FeatureEngineer # Import FeatureEngineer

# --- Helper function to wrap PyTorch model for SHAP ---
class ModelWrapper(nn.Module):
    """
    A wrapper around the PyTorch model to make it compatible with SHAP's KernelExplainer.
    KernelExplainer expects a function/model that takes a NumPy array and returns a NumPy array.
    """
    def __init__(self, model: CNNLSTM, timesteps: int, features_per_step: int, device: torch.device):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.features_per_step = features_per_step
        self.device = device
        self.model.eval() # Ensure model is in evaluation mode

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for SHAP.
        Args:
            x (np.ndarray): Input data from SHAP, typically (N_samples, TOTAL_FLATTENED_FEATURES).
                            N_samples can be 1 for single explanation or more for background.
        Returns:
            np.ndarray: Model probabilities (softmax output) as a NumPy array.
        """
        # Convert NumPy array to Torch tensor and move to device
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        # Reshape the flattened input from (N_samples, TOTAL_FEATURES) to (N_samples, TIMESTEPS, FEATURES_PER_STEP)
        total_features_flat = x_tensor.shape[-1] 
        
        if total_features_flat != (self.timesteps * self.features_per_step):
             raise ValueError(
                 f"Input features {total_features_flat} from SHAP does not match "
                 f"expected {self.timesteps * self.features_per_step} based on TIMESTEPS and FEATURES_PER_STEP."
             )

        x_reshaped = x_tensor.view(-1, self.timesteps, self.features_per_step)

        with torch.no_grad(): # Ensure no gradients are computed
            outputs = self.model(x_reshaped)
            # probabilities = torch.softmax(outputs, dim=1) # <--- ORIGINAL LINE
            # return probabilities.cpu().numpy() # <--- ORIGINAL LINE

            # FIX: Return raw logits instead of probabilities.
            # KernelExplainer often handles raw logits better for multi-class.
            return outputs.cpu().numpy() 


# --- XAI Explanation Generation Functions ---

def get_feature_value_description(
    feature_value: float, 
    background_mean: float, 
    background_std: float, 
    threshold_std: float = 1.0
) -> str:
    """
    Describes if a feature value is 'high', 'low', or 'average' relative to background data.
    Args:
        feature_value (float): The actual value of the feature in the instance being explained.
        background_mean (float): Mean of this feature in the background dataset.
        background_std (float): Standard deviation of this feature in the background dataset.
        threshold_std (float): Number of standard deviations to consider a value 'high' or 'low'.
    Returns:
        str: 'high', 'low', or 'average'.
    """
    if background_std == 0: # Avoid division by zero if std is 0 (constant feature)
        return "typical"
    
    if feature_value > background_mean + threshold_std * background_std:
        return "high"
    elif feature_value < background_mean - threshold_std * background_std:
        return "low"
    else:
        return "typical"

def generate_explanation_sentences(
    shap_values_for_class: np.ndarray, 
    instance_values_flat: np.ndarray, 
    feature_names_flat: List[str], 
    background_data_flat: np.ndarray,
    predicted_class_name: str,
    top_n: int = 10
) -> List[Dict]:
    """
    Generates human-readable explanation sentences for the top N contributing features.
    Args:
        shap_values_for_class (np.ndarray): SHAP values for the predicted class for one instance.
                                            Shape: (TOTAL_FLATTENED_FEATURES,)
        instance_values_flat (np.ndarray): The flattened feature values of the instance being explained.
                                           Shape: (TOTAL_FLATTENED_FEATURES,)
        feature_names_flat (List[str]): List of flattened feature names (e.g., ['T0_MEAN_RR', 'T0_MEDIAN_RR', ...])
        background_data_flat (np.ndarray): The flattened background dataset used by SHAP.
                                           Shape: (N_background_samples, TOTAL_FLATTENED_FEATURES)
        predicted_class_name (str): The decoded name of the predicted class.
        top_n (int): Number of top features to include in the explanation.
    Returns:
        List[Dict]: A list of dictionaries, each describing a feature's contribution.
                    Example: [{'feature': 'T5_HR', 'value': 75.2, 'description': 'high', 
                               'contribution': 'positive', 'sentence': '...'}]
    """
    if len(shap_values_for_class) != len(instance_values_flat) or \
       len(shap_values_for_class) != len(feature_names_flat):
        raise ValueError("Mismatched lengths for SHAP values, instance values, or feature names.")

    # Calculate mean and std for each feature from the background data
    background_means = np.mean(background_data_flat, axis=0)
    background_stds = np.std(background_data_flat, axis=0)

    explanations = []
    
    # Create a DataFrame for easy sorting and processing
    shap_df = pd.DataFrame({
        'feature_name_flat': feature_names_flat,
        'shap_value': shap_values_for_class,
        'instance_value': instance_values_flat,
        'background_mean': background_means,
        'background_std': background_stds
    })
    
    # Sort by absolute SHAP value to get most impactful features
    shap_df['abs_shap_value'] = shap_df['shap_value'].abs()
    shap_df_sorted = shap_df.sort_values(by='abs_shap_value', ascending=False).head(top_n)

    for _, row in shap_df_sorted.iterrows():
        feature_name_flat = row['feature_name_flat']
        shap_value = row['shap_value']
        instance_value = row['instance_value']
        bg_mean = row['background_mean']
        bg_std = row['background_std']

        value_description = get_feature_value_description(instance_value, bg_mean, bg_std)
        contribution_direction = "positively" if shap_value > 0 else "negatively"
        
        # Determine if the feature pushed towards or away from the predicted class
        # Positive SHAP value means it pushed towards the predicted class
        # Negative SHAP value means it pushed away from the predicted class
        
        sentence = (
            f"{feature_name_flat} was {value_description} (value: {instance_value:.4f}) "
            f"and contributed {contribution_direction} to the '{predicted_class_name}' prediction."
        )
        
        explanations.append({
            'feature': feature_name_flat,
            'value': instance_value,
            'shap_value': shap_value,
            'value_description': value_description, # e.g., 'high', 'low', 'typical'
            'contribution_direction': contribution_direction, # e.g., 'positively', 'negatively'
            'sentence': sentence
        })
    
    return explanations

def explain_single_instance_shap(
    model: CNNLSTM, 
    instance_raw_features: np.ndarray, # Unscaled, unreshaped raw features for the instance
    data_loader: SWELLDataLoader, # To access scaler, encoder, and original feature names
    feature_engineer: FeatureEngineer, # To access reshape logic
    timesteps: int, 
    features_per_step: int,
    device: torch.device,
    background_data_raw: np.ndarray, # Raw background data (e.g., X_train_raw)
    original_feature_names: List[str], # <--- ADDED THIS PARAMETER TO THE FUNCTION SIGNATURE
    top_n_features: int = 10
) -> Tuple[str, List[Dict]]:
    """
    Performs end-to-end SHAP explanation for a single raw input instance.
    Args:
        model (CNNLSTM): The trained PyTorch model.
        instance_raw_features (np.ndarray): A single sample's raw features (1D array or 2D with 1 row).
                                            Shape: (TOTAL_RAW_FEATURES,) or (1, TOTAL_RAW_FEATURES)
        data_loader (SWELLDataLoader): An initialized SWELLDataLoader instance (with scaler/encoder fitted).
        feature_engineer (FeatureEngineer): An initialized FeatureEngineer instance.
        timesteps (int): Number of timesteps for model input.
        features_per_step (int): Number of features per timestep for model input.
        device (torch.device): The device (CPU/GPU) to run the model on.
        background_data_raw (np.ndarray): Raw feature data from the training set for SHAP background.
                                          Shape: (N_samples, TOTAL_RAW_FEATURES)
        original_feature_names (List[str]): List of original feature column names from the raw data.
        top_n_features (int): Number of top features to explain.
    Returns:
        Tuple[str, List[Dict]]: Predicted class name and a list of explanation dictionaries.
    """
    if instance_raw_features.ndim == 1:
        instance_raw_features = instance_raw_features.reshape(1, -1) # Ensure 2D (1, N_features)

    # 1. Preprocess the instance using the fitted scaler from data_loader
    instance_scaled = data_loader.preprocess(instance_raw_features, fit=False)

    # 2. Reshape the instance for the model input
    instance_processed = feature_engineer.reshape_for_sequence(instance_scaled, timesteps=timesteps)
    instance_flat = instance_processed.reshape(1, -1) # Flatten for SHAP KernelExplainer

    # 3. Get background data for SHAP (preprocessed and flattened)
    background_scaled = data_loader.preprocess(background_data_raw, fit=False)
    background_processed = feature_engineer.reshape_for_sequence(background_scaled, timesteps=timesteps)
    background_flat = background_processed.reshape(background_processed.shape[0], -1)
    
    # Sample a smaller background for faster SHAP computation if background_flat is large
    if background_flat.shape[0] > 100: # Adjust this threshold as needed
        background_for_shap = shap.utils.sample(background_flat, 100, random_state=42) # Added random_state for reproducibility
    else:
        background_for_shap = background_flat

    # 4. Create flattened feature names for SHAP plots and explanations
    # This logic correctly maps the flattened index back to original feature names and timesteps
    feature_names_flat = []
    for t_idx in range(timesteps):
        for f_idx_in_step in range(features_per_step):
            # The original feature name for this position in the flattened array
            # is simply its index in the original_feature_names list.
            original_idx = t_idx * features_per_step + f_idx_in_step
            if original_idx < len(original_feature_names): # Safety check
                feature_names_flat.append(f"T{t_idx}_{original_feature_names[original_idx]}")
            else:
                # Fallback if original_feature_names is somehow shorter than expected
                feature_names_flat.append(f"T{t_idx}_UNKNOWN_F{f_idx_in_step}")

    # 5. Make prediction
    wrapped_model = ModelWrapper(model, timesteps, features_per_step, device)
    model_prediction_probs = wrapped_model(instance_flat)[0] # Get probabilities for the single instance
    predicted_class_encoded = np.argmax(model_prediction_probs)
    predicted_class_name = data_loader.label_encoder.inverse_transform([predicted_class_encoded])[0]

    # 6. Calculate SHAP values
    explainer = shap.KernelExplainer(wrapped_model, background_for_shap)
    shap_values = explainer.shap_values(instance_flat, nsamples='auto') 

    # --- CRITICAL DEBUGGING STEP ---
    print(f"\nDEBUG: After shap.KernelExplainer.shap_values() call:")
    print(f"DEBUG: Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"DEBUG: Length of shap_values list: {len(shap_values)}")
        for i, arr in enumerate(shap_values):
            print(f"DEBUG: Shape of shap_values[{i}]: {arr.shape}")
            # Print first few elements to see actual values
            print(f"DEBUG: First 5 values of shap_values[{i}]: {arr.flatten()[:5]}") 
    else:
        print(f"DEBUG: shap_values is a numpy array with shape: {shap_values.shape}")
        print(f"DEBUG: First 5 values of shap_values: {shap_values.flatten()[:5]}")
    print(f"DEBUG: Value of predicted_class_encoded: {predicted_class_encoded}")
    print("--- END CRITICAL DEBUGGING STEP ---\n")

   
    # FIX: Correctly index shap_values for shape (1, num_features, num_classes)
    shap_values_for_predicted_class = shap_values[0, :, predicted_class_encoded]

    # 7. Generate explanations
    explanations = generate_explanation_sentences(
        shap_values_for_predicted_class, 
        instance_flat[0], # Pass the single instance's flattened values
        feature_names_flat, 
        background_for_shap, # Use the sampled background data for mean/std calculation
        predicted_class_name,
        top_n=top_n_features
    )

    return predicted_class_name, explanations

# --- Example Usage (for testing xai_utils.py) ---
if __name__ == "__main__":
    # --- Configuration (MUST MATCH your training script) ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_CSV_PATH = 'dummy_train.csv' # Use dummy files for testing
    TEST_CSV_PATH = 'dummy_test.csv' # Use dummy files for testing
    TIMESTEPS = 17 
    FEATURES_PER_STEP = 2 
    MODEL_LOAD_PATH = 'best_model.pth' 

    # --- Create Dummy CSVs for testing (if not already present) ---
    # In a real scenario, replace these with your actual file paths.
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"Creating dummy {TRAIN_CSV_PATH} and {TEST_CSV_PATH} for xai_utils test...")
        num_raw_features = TIMESTEPS * FEATURES_PER_STEP # 17 * 2 = 34
        # Create dummy feature names matching the expected order
        dummy_feature_names = [f"FEATURE_{i}" for i in range(num_raw_features)]
        
        dummy_train_data = np.random.rand(100, num_raw_features) # 100 samples
        dummy_train_labels = np.random.choice(["no stress", "time pressure", "interruption"], 100)
        dummy_train_df = pd.DataFrame(dummy_train_data, columns=dummy_feature_names) # Pass column names
        dummy_train_df['datasetId'] = np.arange(100) + 1
        dummy_train_df['condition'] = dummy_train_labels
        dummy_train_df.to_csv(TRAIN_CSV_PATH, index=False)

        dummy_test_data = np.random.rand(10, num_raw_features) # 10 samples
        dummy_test_labels = np.random.choice(["no stress", "time pressure", "interruption"], 10)
        dummy_test_df = pd.DataFrame(dummy_test_data, columns=dummy_feature_names) # Pass column names
        dummy_test_df['datasetId'] = np.arange(10) + 101
        dummy_test_df['condition'] = dummy_test_labels
        dummy_test_df.to_csv(TEST_CSV_PATH, index=False)
        print("Dummy CSV files created.")
    
    # --- Data Loading and Preprocessing ---
    print("Loading data and preparing for XAI...")
    data_loader = SWELLDataLoader(train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH)
    
    # Correctly unpack all 5 values returned by load_data()
    X_train_raw, X_test_raw, y_train_encoded, y_test_encoded, original_feature_names_from_loader = data_loader.load_data()
    

    _ = data_loader.preprocess(X_train_raw, fit=True)
    
    # Verify the actual number of features from the loaded data
    if X_train_raw.shape[1] != (TIMESTEPS * FEATURES_PER_STEP):
        print(f"Warning: Configured TIMESTEPS ({TIMESTEPS}) and FEATURES_PER_STEP ({FEATURES_PER_STEP}) "
              f"do not match actual raw features ({X_train_raw.shape[1]}). Adjusting FEATURES_PER_STEP.")
        FEATURES_PER_STEP = X_train_raw.shape[1] // TIMESTEPS
        if X_train_raw.shape[1] % TIMESTEPS != 0:
            raise ValueError("Raw feature count not divisible by TIMESTEPS. Cannot proceed.")

    feature_engineer = FeatureEngineer(n_components=None) # Same PCA setting as training
    NUM_CLASSES = len(data_loader.label_encoder.classes_)

    # --- Load Model ---
    print("Loading model...")
    model = CNNLSTM(input_channels=FEATURES_PER_STEP, timesteps=TIMESTEPS, num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_LOAD_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE, weights_only=True))
            model.eval()
            print(f"Model loaded from {MODEL_LOAD_PATH}")
        except Exception as e:
            print(f"Error loading model weights from {MODEL_LOAD_PATH}. SHAP will use untrained model: {e}")
            # If model loading fails, proceed with untrained model for XAI test,
            # but in a real scenario, you'd exit.
    else:
        print(f"Warning: Model file '{MODEL_LOAD_PATH}' not found. SHAP will use an untrained model.")

    # --- Select an instance to explain ---
    sample_idx_to_explain = 0 # Explain the first sample in the test set
    if sample_idx_to_explain >= X_test_raw.shape[0]:
        print(f"Sample index {sample_idx_to_explain} out of bounds. Using index 0.")
        sample_idx_to_explain = 0
    
    instance_raw_features = X_test_raw[sample_idx_to_explain]
    true_label_decoded = data_loader.label_encoder.inverse_transform([y_test_encoded[sample_idx_to_explain]])[0]
    print(f"\nExplaining test sample index {sample_idx_to_explain} (True Label: {true_label_decoded})...")

    # --- Generate Explanation ---
    predicted_class, explanations = explain_single_instance_shap(
        model=model,
        instance_raw_features=instance_raw_features,
        data_loader=data_loader,
        feature_engineer=feature_engineer,
        timesteps=TIMESTEPS,
        features_per_step=FEATURES_PER_STEP,
        device=DEVICE,
        background_data_raw=X_train_raw, # Use full raw training data as background
        original_feature_names=original_feature_names_from_loader, # <--- PASS THE CORRECT VARIABLE HERE
        top_n_features=10 # Explain top 10 features
    )

    print(f"\n--- Prediction and Explanation for Sample {sample_idx_to_explain} ---")
    print(f"Predicted Class: {predicted_class}")
    print("\nDetailed Explanations:")
    for exp in explanations:
        print(f"- {exp['sentence']}")

    # Clean up dummy files
    if os.path.exists(TRAIN_CSV_PATH):
        os.remove(TRAIN_CSV_PATH)
    if os.path.exists(TEST_CSV_PATH):
        os.remove(TEST_CSV_PATH)
    print("\nDummy CSV files cleaned up.")