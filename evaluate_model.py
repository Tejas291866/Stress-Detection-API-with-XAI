# evaluate_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import your custom modules
from data_loader import SWELLDataLoader
from feature_engineering import FeatureEngineer
from dataset import SWELLDataset # Your SWELLDataset class
from model import CNNLSTM # Your CNNLSTM model

# --- Configuration (MUST MATCH your training script's final configuration) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adjust these paths to your actual CSV files
TRAIN_CSV_PATH = 'train.csv' # Needed to fit scaler and label encoder from training data
TEST_CSV_PATH = 'test.csv'

# Model specific parameters - THESE MUST MATCH THE VALUES USED IN YOUR TRAINED MODEL
# (i.e., the TIMESTEPS and FEATURES_PER_STEP your model was trained with)
TIMESTEPS = 17 # Assuming this was the final TIMESTEPS used in train.py
FEATURES_PER_STEP = 2 # Assuming this was the final FEATURES_PER_STEP used in train.py

BATCH_SIZE = 32 # Can be adjusted for evaluation, but typically kept consistent
MODEL_LOAD_PATH = 'best_model.pth' # Path to your saved model weights

# --- Evaluation Function ---
def evaluate_model_full(model, dataloader, criterion, device, label_encoder):
    model.eval() # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    running_loss = 0.0

    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100

    print(f"\n--- Evaluation Results ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Decode labels for better readability in reports
    true_labels_decoded = label_encoder.inverse_transform(all_labels)
    predicted_labels_decoded = label_encoder.inverse_transform(all_predictions)
    
    # Get class names in the order used by LabelEncoder
    class_names = label_encoder.classes_

    print("\n--- Classification Report ---")
    print(classification_report(true_labels_decoded, predicted_labels_decoded, target_names=class_names))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(true_labels_decoded, predicted_labels_decoded, labels=class_names)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return avg_loss, accuracy, all_labels, all_predictions

# --- Main Execution ---
if __name__ == "__main__":
    print("Step 1: Loading and Preprocessing Data (for evaluation)...")
    # Load data for both train and test to ensure scaler and label encoder are fitted consistently
    data_loader = SWELLDataLoader(train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH)
    # Call load_data which handles preprocessing and encoding
    X_train_raw, X_test_raw, y_train_encoded, y_test_encoded = data_loader.load_data()

    # The actual FEATURES_PER_STEP is determined by the data and TIMESTEPS
    # This recalculation ensures consistency, but ensure TIMESTEPS matches training
    if X_train_raw.shape[1] % TIMESTEPS != 0:
        print(f"Warning: TIMESTEPS ({TIMESTEPS}) is not a perfect divisor of total features ({X_train_raw.shape[1]}). "
              "This may indicate a mismatch with training settings.")
    actual_features_per_step = X_train_raw.shape[1] // TIMESTEPS
    if actual_features_per_step != FEATURES_PER_STEP:
        print(f"Warning: FEATURES_PER_STEP in config ({FEATURES_PER_STEP}) does not match "
              f"calculated ({actual_features_per_step}). Using calculated value.")
        FEATURES_PER_STEP = actual_features_per_step


    print("Step 2: Applying Feature Engineering (Reshaping for CNN-LSTM)...")
    feature_engineer = FeatureEngineer(n_components=None) # Use same PCA setting as training
    X_train_processed, X_test_processed = feature_engineer.engineer(
        X_train_raw, X_test_raw, timesteps=TIMESTEPS
    )

    print(f"Shape of X_test_processed after reshaping: {X_test_processed.shape}")
    
    # Get the number of classes from the fitted LabelEncoder
    NUM_CLASSES = len(data_loader.label_encoder.classes_)
    print(f"Number of classes detected: {NUM_CLASSES}")

    print("Step 3: Creating PyTorch DataLoader for Test Set...")
    test_dataset = SWELLDataset(data=X_test_processed, labels=y_test_encoded)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Test Dataset Size: {len(test_dataset)}")
    print(f"Number of batches in Test Loader: {len(test_loader)}")

    print("Step 4: Initializing Model and Loading Weights...")
    model = CNNLSTM(input_channels=FEATURES_PER_STEP, timesteps=TIMESTEPS, num_classes=NUM_CLASSES).to(DEVICE)
    
    # Check if the model file exists before loading
    if os.path.exists(MODEL_LOAD_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
            print(f"Successfully loaded model weights from {MODEL_LOAD_PATH}")
        except Exception as e:
            print(f"Error loading model weights from {MODEL_LOAD_PATH}: {e}")
            exit(1)
    else:
        print(f"Error: Model file '{MODEL_LOAD_PATH}' not found. Please ensure it exists.")
        exit(1)

    criterion = nn.CrossEntropyLoss() # Use the same loss function as training

    print("Step 5: Performing Evaluation...")
    # Pass the label_encoder to the evaluation function for decoding
    test_loss, test_accuracy, all_labels, all_predictions = evaluate_model_full(
        model, test_loader, criterion, DEVICE, data_loader.label_encoder
    )

    print("\nEvaluation complete!")