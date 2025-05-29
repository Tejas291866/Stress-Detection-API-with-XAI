import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Keep random_split for splitting processed train data
import numpy as np
from tqdm import tqdm # For nice progress bars
import os
# Import your custom modules
from data_loader import SWELLDataLoader
from feature_engineering import FeatureEngineer
from dataset import SWELLDataset # Your newly created SWELLDataset class
from model import CNNLSTM # Your CNNLSTM model

# --- Configuration and Hyperparameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adjust these paths to your actual CSV files
TRAIN_CSV_PATH = 'train.csv'
TEST_CSV_PATH = 'test.csv' # For final testing, usually not used in main training loop validation

# Model specific parameters (must match your feature engineering)
# You need to determine these based on your raw data's structure
# and how you want to window it.
# For example, if each row in your CSV is 60 features, and you want 6 timesteps,
# then FEATURES_PER_STEP would be 10.
# Example: 60 total features / 6 timesteps = 10 features_per_step
TIMESTEPS = 17 # This should be the length of your time window
FEATURES_PER_STEP = 2 # This should be the number of features per timestamp in your window

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
PATIENCE = 3 # For early stopping: number of epochs to wait for improvement
MIN_DELTA = 0.001 # Minimum change to be considered an improvement for early stopping
MODEL_SAVE_PATH = 'best_model.pth' # Or 'models/best_model.pth' if you prefer a subdirectory

# Get the directory name from the path. If it's just a filename, dirname will be empty.
save_dir = os.path.dirname(MODEL_SAVE_PATH)

# If a directory is specified, ensure it exists.
if save_dir and not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
        print(f"Created directory for saving models: {save_dir}")
    except OSError as e:
        print(f"Error: Could not create directory '{save_dir}'. Please check permissions. Error: {e}")
        exit(1) # Exit if directory cannot be created

# Check if we can write a dummy file to the target location
try:
    with open(MODEL_SAVE_PATH, 'w') as f:
        f.write('temp_check')
    os.remove(MODEL_SAVE_PATH) # Clean up the dummy file
    print(f"Successfully verified write access to '{MODEL_SAVE_PATH}'.")
except IOError as e:
    print(f"Error: No write access to '{MODEL_SAVE_PATH}'. Please check directory permissions or path. Error: {e}")
    exit(1) # Exit if we can't write to the path
# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
# import random
# random.seed(SEED) # Uncomment if you use Python's random module


# --- Data Loading, Preprocessing, and Feature Engineering ---

print("Step 1: Loading and Preprocessing Data...")
data_loader = SWELLDataLoader(train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH)
X_train_raw, X_test_raw, y_train_encoded, y_test_encoded, _ = data_loader.load_data()

# Determine the actual number of features from your processed data
# After StandardScaler, X_train_raw.shape[1] gives the total number of features in a flattened row.
if X_train_raw.shape[1] % TIMESTEPS != 0:
    raise ValueError(
        f"The total number of features ({X_train_raw.shape[1]}) "
        f"is not divisible by the specified TIMESTEPS ({TIMESTEPS}). "
        f"Please adjust TIMESTEPS or verify your data structure."
    )
FEATURES_PER_STEP = X_train_raw.shape[1] // TIMESTEPS
print(f"Detected {X_train_raw.shape[1]} total features per sample.")
print(f"With {TIMESTEPS} timesteps, each timestep will have {FEATURES_PER_STEP} features.")


print("Step 2: Applying Feature Engineering (Reshaping for CNN-LSTM)...")
# PCA is optional. Set n_components=None if not using PCA.
feature_engineer = FeatureEngineer(n_components=None) # Set n_components to an int if you want to use PCA

# Engineer features and reshape for sequence model
X_train_processed, X_test_processed = feature_engineer.engineer(
    X_train_raw, X_test_raw, timesteps=TIMESTEPS
)

print(f"Shape of X_train_processed after reshaping: {X_train_processed.shape}")
print(f"Shape of X_test_processed after reshaping: {X_test_processed.shape}")
print(f"Number of encoded classes: {len(data_loader.label_encoder.classes_)}")

# Dynamically get number of classes from the label encoder
NUM_CLASSES = len(data_loader.label_encoder.classes_)


# --- Create PyTorch Datasets and DataLoaders ---
print("Step 3: Creating PyTorch Datasets and DataLoaders...")

# The SWELLDataset takes pre-processed numpy arrays
full_train_dataset = SWELLDataset(data=X_train_processed, labels=y_train_encoded)

# Split the training data into training and validation sets
TRAIN_RATIO = 0.8 # 80% for training, 20% for validation
train_size = int(TRAIN_RATIO * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create DataLoaders
# num_workers > 0 for faster data loading, but consumes more RAM/CPU.
# pin_memory=True moves data to CUDA pinned memory for faster GPU transfer.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Validation Dataset Size: {len(val_dataset)}")
print(f"Number of batches in Train Loader: {len(train_loader)}")
print(f"Number of batches in Val Loader: {len(val_loader)}")


# --- Model, Loss, Optimizer, and Scheduler ---
print("Step 4: Initializing Model, Loss, Optimizer, and Scheduler...")
# Ensure CNNLSTM's input_channels matches FEATURES_PER_STEP and timesteps matches TIMESTEPS
# Pass NUM_CLASSES to your model if its output layer needs to be dynamic
model = CNNLSTM(input_channels=FEATURES_PER_STEP, timesteps=TIMESTEPS, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler: Reduces LR when validation loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# You can print the model summary here if you like, now that the input_size is dynamic
# from torchsummary import summary
# summary(model, input_size=(TIMESTEPS, FEATURES_PER_STEP)) # (seq_len, num_features)


# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # Zero the gradients

        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        running_loss += loss.item() # Accumulate loss
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader) # Average loss per batch
    epoch_accuracy = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_accuracy

def validate_model(model, dataloader, criterion, device):
    model.eval() # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculations during validation
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_loss = running_loss / len(dataloader) # Average loss per batch
    val_accuracy = (correct_predictions / total_samples) * 100
    return val_loss, val_accuracy


# --- Main Training Loop ---
if __name__ == "__main__":
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\nStep 5: Training started on {DEVICE}")

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate_model(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Step the learning rate scheduler
        scheduler.step(val_loss)

        # Early Stopping and Model Saving
        # Check for significant improvement (val_loss decreased by at least MIN_DELTA)
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"  Validation loss improved. Saving model checkpoint...")
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not significantly improve. Patience: {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping triggered after {epoch+1} epochs due to no improvement.")
                break

    print("\nTraining complete!")
    # Optionally, you can load the best model here for final evaluation on the test set
    # model.load_state_dict(torch.load('best_model.pth'))
    # print("Loaded best model for final evaluation (if needed).")

    # --- Final Evaluation on Test Set (Optional but Recommended) ---
    print("\nStep 6: Final evaluation on the test set (using best model)...")
    # First, create the test dataset and dataloader
    test_dataset = SWELLDataset(data=X_test_processed, labels=y_test_encoded)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Load the best model's weights
    if 'best_model.pth' in os.listdir('.'): # Check if the file exists
        model.load_state_dict(torch.load('best_model.pth'))
        test_loss, test_acc = validate_model(model, test_loader, criterion, DEVICE)
        print(f"Final Test Loss (Best Model): {test_loss:.4f} | Final Test Acc (Best Model): {test_acc:.2f}%")
    else:
        print("No 'best_model.pth' found to evaluate on test set.")