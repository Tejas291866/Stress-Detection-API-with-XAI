# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional
import pandas as pd

# Assuming SWELLDataLoader and FeatureEngineer are correctly implemented
# and accessible (e.g., in the same directory or installed as a package)
from data_loader import SWELLDataLoader
from feature_engineering import FeatureEngineer


class SWELLDataset(Dataset):
    """
    A custom PyTorch Dataset for the SWELL data, integrating
    data loading and feature engineering.
    """
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 transform: Optional[object] = None):
        """
        Args:
            data (np.ndarray): The feature data (X_train or X_val/test) after
                               preprocessing and feature engineering.
                               Expected shape: (num_samples, timesteps, features_per_step).
            labels (np.ndarray): The corresponding labels (y_train or y_val/test).
                                 Expected shape: (num_samples,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if data.shape[0] != labels.shape[0]:
            raise ValueError("Number of samples in data and labels must match.")

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # Labels for CrossEntropyLoss should be long
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample (features and label) by index.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature tensor
                                              and the label tensor for the given index.
        """
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# --- Example Usage (for testing the dataset class) ---
if __name__ == "__main__":
    # You would typically have 'train.csv' and 'test.csv' in a 'data' directory
    # For this example, let's create dummy CSVs to make it runnable.
    # In a real scenario, replace these with your actual file paths.

    # Dummy CSVs (create these files in your working directory for testing)
    # train.csv: 2 samples, 60 features (e.g., 6 timesteps * 10 features_per_step), 1 label
    dummy_train_data = np.random.rand(2, 60) # 2 samples, 60 features each
    dummy_train_labels = np.array(["no stress", "time pressure"])
    dummy_train_df = pd.DataFrame(dummy_train_data)
    dummy_train_df['condition'] = dummy_train_labels
    dummy_train_df.to_csv('dummy_train.csv', index=False)

    # test.csv: 1 sample, 60 features, 1 label
    dummy_test_data = np.random.rand(1, 60)
    dummy_test_labels = np.array(["no stress"])
    dummy_test_df = pd.DataFrame(dummy_test_data)
    dummy_test_df['condition'] = dummy_test_labels
    dummy_test_df.to_csv('dummy_test.csv', index=False)

    print("Dummy CSV files created: dummy_train.csv, dummy_test.csv")

    # --- Data Loading and Preprocessing ---
    data_loader = SWELLDataLoader(train_path='dummy_train.csv', test_path='dummy_test.csv')
    X_train_raw, X_test_raw, y_train_encoded, y_test_encoded = data_loader.load_data()

    # --- Feature Engineering ---
    # Assuming your model expects 10 features per timestep and 6 timesteps
    # This means your raw data rows should have 60 features in total.
    TIMESTEPS = 6
    FEATURES_PER_STEP = 10 # This should match your CNNLSTM's input_channels
    if X_train_raw.shape[1] % TIMESTEPS != 0:
        print(f"Warning: Raw feature count {X_train_raw.shape[1]} not divisible by timesteps {TIMESTEPS}.")
        print("Adjust TIMESTEPS or your data's feature count.")
    
    feature_engineer = FeatureEngineer(n_components=None) # No PCA for this example
    
    # Engineer features and reshape for sequence model
    X_train_processed, X_test_processed = feature_engineer.engineer(
        X_train_raw, X_test_raw, timesteps=TIMESTEPS
    )

    print(f"\nShape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of y_train_encoded: {y_train_encoded.shape}")
    print(f"Shape of X_test_processed: {X_test_processed.shape}")
    print(f"Shape of y_test_encoded: {y_test_encoded.shape}")
    print(f"Number of classes: {len(data_loader.label_encoder.classes_)}")
    print(f"Class mapping: {list(data_loader.label_encoder.classes_)} -> {list(range(len(data_loader.label_encoder.classes_)))}")


    # --- Create Dataset Instances ---
    train_dataset = SWELLDataset(data=X_train_processed, labels=y_train_encoded)
    test_dataset = SWELLDataset(data=X_test_processed, labels=y_test_encoded)

    print(f"\nTrain Dataset Size: {len(train_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Example of retrieving a sample
    sample_features, sample_label = train_dataset[0]
    print(f"\nSample 0 features shape: {sample_features.shape}")
    print(f"Sample 0 label: {sample_label.item()}") # .item() to get scalar value

    # Clean up dummy files
    import os
    os.remove('dummy_train.csv')
    os.remove('dummy_test.csv')
    print("\nDummy CSV files cleaned up.")