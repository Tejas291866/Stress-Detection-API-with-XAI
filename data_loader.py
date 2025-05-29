# data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Union
import os

class SWELLDataLoader:
    def __init__(self, train_path: str, test_path: str, target_column: str = 'condition', id_column: str = 'datasetId'):
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.id_column = id_column
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_column_names: List[str] = [] # To store names of actual features used

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Loads and prepares the training and testing datasets.
        Fits scaler and label encoder on training data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
            - X_train_raw: Unscaled training features.
            - X_test_raw: Unscaled testing features.
            - y_train_encoded: Encoded training labels.
            - y_test_encoded: Encoded testing labels.
            - original_feature_names: List of column names corresponding to the features.
        """
        # Load datasets
        try:
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Ensure '{self.train_path}' and '{self.test_path}' exist. Error: {e}")

        # Separate features (X) and target (y)
        # Identify feature columns by excluding target and ID columns
        all_columns = train_df.columns.tolist()
        non_feature_columns = [self.target_column, self.id_column]
        
        # Determine original feature names based on columns present in dataframes
        # This is where we correctly identify the feature columns from the loaded data
        self.feature_column_names = [col for col in all_columns if col not in non_feature_columns]

        X_train_df = train_df[self.feature_column_names]
        y_train_df = train_df[self.target_column]
        
        X_test_df = test_df[self.feature_column_names]
        y_test_df = test_df[self.target_column]

        # Debugging: Print loaded columns and head for verification
        print("\n--- Debugging: Columns loaded from train.csv ---")
        print(train_df.columns.tolist())
        print("First 5 rows (head):\n", train_df.head())
        print("-------------------------------------------------\n")

        print("--- Debugging: Columns loaded from test.csv ---")
        print(test_df.columns.tolist())
        print("First 5 rows (head):\n", test_df.head())
        print("-------------------------------------------------\n")

        # Store raw numpy arrays before scaling/encoding (for XAI background data)
        X_train_raw = X_train_df.values
        X_test_raw = X_test_df.values

        # Fit and transform labels on training data, then transform test data
        y_train_encoded = self.label_encoder.fit_transform(y_train_df)
        y_test_encoded = self.label_encoder.transform(y_test_df)

        print(f"Data loaded. X_train_raw shape: {X_train_raw.shape}, y_train_encoded shape: {y_train_encoded.shape}")
        print(f"Number of original features: {len(self.feature_column_names)}")
        print(f"Classes: {self.label_encoder.classes_}")

        # Return X_train_raw and X_test_raw (unscaled) for consistent background data use in XAI
        # and also return the feature names
        # FIX: Now explicitly return the original_feature_names
        return X_train_raw, X_test_raw, y_train_encoded, y_test_encoded, self.feature_column_names

    def preprocess(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Applies StandardScaler to the data.
        
        Args:
            data (np.ndarray): The feature data to preprocess.
            fit (bool): If True, the scaler is fitted on the data, otherwise just transformed.
                        Should be True for training data, False for test/inference data.
        Returns:
            np.ndarray: The scaled feature data.
        """
        if fit:
            return self.scaler.fit_transform(data)
        else:
            if not hasattr(self.scaler, 'scale_'): # Check if scaler has been fitted
                raise RuntimeError("Scaler has not been fitted. Call preprocess with fit=True on training data first.")
            return self.scaler.transform(data)

# Example Usage (for testing data_loader.py independently)
if __name__ == "__main__":
    # Create dummy CSV files for testing if they don't exist
    TRAIN_CSV_PATH = 'dummy_train.csv'
    TEST_CSV_PATH = 'dummy_test.csv'
    NUM_FEATURES = 34 # Matches your expected 17 timesteps * 2 features_per_step
    NUM_TRAIN_SAMPLES = 100
    NUM_TEST_SAMPLES = 10

    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        print(f"Creating dummy {TRAIN_CSV_PATH} and {TEST_CSV_PATH}...")
        
        # Generate dummy feature names
        dummy_feature_names = [f"Feature_{i:02d}" for i in range(NUM_FEATURES)]
        
        # Train data
        dummy_train_data = np.random.rand(NUM_TRAIN_SAMPLES, NUM_FEATURES)
        dummy_train_labels = np.random.choice(["no stress", "time pressure", "interruption"], NUM_TRAIN_SAMPLES)
        dummy_train_df = pd.DataFrame(dummy_train_data, columns=dummy_feature_names)
        dummy_train_df['datasetId'] = np.arange(NUM_TRAIN_SAMPLES) + 1
        dummy_train_df['condition'] = dummy_train_labels
        dummy_train_df.to_csv(TRAIN_CSV_PATH, index=False)

        # Test data
        dummy_test_data = np.random.rand(NUM_TEST_SAMPLES, NUM_FEATURES)
        dummy_test_labels = np.random.choice(["no stress", "time pressure", "interruption"], NUM_TEST_SAMPLES)
        dummy_test_df = pd.DataFrame(dummy_test_data, columns=dummy_feature_names)
        dummy_test_df['datasetId'] = np.arange(NUM_TEST_SAMPLES) + 101
        dummy_test_df['condition'] = dummy_test_labels
        dummy_test_df.to_csv(TEST_CSV_PATH, index=False)
        print("Dummy CSV files created.")

    # Initialize data loader
    data_loader = SWELLDataLoader(train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH)

    # Load data and unpack the 5 values
    X_train_raw, X_test_raw, y_train_encoded, y_test_encoded, feature_names = data_loader.load_data()

    print("\n--- Verification from data_loader.py test block ---")
    print(f"X_train_raw shape: {X_train_raw.shape}")
    print(f"X_test_raw shape: {X_test_raw.shape}")
    print(f"y_train_encoded shape: {y_train_encoded.shape}")
    print(f"y_test_encoded shape: {y_test_encoded.shape}")
    print(f"Feature names (first 5): {feature_names[:5]}")
    print(f"Total feature names count: {len(feature_names)}")
    print(f"Label classes: {data_loader.label_encoder.classes_}")

    # Test preprocessing
    X_train_scaled = data_loader.preprocess(X_train_raw, fit=True)
    X_test_scaled = data_loader.preprocess(X_test_raw, fit=False)
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")

    # Clean up dummy files
    if os.path.exists(TRAIN_CSV_PATH):
        os.remove(TRAIN_CSV_PATH)
    if os.path.exists(TEST_CSV_PATH):
        os.remove(TEST_CSV_PATH)
    print("\nDummy CSV files cleaned up.")