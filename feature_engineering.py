# feature_engineering.py

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional


class FeatureEngineer:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.pca_model = PCA(n_components=n_components) if n_components else None

    def apply_pca(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PCA to reduce dimensions (if enabled)"""
        if not self.pca_model:
            return X_train, X_test

        X_train_reduced = self.pca_model.fit_transform(X_train)
        X_test_reduced = self.pca_model.transform(X_test)
        return X_train_reduced, X_test_reduced

    def reshape_for_sequence(self, X: np.ndarray, timesteps: int) -> np.ndarray:
        """
        Reshape flat features to (samples, timesteps, features_per_step)
        Assumes features_per_step = original_features // timesteps
        """
        if X.shape[1] % timesteps != 0:
            raise ValueError("Number of features must be divisible by timesteps.")
        features_per_step = X.shape[1] // timesteps
        return X.reshape((X.shape[0], timesteps, features_per_step))

    def engineer(self, X_train: np.ndarray, X_test: np.ndarray, timesteps: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all feature engineering steps"""
        X_train, X_test = self.apply_pca(X_train, X_test)
        if timesteps:
            X_train = self.reshape_for_sequence(X_train, timesteps)
            X_test = self.reshape_for_sequence(X_test, timesteps)
        return X_train, X_test
