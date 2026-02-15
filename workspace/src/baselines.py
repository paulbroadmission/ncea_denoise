"""
Baseline implementations for comparison.

Includes:
- Traditional TRCA
- CNN baseline (no consistency)
- TRCA + CNN hybrid (Li et al. 2024)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Tuple


class TraditionalTRCA:
    """
    Task-Related Component Analysis (Nakanishi et al., 2014).

    Linear method that maximizes inter-trial reproducibility via
    generalized eigenvalue problem.
    """

    def __init__(self, n_components: int = 4):
        """
        Args:
            n_components: Number of TRCA components to extract
        """
        self.n_components = n_components
        self.W = None  # Projection matrix
        self.templates = None  # Class templates

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit TRCA model.

        Args:
            X_train: Training data of shape (num_trials, num_channels, samples)
            y_train: Training labels of shape (num_trials,)
        """
        num_channels, num_samples = X_train.shape[1], X_train.shape[2]
        num_classes = len(np.unique(y_train))

        # For simplicity, implement basic TRCA for single class
        # (full multi-class TRCA is more complex)

        # Reshape to (num_trials, num_channels * samples)
        X_flat = X_train.reshape(X_train.shape[0], -1)

        # Compute within-trial and between-trial covariances
        Qw = np.zeros((X_flat.shape[1], X_flat.shape[1]))
        Qb = np.zeros_like(Qw)

        # For each class
        for class_idx in range(num_classes):
            class_trials = X_flat[y_train == class_idx]

            # Between-trial covariance
            for i in range(len(class_trials)):
                for j in range(len(class_trials)):
                    if i != j:
                        Qb += np.outer(class_trials[i], class_trials[j])

            # Within-trial covariance
            for trial in class_trials:
                Qw += np.outer(trial, trial)

        # Normalize
        Qb /= max(Qb.sum(), 1e-8)
        Qw /= max(Qw.sum(), 1e-8)

        # Generalized eigenvalue problem: Qb w = lambda * Qw w
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(Qb, Qw)
            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            self.W = eigenvectors[:, idx[:self.n_components]]
        except np.linalg.LinAlgError:
            # Fallback: use PCA if eigh fails
            U, S, Vt = np.linalg.svd(X_flat, full_matrices=False)
            self.W = U[:, :self.n_components]

        # Compute class templates
        self.templates = {}
        for class_idx in range(num_classes):
            class_trials = X_flat[y_train == class_idx]
            template = self.W.T @ class_trials.mean(axis=0)
            self.templates[class_idx] = template

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X_test: Test data of shape (num_trials, num_channels, samples)

        Returns:
            Predicted labels of shape (num_trials,)
        """
        X_flat = X_test.reshape(X_test.shape[0], -1)
        X_proj = X_flat @ self.W  # Project to TRCA space

        # Classify as nearest template
        predictions = []
        for x_proj in X_proj:
            distances = {
                class_idx: np.linalg.norm(x_proj - template)
                for class_idx, template in self.templates.items()
            }
            pred = min(distances.keys(), key=lambda k: distances[k])
            predictions.append(pred)

        return np.array(predictions)

    def evaluate(self, X_train, y_train, X_test, y_test) -> dict:
        """Train and evaluate TRCA."""
        self.fit(X_train, y_train)
        predictions = self.predict(X_test)
        accuracy = (predictions == y_test).mean()
        return {"accuracy": accuracy}


class CNNBaseline(nn.Module):
    """
    CNN baseline without consistency regularization.

    Simple CNN for classification without ensemble principles.
    """

    def __init__(self, num_channels=8, num_samples=250, num_classes=12):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        flattened_size = 64 * (num_samples // 4)

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class TRCACNNHybrid(nn.Module):
    """
    TRCA + CNN Hybrid (Li et al., 2024).

    Combines TRCA preprocessing with CNN classification.
    """

    def __init__(self, num_channels=8, num_samples=250, num_classes=12, n_trca=4):
        super().__init__()

        self.n_trca = n_trca
        self.trca_components = nn.Parameter(
            torch.randn(num_channels * num_samples, n_trca)
        )

        # CNN operates on TRCA-projected features
        self.cnn = CNNBaseline(
            num_channels=n_trca,
            num_samples=num_samples,
            num_classes=num_classes,
        )

    def forward(self, x):
        # TRCA projection
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_trca = x_flat @ self.trca_components  # (batch_size, n_trca)
        x_trca = x_trca.view(batch_size, self.n_trca, -1)  # Reshape for CNN

        # CNN classification
        logits = self.cnn(x_trca)
        return logits


# Test code
if __name__ == "__main__":
    print("Testing baselines...\n")

    # Create dummy data
    num_trials = 100
    num_channels = 8
    num_samples = 250
    num_classes = 12

    X = np.random.randn(num_trials, num_channels, num_samples)
    y = np.random.randint(0, num_classes, num_trials)

    # Split
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Test 1: TRCA
    print("=== Testing TRCA ===")
    trca = TraditionalTRCA(n_components=4)
    metrics = trca.evaluate(X_train, y_train, X_test, y_test)
    print(f"TRCA Accuracy: {metrics['accuracy']:.4f}\n")

    # Test 2: CNN Baseline
    print("=== Testing CNN Baseline ===")
    cnn = CNNBaseline(num_channels=num_channels, num_samples=num_samples, num_classes=num_classes)
    x_batch = torch.randn(8, num_channels, num_samples)
    output = cnn(x_batch)
    print(f"CNN output shape: {output.shape}")
    print(f"CNN Baseline Accuracy: N/A (not trained)\n")

    # Test 3: TRCA+CNN Hybrid
    print("=== Testing TRCA+CNN Hybrid ===")
    hybrid = TRCACNNHybrid(
        num_channels=num_channels,
        num_samples=num_samples,
        num_classes=num_classes,
        n_trca=4
    )
    output = hybrid(x_batch)
    print(f"TRCA+CNN output shape: {output.shape}")
    print(f"TRCA+CNN Accuracy: N/A (not trained)")
