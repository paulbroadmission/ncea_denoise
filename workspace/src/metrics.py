"""
Evaluation metrics for SSVEP classification.

Implements:
- Accuracy
- F1 score (weighted)
- Information Transfer Rate (ITR)
- Confusion matrix
- Inter-trial consistency measures
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class indices of shape (num_samples,)
        labels: True class indices of shape (num_samples,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    return accuracy_score(labels, predictions)


def compute_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute weighted F1 score.

    Args:
        predictions: Predicted class indices of shape (num_samples,)
        labels: True class indices of shape (num_samples,)

    Returns:
        F1 score (weighted average)
    """
    return f1_score(labels, predictions, average="weighted", zero_division=0)


def compute_itr(
    accuracy: float,
    num_classes: int,
    trial_duration: float = 1.0,
    num_trials: int = 1,
) -> float:
    """
    Compute Information Transfer Rate (ITR) in bits per minute.

    ITR formula (for single-trial classification):
        ITR = [log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))] * 60 / T

    where:
        N = number of classes
        P = accuracy (0 to 1)
        T = trial time in seconds

    Args:
        accuracy: Classification accuracy (0 to 1)
        num_classes: Number of SSVEP classes
        trial_duration: Duration of a single trial in seconds
        num_trials: Number of trials averaged (for multi-trial classification)

    Returns:
        ITR in bits per minute
    """
    # Effective duration
    total_time = trial_duration * num_trials

    # Handle edge cases
    if accuracy >= 1.0:
        accuracy = 0.9999
    elif accuracy < 1.0 / num_classes:
        accuracy = 1.0 / num_classes

    # ITR formula
    itr = (
        np.log2(num_classes) +
        accuracy * np.log2(accuracy) +
        (1 - accuracy) * np.log2((1 - accuracy) / (num_classes - 1))
    ) * 60.0 / total_time

    return itr


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices
        labels: True class indices

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    return confusion_matrix(labels, predictions)


def compute_inter_trial_consistency(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, dict]:
    """
    Compute within-class and between-class consistency metrics.

    This measures how well the learned features separate different SSVEP classes.

    Args:
        features: Encoded features of shape (num_samples, feature_dim)
        labels: Class labels of shape (num_samples,)

    Returns:
        (overall_consistency, detailed_dict) where:
        - overall_consistency: Average within-class distance (lower is better)
        - detailed_dict: Contains within-class and between-class metrics
    """
    num_classes = len(np.unique(labels))

    within_class_distances = []
    between_class_distances = []

    # Compute class centers
    class_centers = {}
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        if len(class_features) > 0:
            class_centers[class_idx] = class_features.mean(axis=0)

    # Within-class distances (should be small)
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        if len(class_features) > 1:
            center = class_centers[class_idx]
            distances = np.linalg.norm(class_features - center, axis=1)
            within_class_distances.append(distances.mean())

    # Between-class distances (should be large)
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            center_i = class_centers[i]
            center_j = class_centers[j]
            distance = np.linalg.norm(center_i - center_j)
            between_class_distances.append(distance)

    avg_within = np.mean(within_class_distances) if within_class_distances else 0.0
    avg_between = np.mean(between_class_distances) if between_class_distances else 0.0

    # Consistency metric: ratio of between to within
    consistency_ratio = avg_between / (avg_within + 1e-8)

    return avg_within, {
        "within_class_distance": avg_within,
        "between_class_distance": avg_between,
        "consistency_ratio": consistency_ratio,
    }


def compute_trial_alignment_quality(
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute a metric of trial alignment quality.

    Lower within-class variance and higher between-class variance is better.

    Returns:
        Alignment quality score (0 to 1, higher is better)
    """
    num_classes = len(np.unique(labels))

    # Compute class-wise statistics
    within_class_vars = []
    between_class_vars = []

    # Within-class variance
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        if len(class_features) > 1:
            var = np.var(class_features, axis=0).mean()
            within_class_vars.append(var)

    # Between-class variance
    all_mean = features.mean(axis=0)
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        class_mean = class_features.mean(axis=0)
        var = np.mean((class_mean - all_mean) ** 2)
        between_class_vars.append(var)

    avg_within_var = np.mean(within_class_vars) if within_class_vars else 0.0
    avg_between_var = np.mean(between_class_vars) if between_class_vars else 0.0

    # Alignment quality: high between-class variance, low within-class variance
    if avg_within_var > 0:
        alignment_quality = avg_between_var / (avg_within_var + avg_between_var)
    else:
        alignment_quality = 0.0

    return alignment_quality


class MetricTracker:
    """Track and accumulate metrics across batches."""

    def __init__(self, metric_names: list):
        self.metric_names = metric_names
        self.metrics = {name: [] for name in metric_names}

    def update(self, **kwargs):
        """Update metrics."""
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].append(value)

    def get_averages(self) -> dict:
        """Get averaged metrics."""
        averages = {}
        for name, values in self.metrics.items():
            if values:
                averages[name] = np.mean(values)
            else:
                averages[name] = 0.0
        return averages

    def reset(self):
        """Reset metrics."""
        for name in self.metric_names:
            self.metrics[name] = []


# Test code
if __name__ == "__main__":
    print("=== Testing Metrics ===\n")

    # Create synthetic data
    num_samples = 100
    num_classes = 12
    feature_dim = 80

    predictions = np.random.randint(0, num_classes, size=num_samples)
    labels = np.random.randint(0, num_classes, size=num_samples)
    features = np.random.randn(num_samples, feature_dim)

    # Test 1: Accuracy
    acc = compute_accuracy(predictions, labels)
    print(f"Accuracy: {acc:.4f}")

    # Test 2: F1
    f1 = compute_f1(predictions, labels)
    print(f"F1 (weighted): {f1:.4f}")

    # Test 3: ITR
    itr = compute_itr(accuracy=acc, num_classes=num_classes, trial_duration=1.0)
    print(f"ITR (bits/min): {itr:.2f}")

    # Test 4: Consistency
    within_dist, consistency_dict = compute_inter_trial_consistency(features, labels)
    print(f"\nConsistency metrics:")
    print(f"  Within-class distance: {consistency_dict['within_class_distance']:.4f}")
    print(f"  Between-class distance: {consistency_dict['between_class_distance']:.4f}")
    print(f"  Consistency ratio: {consistency_dict['consistency_ratio']:.4f}")

    # Test 5: Alignment quality
    alignment = compute_trial_alignment_quality(features, labels)
    print(f"\nAlignment quality: {alignment:.4f}")

    # Test 6: Metric tracker
    print(f"\n=== Testing MetricTracker ===")
    tracker = MetricTracker(["loss", "accuracy"])
    for i in range(10):
        tracker.update(loss=np.random.rand(), accuracy=0.8 + np.random.rand() * 0.2)
    print(f"Averages: {tracker.get_averages()}")
