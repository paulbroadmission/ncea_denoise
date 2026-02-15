"""
Evaluation module for testing trained models.

Computes metrics on test set and generates reports.
"""

import torch
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from metrics import (
    compute_accuracy, compute_f1, compute_itr,
    compute_confusion_matrix, compute_inter_trial_consistency,
    compute_trial_alignment_quality
)


class Evaluator:
    """Evaluate trained models on test set."""

    def __init__(self, model, test_loader, device="cuda", num_classes=12):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Full evaluation on test set.

        Returns:
            Dictionary with metrics
        """
        self.model.eval()

        all_features = []
        all_labels = []

        # Extract features from all test samples
        for data, labels in self.test_loader:
            data = data.to(self.device)
            features = self.model(data)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

        all_features = np.concatenate(all_features, axis=0)  # (num_samples, feature_dim)
        all_labels = np.concatenate(all_labels, axis=0)  # (num_samples,)

        # Use KNN classifier on features to get predictions
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(all_features, all_labels)
        predictions = knn.predict(all_features)

        # Compute metrics
        metrics = {}

        # 1. Classification metrics
        metrics["accuracy"] = compute_accuracy(predictions, all_labels)
        metrics["f1_score"] = compute_f1(predictions, all_labels)
        metrics["itr"] = compute_itr(
            accuracy=metrics["accuracy"],
            num_classes=self.num_classes,
            trial_duration=1.0,
        )

        # 2. Feature quality metrics
        within_dist, consistency_dict = compute_inter_trial_consistency(
            all_features, all_labels
        )
        metrics["within_class_distance"] = consistency_dict["within_class_distance"]
        metrics["between_class_distance"] = consistency_dict["between_class_distance"]
        metrics["consistency_ratio"] = consistency_dict["consistency_ratio"]

        # 3. Alignment quality
        metrics["alignment_quality"] = compute_trial_alignment_quality(
            all_features, all_labels
        )

        return metrics

    @torch.no_grad()
    def get_predictions(self) -> tuple:
        """Get predictions for all test samples."""
        self.model.eval()

        all_features = []
        all_labels = []

        for data, labels in self.test_loader:
            data = data.to(self.device)
            features = self.model(data)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Use KNN for classification
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(all_features, all_labels)
        predictions = knn.predict(all_features)

        return predictions, all_labels, all_features

    def plot_confusion_matrix(self, save_path=None):
        """Plot and optionally save confusion matrix."""
        predictions, labels, _ = self.get_predictions()
        cm = compute_confusion_matrix(predictions, labels)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")

        return cm

    def plot_feature_distribution(self, save_path=None):
        """Plot 2D projection of learned features using t-SNE."""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("scikit-learn not available for t-SNE")
            return

        predictions, labels, features = self.get_predictions()

        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # Plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='tab20', s=50, alpha=0.6
        )
        plt.colorbar(scatter, label='Class')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Learned Feature Distribution (t-SNE)')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved feature distribution plot to {save_path}")

        return features_2d

    def generate_report(self, save_dir=None) -> str:
        """Generate evaluation report."""
        metrics = self.evaluate()

        report = []
        report.append("="*60)
        report.append("EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        report.append("Classification Metrics:")
        report.append(f"  Accuracy:     {metrics['accuracy']:.4f}")
        report.append(f"  F1 Score:     {metrics['f1_score']:.4f}")
        report.append(f"  ITR (bpm):    {metrics['itr']:.2f}")
        report.append("")
        report.append("Feature Quality Metrics:")
        report.append(f"  Within-class distance: {metrics['within_class_distance']:.6f}")
        report.append(f"  Between-class distance: {metrics['between_class_distance']:.6f}")
        report.append(f"  Consistency ratio:     {metrics['consistency_ratio']:.4f}")
        report.append(f"  Alignment quality:     {metrics['alignment_quality']:.4f}")
        report.append("="*60)

        report_str = "\n".join(report)
        print(report_str)

        if save_dir:
            report_path = f"{save_dir}/evaluation_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_str)
            print(f"Saved report to {report_path}")

        return report_str


if __name__ == "__main__":
    from data import create_data_loaders
    from model import create_encoder

    # Test evaluation
    print("Testing Evaluator...")

    # Load data
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="synthetic",
        batch_size=32,
    )

    # Create model
    model = create_encoder(encoder_type="cnn")

    # Evaluate (with untrained model, just for testing)
    evaluator = Evaluator(model, test_loader, device="cpu", num_classes=12)

    # Get metrics
    metrics = evaluator.evaluate()
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Generate report
    print("\n" + "="*60)
    evaluator.generate_report()
