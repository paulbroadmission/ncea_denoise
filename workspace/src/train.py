"""
Training loop for Neural Conditional Ensemble Averaging.

Implements:
- Forward pass with loss computation
- Backward pass and optimization
- Validation and evaluation
- Logging and checkpointing
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
from typing import Dict, Tuple
import json
import os
from datetime import datetime

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, LAMBDA_CONSISTENCY,
    OPTIMIZER, LR_SCHEDULER, WARMUP_EPOCHS,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_METRIC,
    LOG_INTERVAL, CHECKPOINT_ROOT, LOG_ROOT, DEBUG
)
from model import SSVEPEncoder, SSVEPClassifier, create_encoder, create_classifier
from losses import CombinedLoss, ConsistencyLoss, CrossEntropyWithConsistency
from data import create_data_loaders
from metrics import compute_accuracy, compute_f1, compute_itr


class Trainer:
    """
    Trainer class for neural conditional ensemble averaging.

    Handles:
    - Training loop (forward, backward, update)
    - Validation loop
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        test_loader,
        lambda_consistency: float = LAMBDA_CONSISTENCY,
        learning_rate: float = LEARNING_RATE,
        num_epochs: int = NUM_EPOCHS,
        checkpoint_dir: str = CHECKPOINT_ROOT,
        log_dir: str = LOG_ROOT,
        device: str = DEVICE,
    ):
        """
        Args:
            model: Encoder or classifier model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            lambda_consistency: Consistency loss weight
            learning_rate: Learning rate
            num_epochs: Number of epochs
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for logging
            device: "cuda" or "cpu"
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.lambda_consistency = lambda_consistency
        self.num_epochs = num_epochs

        # Loss function (consistency only, as per user choice)
        self.loss_fn = CombinedLoss(
            lambda_consistency=lambda_consistency,
            lambda_template=0.0,
            template_variant="none"
        )

        # Optimizer
        if OPTIMIZER == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
            )

        # Learning rate scheduler
        if LR_SCHEDULER == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - WARMUP_EPOCHS,
            )
        else:
            self.scheduler = None

        # Checkpointing and logging
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Training history
        self.history = {
            "train_loss": [],
            "train_consistency": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_val_accuracy": 0.0,
            "best_epoch": 0,
        }

        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Single training epoch.

        Returns:
            Dictionary with loss metrics
        """
        self.model.train()
        total_loss = 0.0
        total_consistency = 0.0
        num_batches = 0

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)

            # Forward pass: Get encoded features
            if isinstance(self.model, SSVEPClassifier):
                features = self.model.encode(data)
            else:
                features = self.model(data)

            # Compute loss (consistency only)
            loss_dict = self.loss_fn(features, epoch=epoch)
            loss = loss_dict["total"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_consistency += loss_dict["consistency"].item()
            num_batches += 1

            # Logging
            if batch_idx % LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Consistency: {loss_dict['consistency'].item():.6f}"
                )

        avg_loss = total_loss / num_batches
        avg_consistency = total_consistency / num_batches

        return {
            "loss": avg_loss,
            "consistency": avg_consistency,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validation loop.

        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_features = []
        all_labels = []
        num_batches = 0

        for data, labels in self.val_loader:
            data = data.to(self.device)
            all_labels.append(labels.numpy())

            # Forward pass
            if isinstance(self.model, SSVEPClassifier):
                features = self.model.encode(data)
            else:
                features = self.model(data)

            all_features.append(features.cpu().numpy())

            # Compute loss
            loss_dict = self.loss_fn(features, epoch=None)
            total_loss += loss_dict["total"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics on encoded features using simple NN classifier
        # (For now, just compute feature consistency)
        feature_consistency = self._compute_feature_consistency(all_features, all_labels)
        accuracy = self._compute_clustering_accuracy(all_features, all_labels)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "feature_consistency": feature_consistency,
        }

    def _compute_feature_consistency(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute average within-class feature distance (lower is better).

        This measures how well L_consistency achieves manifold clustering.
        """
        num_classes = len(np.unique(labels))
        within_class_distances = []

        for class_idx in range(num_classes):
            class_features = features[labels == class_idx]
            if len(class_features) > 1:
                # Compute mean distance to class center
                class_center = class_features.mean(axis=0, keepdims=True)
                distances = np.linalg.norm(class_features - class_center, axis=1)
                within_class_distances.append(distances.mean())

        return np.mean(within_class_distances) if within_class_distances else 0.0

    def _compute_clustering_accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute simple clustering accuracy.

        Uses nearest class center classification.
        """
        num_classes = len(np.unique(labels))
        num_samples = len(labels)
        correct = 0

        # Compute class centers
        class_centers = []
        for class_idx in range(num_classes):
            class_features = features[labels == class_idx]
            class_centers.append(class_features.mean(axis=0))

        class_centers = np.array(class_centers)

        # Classify each sample to nearest class center
        for i in range(num_samples):
            distances = np.linalg.norm(class_centers - features[i], axis=1)
            pred_label = np.argmin(distances)
            if pred_label == labels[i]:
                correct += 1

        return correct / num_samples

    def train(self) -> Dict:
        """
        Full training loop.

        Returns:
            Training history
        """
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Batch size: {len(next(iter(self.train_loader))[0])}")

        for epoch in range(self.num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate()

            # Learning rate update
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            print(
                f"\nEpoch {epoch+1}/{self.num_epochs} Summary:\n"
                f"  Train Loss: {train_metrics['loss']:.6f} "
                f"(Consistency: {train_metrics['consistency']:.6f})\n"
                f"  Val Loss: {val_metrics['loss']:.6f} "
                f"(Accuracy: {val_metrics['accuracy']:.4f})"
            )

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_consistency"].append(train_metrics["consistency"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            # Early stopping check
            if val_metrics["accuracy"] > self.history["best_val_accuracy"]:
                self.history["best_val_accuracy"] = val_metrics["accuracy"]
                self.history["best_epoch"] = epoch + 1
                self.patience_counter = 0

                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Save final history
        self.save_history()
        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_history(self):
        """Save training history as JSON."""
        history_path = os.path.join(self.log_dir, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(history_path, "w") as f:
            # Convert to JSON-serializable format
            history_json = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in self.history.items()
            }
            json.dump(history_json, f, indent=2)
        print(f"Saved history: {history_path}")


def main():
    """Main training script."""
    print("=== Neural Conditional Ensemble Averaging Training ===\n")

    # Load data
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="synthetic" if DEBUG else "BETA",
        batch_size=4 if DEBUG else 32,
    )

    # Create model (encoder only, using consistency loss)
    model = create_encoder(encoder_type="cnn")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lambda_consistency=LAMBDA_CONSISTENCY,
        learning_rate=LEARNING_RATE,
        num_epochs=2 if DEBUG else NUM_EPOCHS,
        device=DEVICE,
    )

    # Train
    history = trainer.train()

    print("\n=== Training Complete ===")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.4f} (epoch {history['best_epoch']})")

    return history


if __name__ == "__main__":
    main()
