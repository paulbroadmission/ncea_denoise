"""
Data loading and preprocessing for SSVEP datasets.

Supports:
- BETA dataset (standard benchmark)
- OpenBMI (if available)
- Synthetic data (for testing)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import os
from config import (
    SAMPLING_RATE, NUM_CHANNELS, SAMPLES_PER_TRIAL,
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ,
    DATA_ROOT, DEBUG
)

try:
    from scipy.signal import butter, sosfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SSVEPDataset(Dataset):
    """
    Generic SSVEP dataset class.

    Stores trials and labels for a single subject.
    """

    def __init__(
        self,
        data: np.ndarray,  # (num_trials, num_channels, samples_per_trial)
        labels: np.ndarray,  # (num_trials,)
        subject_id: int,
        condition: str = "train",  # "train", "val", "test"
        apply_preprocessing: bool = True,
    ):
        """
        Args:
            data: Trials of shape (num_trials, num_channels, samples_per_trial)
            labels: Class labels of shape (num_trials,)
            subject_id: Subject ID
            condition: Train/val/test split
            apply_preprocessing: Whether to apply bandpass filtering
        """
        self.data = torch.from_numpy(data).float()  # (num_trials, C, T)
        self.labels = torch.from_numpy(labels).long()
        self.subject_id = subject_id
        self.condition = condition
        self.apply_preprocessing = apply_preprocessing

        if apply_preprocessing:
            self.data = self._apply_preprocessing(self.data)

    def _apply_preprocessing(self, data):
        """Apply bandpass filtering to the data."""
        if not SCIPY_AVAILABLE:
            return data

        # Simple preprocessing: normalize each trial
        # (More advanced filtering can be added here)
        data_np = data.numpy()

        # Z-score normalization per trial
        for trial_idx in range(data_np.shape[0]):
            for ch in range(data_np.shape[1]):
                trial_ch = data_np[trial_idx, ch, :]
                trial_ch_norm = (trial_ch - trial_ch.mean()) / (trial_ch.std() + 1e-8)
                data_np[trial_idx, ch, :] = trial_ch_norm

        return torch.from_numpy(data_np).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a single trial and its label."""
        trial = self.data[idx]  # (C, T)
        label = self.labels[idx]  # Scalar
        return trial, label

    def get_trial_and_subject(self, idx):
        """Return trial, label, and subject ID."""
        trial = self.data[idx]
        label = self.labels[idx]
        return trial, label, self.subject_id


def create_synthetic_data(
    num_subjects: int = 2,
    num_trials_per_class: int = 20,
    num_classes: int = 12,
    num_channels: int = NUM_CHANNELS,
    samples_per_trial: int = SAMPLES_PER_TRIAL,
    sampling_rate: float = SAMPLING_RATE,
) -> dict:
    """
    Create synthetic SSVEP data for testing.

    Each class corresponds to a different frequency.

    Args:
        num_subjects: Number of synthetic subjects
        num_trials_per_class: Trials per class per subject
        num_classes: Number of SSVEP frequencies
        num_channels: Number of EEG channels
        samples_per_trial: Samples per trial
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with data, labels, and metadata
    """
    print(f"Creating synthetic SSVEP data ({num_subjects} subjects, {num_trials_per_class} trials/class)")

    # SSVEP frequencies (typical: 8-15 Hz)
    frequencies = np.linspace(8, 15, num_classes)
    time = np.arange(samples_per_trial) / sampling_rate

    data_dict = {}

    for subject_idx in range(num_subjects):
        subject_data = []
        subject_labels = []

        for class_idx, freq in enumerate(frequencies):
            for trial_idx in range(num_trials_per_class):
                # Generate sinusoidal signal at target frequency
                signal = np.sin(2 * np.pi * freq * time)

                # Add noise and channel variation
                trial = np.zeros((num_channels, samples_per_trial))
                for ch in range(num_channels):
                    # Channel-specific phase shift and amplitude
                    phase_shift = (ch / num_channels) * 2 * np.pi
                    amplitude = 0.8 + 0.2 * (ch / num_channels)

                    trial[ch, :] = amplitude * (
                        signal * np.cos(phase_shift) +
                        0.5 * np.random.randn(samples_per_trial)  # noise
                    )

                subject_data.append(trial)
                subject_labels.append(class_idx)

        subject_data = np.array(subject_data)  # (num_trials, C, T)
        subject_labels = np.array(subject_labels)  # (num_trials,)

        data_dict[f"subject_{subject_idx:02d}"] = {
            "data": subject_data,
            "labels": subject_labels,
        }

    return data_dict


def load_beta_dataset(data_root: str = DATA_ROOT) -> dict:
    """
    Load BETA SSVEP dataset.

    BETA dataset: https://github.com/gumpy-bci/data
    10 subjects, 12 SSVEP frequencies, 240 trials per subject

    This is a placeholder. In practice, you would:
    1. Download the dataset
    2. Extract and organize it
    3. Load with scipy.io.loadmat or similar

    For now, we'll create synthetic data as a placeholder.
    """
    print("BETA dataset loader (using synthetic data as placeholder)")
    print("To use real BETA data:")
    print("  1. Download from: https://github.com/gumpy-bci/data")
    print("  2. Extract to:", data_root)
    print("  3. Implement loading logic below")

    # For development, use synthetic data
    return create_synthetic_data(
        num_subjects=3 if DEBUG else 10,
        num_trials_per_class=10 if DEBUG else 20,
        num_classes=12,
    )


def load_openbmi_dataset(data_root: str = DATA_ROOT) -> dict:
    """Load OpenBMI dataset (placeholder)."""
    print("OpenBMI dataset loader (not implemented yet)")
    return create_synthetic_data(num_subjects=2)


def split_data(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[dict, dict, dict]:
    """
    Split data into train/val/test sets.

    Returns:
        (train_dict, val_dict, test_dict) with 'data' and 'labels' keys
    """
    num_trials = len(labels)
    indices = np.arange(num_trials)
    np.random.shuffle(indices)

    train_size = int(num_trials * train_ratio)
    val_size = int(num_trials * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return (
        {"data": data[train_indices], "labels": labels[train_indices]},
        {"data": data[val_indices], "labels": labels[val_indices]},
        {"data": data[test_indices], "labels": labels[test_indices]},
    )


def create_data_loaders(
    dataset_name: str = "BETA",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        dataset_name: "BETA", "OpenBMI", or "synthetic"
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print(f"Loading {dataset_name} dataset...")

    if dataset_name == "BETA":
        raw_data = load_beta_dataset()
    elif dataset_name == "OpenBMI":
        raw_data = load_openbmi_dataset()
    elif dataset_name == "synthetic":
        raw_data = create_synthetic_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # For simplicity, merge all subjects for now
    # (Later: implement per-subject and cross-subject evaluation)
    all_data = []
    all_labels = []

    for subject_key, subject_dict in raw_data.items():
        all_data.append(subject_dict["data"])
        all_labels.append(subject_dict["labels"])

    all_data = np.concatenate(all_data, axis=0)  # (total_trials, C, T)
    all_labels = np.concatenate(all_labels, axis=0)  # (total_trials,)

    print(f"Total trials: {len(all_labels)}")
    print(f"Data shape: {all_data.shape}")
    print(f"Unique classes: {len(np.unique(all_labels))}")

    # Split into train/val/test
    train_dict, val_dict, test_dict = split_data(
        all_data, all_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # Create datasets
    train_dataset = SSVEPDataset(
        train_dict["data"], train_dict["labels"],
        subject_id=-1, condition="train", apply_preprocessing=True
    )
    val_dataset = SSVEPDataset(
        val_dict["data"], val_dict["labels"],
        subject_id=-1, condition="val", apply_preprocessing=True
    )
    test_dataset = SSVEPDataset(
        test_dict["data"], test_dict["labels"],
        subject_id=-1, condition="test", apply_preprocessing=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# Test code
if __name__ == "__main__":
    # Test 1: Synthetic data creation
    print("=== Test 1: Synthetic Data ===")
    synthetic_data = create_synthetic_data(
        num_subjects=2,
        num_trials_per_class=5,
        num_classes=3,
    )
    for subject, data_dict in synthetic_data.items():
        print(f"{subject}: data {data_dict['data'].shape}, labels {data_dict['labels'].shape}")

    # Test 2: Data loaders
    print("\n=== Test 2: Data Loaders ===")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="synthetic",
        batch_size=8,
    )

    # Get a batch
    batch_data, batch_labels = next(iter(train_loader))
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch labels: {batch_labels}")
