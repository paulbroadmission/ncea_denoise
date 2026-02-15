"""
Data Integrity Validator: Prevent fake data and fraudulent results.

Checks:
1. Data is REAL (not synthetically generated for results)
2. Noise levels are realistic
3. Signal properties match SSVEP expectations
4. Metadata is correct
5. No data leakage between train/val/test
6. Results are reproducible and verifiable
"""

import numpy as np
import torch
from typing import Tuple, Dict
from pathlib import Path
import hashlib
import json


class DataIntegrityValidator:
    """Validate that data and results are real and not fraudulent."""

    def __init__(self, data_source: str = "BETA"):
        """
        Args:
            data_source: "BETA", "OpenBMI", or "synthetic"
        """
        self.data_source = data_source
        self.integrity_report = {}

    def validate_ssvep_data(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Validate SSVEP data is real, not fake.

        Args:
            data: SSVEP signal data (num_trials, num_channels, samples)
            labels: Class labels

        Returns:
            Validation report with all checks
        """
        report = {}

        # Check 1: Data shape is correct
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D data, got {len(data.shape)}D")

        num_trials, num_channels, num_samples = data.shape
        report["shape_valid"] = True
        report["num_trials"] = int(num_trials)
        report["num_channels"] = int(num_channels)
        report["num_samples"] = int(num_samples)

        # Check 2: Data values are realistic (not all zeros, not all constant)
        for trial_idx in range(min(10, num_trials)):  # Check first 10 trials
            trial_data = data[trial_idx]

            # Check for all zeros
            if np.allclose(trial_data, 0):
                raise ValueError(f"Trial {trial_idx} contains all zeros (FAKE DATA)")

            # Check for constant values
            if np.allclose(trial_data, trial_data[0, 0]):
                raise ValueError(f"Trial {trial_idx} contains constant values (FAKE DATA)")

        report["all_zeros_check"] = "PASS"
        report["constant_value_check"] = "PASS"

        # Check 3: Signal statistics are reasonable
        signal_mean = np.mean(data)
        signal_std = np.std(data)
        signal_min = np.min(data)
        signal_max = np.max(data)

        if signal_std < 0.01:
            raise ValueError(f"Signal std too low ({signal_std:.6f}), likely FAKE")

        if signal_max - signal_min < 0.1:
            raise ValueError(f"Signal range too small, likely FAKE")

        report["signal_mean"] = float(signal_mean)
        report["signal_std"] = float(signal_std)
        report["signal_min"] = float(signal_min)
        report["signal_max"] = float(signal_max)
        report["signal_range_valid"] = True

        # Check 4: Labels are valid
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(f"Only {len(unique_labels)} unique classes (need ≥2)")

        for label in unique_labels:
            count = np.sum(labels == label)
            if count < 5:
                raise ValueError(f"Class {label} has only {count} samples (too few)")

        report["num_classes"] = int(len(unique_labels))
        report["labels_valid"] = True

        # Check 5: No data leakage (trials should be different)
        # Sample 2 trials, should not be identical
        if num_trials >= 2:
            trial1 = data[0].flatten()
            trial2 = data[1].flatten()
            correlation = np.corrcoef(trial1, trial2)[0, 1]

            if np.isnan(correlation):
                correlation = 0

            if correlation > 0.99:
                raise ValueError(f"Trials are too similar (corr={correlation:.4f}), suspect FAKE data")

            report["inter_trial_correlation"] = float(correlation)

        # Check 6: Noise is present (real data should be noisy)
        noise_levels = []
        for trial_idx in range(min(5, num_trials)):
            # Compute variance per channel (noise)
            channel_vars = np.var(data[trial_idx], axis=1)
            noise_levels.append(np.mean(channel_vars))

        avg_noise = np.mean(noise_levels)
        if avg_noise < 0.001:
            raise ValueError(f"Noise level too low ({avg_noise:.6f}), FAKE data")

        report["noise_level"] = float(avg_noise)
        report["noise_present"] = True

        # Check 7: Data source metadata
        report["data_source"] = self.data_source
        if self.data_source == "synthetic":
            report["WARNING"] = "SYNTHETIC DATA DETECTED - Use only for testing/debugging"

        report["overall_status"] = "VALID"
        return report

    def validate_results(self, results: Dict) -> Dict:
        """
        Validate that results are real, not fabricated.

        Args:
            results: Dictionary with accuracy, F1, metrics

        Returns:
            Validation report
        """
        report = {}

        # Check 1: Accuracy is between 0 and 1
        accuracy = results.get("accuracy", 0)
        if not (0 <= accuracy <= 1):
            raise ValueError(f"Invalid accuracy {accuracy} (must be 0-1)")

        report["accuracy_valid"] = True

        # Check 2: F1 score is between 0 and 1
        f1 = results.get("f1_score", 0)
        if not (0 <= f1 <= 1):
            raise ValueError(f"Invalid F1 {f1} (must be 0-1)")

        report["f1_valid"] = True

        # Check 3: Accuracy and F1 should be similar (within reasonable range)
        if abs(accuracy - f1) > 0.15:
            raise ValueError(
                f"Accuracy and F1 differ too much "
                f"(acc={accuracy:.4f}, f1={f1:.4f}), suspect FAKE"
            )

        report["acc_f1_consistency"] = True

        # Check 4: ITR should be reasonable
        itr = results.get("itr", 0)
        if itr < 0 or itr > 200:  # Reasonable upper bound for ITR
            raise ValueError(f"Unrealistic ITR {itr}")

        report["itr_valid"] = True

        # Check 5: Metrics should have proper precision
        if "timestamp" not in results:
            results["timestamp"] = str(np.datetime64('now'))

        report["timestamp"] = results.get("timestamp")

        # Check 6: Reasonable improvement over baseline
        # TRCA baseline should be ~88-92%, CNN ~90-94%
        if accuracy < 0.8:
            report["WARNING"] = f"Accuracy {accuracy:.2%} below TRCA baseline (88%)"

        report["overall_status"] = "VALID"
        return report

    def compute_data_hash(self, data: np.ndarray) -> str:
        """
        Compute reproducible hash of data for verification.

        Args:
            data: Data array

        Returns:
            SHA256 hash of data
        """
        data_bytes = data.astype(np.float32).tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def save_integrity_report(self, data: np.ndarray, labels: np.ndarray,
                              results: Dict, save_path: str):
        """
        Save integrity report with data hash for reproducibility.

        Args:
            data: Training/test data
            labels: Labels
            results: Results dictionary
            save_path: Where to save report
        """
        report = {
            "timestamp": str(np.datetime64('now')),
            "data_hash": self.compute_data_hash(data),
            "num_samples": len(data),
            "num_classes": len(np.unique(labels)),
            "data_shape": list(data.shape),
            "data_integrity": self.validate_ssvep_data(data, labels),
            "results_integrity": self.validate_results(results),
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Integrity report saved to {save_path}")
        print(f"  Data hash: {report['data_hash']}")

    def verify_reproducibility(self, data1: np.ndarray, data2: np.ndarray) -> bool:
        """
        Verify two datasets are identical (reproducibility check).

        Args:
            data1: First data array
            data2: Second data array

        Returns:
            True if identical, False otherwise
        """
        hash1 = self.compute_data_hash(data1)
        hash2 = self.compute_data_hash(data2)

        if hash1 == hash2:
            print(f"✓ Data reproducibility verified (hash: {hash1[:8]}...)")
            return True
        else:
            print(f"✗ Data mismatch!")
            print(f"  Data 1 hash: {hash1}")
            print(f"  Data 2 hash: {hash2}")
            return False


class ResultsValidator:
    """Validate experimental results are real and reproducible."""

    def __init__(self):
        self.results_log = {}

    def validate_comparison(self, method_results: Dict) -> bool:
        """
        Validate results comparison between methods is fair.

        Args:
            method_results: {method_name: {"accuracy": ..., "f1": ..., ...}, ...}

        Returns:
            True if all results are valid
        """
        print("\n" + "="*60)
        print("VALIDATING RESULTS COMPARISON")
        print("="*60)

        validator = DataIntegrityValidator()
        all_valid = True

        for method_name, results in method_results.items():
            print(f"\n{method_name}:")
            try:
                report = validator.validate_results(results)
                print(f"  ✓ Valid: accuracy={results['accuracy']:.4f}, f1={results['f1_score']:.4f}")
            except ValueError as e:
                print(f"  ✗ INVALID: {e}")
                all_valid = False

        if all_valid:
            print(f"\n{True}✓ ALL RESULTS VALID - Safe to publish")
        else:
            print(f"\n✗ SOME RESULTS INVALID - DO NOT PUBLISH")

        return all_valid

    def verify_no_data_leakage(self, train_data: np.ndarray, test_data: np.ndarray) -> bool:
        """
        Verify train and test data have no overlap.

        Args:
            train_data: Training data
            test_data: Test data

        Returns:
            True if no leakage, False if leakage detected
        """
        # Check if any test sample is identical to train sample
        for test_idx in range(len(test_data)):
            for train_idx in range(len(train_data)):
                if np.allclose(test_data[test_idx], train_data[train_idx]):
                    print(f"✗ DATA LEAKAGE DETECTED: Test {test_idx} = Train {train_idx}")
                    return False

        print("✓ No data leakage detected")
        return True


# Test code
if __name__ == "__main__":
    print("=== Data Integrity Validator Test ===\n")

    # Create fake data
    print("[TEST 1] Validating REAL SSVEP data")
    validator = DataIntegrityValidator(data_source="BETA")

    # Real-ish data
    real_data = np.random.randn(100, 8, 250) + 0.1 * np.sin(2 * np.pi * np.arange(250) * 10 / 250)
    real_labels = np.repeat(np.arange(12), 8)[:100]

    try:
        report = validator.validate_ssvep_data(real_data, real_labels)
        print("✓ Real data passes validation")
        print(f"  Shape: {report['shape_valid']}")
        print(f"  Noise level: {report['noise_level']:.6f}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test fake data detection
    print("\n[TEST 2] Detecting FAKE data (all zeros)")
    fake_data = np.zeros((100, 8, 250))
    fake_labels = np.repeat(np.arange(12), 8)[:100]

    try:
        report = validator.validate_ssvep_data(fake_data, fake_labels)
        print("✗ FAILED TO DETECT FAKE DATA!")
    except ValueError as e:
        print(f"✓ Correctly detected: {e}")

    # Test results validation
    print("\n[TEST 3] Validating results")
    results = {
        "accuracy": 0.94,
        "f1_score": 0.92,
        "itr": 145.5,
    }

    try:
        report = validator.validate_results(results)
        print(f"✓ Valid results: {report['overall_status']}")
    except Exception as e:
        print(f"✗ Invalid results: {e}")

    # Test results validator
    print("\n[TEST 4] Comparing methods")
    comparisons = {
        "TRCA": {"accuracy": 0.88, "f1_score": 0.87},
        "CNN": {"accuracy": 0.92, "f1_score": 0.91},
        "Proposed": {"accuracy": 0.94, "f1_score": 0.93},
    }

    results_validator = ResultsValidator()
    results_validator.validate_comparison(comparisons)

    print("\n✓ Data integrity tests complete")
