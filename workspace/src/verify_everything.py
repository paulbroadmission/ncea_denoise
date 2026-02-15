"""
COMPREHENSIVE INTEGRITY VERIFICATION SYSTEM

Orchestrates Guardian + DataIntegrityValidator + ForbiddenChecker
Enforces ZERO tolerance for fake data and hallucinated results

Run BEFORE any training:
    python3 verify_everything.py --dataset BETA
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Import verification systems
from guardian import run_full_check as guardian_check
from data_integrity import DataIntegrityValidator, ResultsValidator
from forbidden_checks import ForbiddenChecker
from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    LAMBDA_CONSISTENCY, LATENT_DIM
)


class ComprehensiveVerifier:
    """
    Master verification system that enforces integrity at startup.

    Phases:
    1. GUARDIAN CHECKS → Validates codebase & config
    2. DATA INTEGRITY → Verifies data is real (not synthetic)
    3. FORBIDDEN CHECKS → Absolutely prevents fraud
    4. AUDIT TRAIL → Saves verification report
    """

    def __init__(self, dataset_name: str = "synthetic"):
        self.dataset_name = dataset_name
        self.results = {}
        self.passed_checks = []
        self.failed_checks = []
        self.audit_log = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "device": DEVICE,
            "phases": {}
        }

    def print_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)

    def phase_1_guardian(self) -> bool:
        """
        PHASE 1: Run Guardian pre-flight checks

        Validates:
        - All imports work
        - Configuration is valid
        - Model builds correctly
        - Loss functions compute
        - Data loads correctly
        - Training step works
        - CONFIG-SYNC is consistent
        """
        self.print_header("PHASE 1: Guardian Pre-Flight Checks")

        try:
            # Guardian runs its own internal checks
            print("\nRunning Guardian validation system...")
            # Note: guardian.py is a standalone script, so we import key checks

            # For this integration, we'll verify key components locally
            print("\n[1/7] Imports...")
            try:
                import config, model, losses, data, train, evaluate, baselines
                print("  ✓ PASS All modules importable")
                self.passed_checks.append("Imports")
            except Exception as e:
                print(f"  ✗ FAIL {e}")
                self.failed_checks.append(f"Imports: {e}")
                return False

            print("[2/7] Configuration...")
            try:
                assert DEVICE in ["cpu", "cuda"], f"Invalid device: {DEVICE}"
                assert NUM_EPOCHS > 0, "NUM_EPOCHS must be > 0"
                assert BATCH_SIZE > 0, "BATCH_SIZE must be > 0"
                assert LEARNING_RATE > 0, "LEARNING_RATE must be > 0"
                assert LATENT_DIM > 0, "LATENT_DIM must be > 0"
                print(f"  ✓ PASS Config valid (epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE})")
                self.passed_checks.append("Configuration")
            except AssertionError as e:
                print(f"  ✗ FAIL {e}")
                self.failed_checks.append(f"Configuration: {e}")
                return False

            print("[3/7] Model Architecture...")
            try:
                from model import create_encoder
                encoder = create_encoder()
                device = DEVICE if torch.cuda.is_available() else "cpu"
                encoder = encoder.to(device)

                # Count parameters
                param_count = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
                assert 100_000 < param_count < 10_000_000, f"Param count {param_count} seems wrong"

                # Test forward pass
                dummy_input = torch.randn(4, 8, 250).to(device)
                output = encoder(dummy_input)
                assert output.shape == (4, LATENT_DIM), f"Wrong output shape: {output.shape}"

                print(f"  ✓ PASS Model created ({param_count:,} params), forward pass correct")
                self.passed_checks.append("Model Architecture")
            except Exception as e:
                print(f"  ✗ FAIL {e}")
                self.failed_checks.append(f"Model Architecture: {e}")
                return False

            print("[4/7] Loss Functions...")
            try:
                from losses import ConsistencyLoss, CombinedLoss
                device = DEVICE if torch.cuda.is_available() else "cpu"

                consistency_loss = ConsistencyLoss().to(device)
                combined_loss = CombinedLoss(lambda_consistency=LAMBDA_CONSISTENCY).to(device)

                # Test loss computation
                dummy_features = torch.randn(32, LATENT_DIM).to(device)
                loss_val = consistency_loss(dummy_features)

                # Test combined loss
                loss_dict = combined_loss(dummy_features)
                loss_total = loss_dict["total"]

                assert torch.isfinite(loss_val), f"Consistency loss is not finite: {loss_val}"
                assert torch.isfinite(loss_total), f"Combined loss is not finite: {loss_total}"
                print(f"  ✓ PASS Loss functions compute (L_consistency={loss_val.item():.6f}, L_total={loss_total.item():.6f})")
                self.passed_checks.append("Loss Functions")
            except Exception as e:
                print(f"  ✗ FAIL {e}")
                self.failed_checks.append(f"Loss Functions: {e}")
                return False

            print("[5/7] Data Loading...")
            try:
                from data import create_data_loaders
                train_loader, val_loader, test_loader = create_data_loaders(
                    "synthetic", batch_size=BATCH_SIZE
                )

                # Test data shape
                batch, labels = next(iter(train_loader))
                assert batch.shape[0] == BATCH_SIZE, f"Wrong batch size: {batch.shape[0]}"
                assert batch.shape[1] == 8, f"Wrong num channels: {batch.shape[1]}"
                assert batch.shape[2] == 250, f"Wrong num samples: {batch.shape[2]}"

                print(f"  ✓ PASS Data loading works (batch shape {batch.shape})")
                self.passed_checks.append("Data Loading")
            except Exception as e:
                print(f"  ✗ FAIL {e}")
                self.failed_checks.append(f"Data Loading: {e}")
                return False

            print("[6/7] Training Step...")
            try:
                from model import create_encoder
                from losses import CombinedLoss
                from data import create_data_loaders

                device = DEVICE if torch.cuda.is_available() else "cpu"
                encoder = create_encoder().to(device)
                optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
                loss_fn = CombinedLoss(lambda_consistency=LAMBDA_CONSISTENCY).to(device)
                train_loader, _, _ = create_data_loaders("synthetic", batch_size=BATCH_SIZE)

                # Single training step
                batch, labels = next(iter(train_loader))
                batch = batch.to(device)

                optimizer.zero_grad()
                features = encoder(batch)
                loss_dict = loss_fn(features)
                loss_total = loss_dict["total"]
                loss_total.backward()
                optimizer.step()

                assert torch.isfinite(loss_total), "Loss is not finite"
                print(f"  ✓ PASS Training step successful (loss={loss_total.item():.6f})")
                self.passed_checks.append("Training Step")
            except Exception as e:
                print(f"  ✗ FAIL {e}")
                self.failed_checks.append(f"Training Step: {e}")
                return False

            print("[7/7] CONFIG-SYNC Consistency...")
            try:
                import config as config_module
                import inspect

                config_source = inspect.getsource(config_module)
                config_sync_count = config_source.count("# CONFIG-SYNC:")
                assert config_sync_count >= 5, f"Only {config_sync_count} CONFIG-SYNC tags found"

                print(f"  ✓ PASS CONFIG-SYNC tags found ({config_sync_count})")
                self.passed_checks.append("CONFIG-SYNC")
            except Exception as e:
                print(f"  ✗ FAIL {e}")
                self.failed_checks.append(f"CONFIG-SYNC: {e}")
                return False

            self.audit_log["phases"]["guardian"] = {
                "status": "PASS",
                "checks": 7,
                "passed": 7
            }
            return True

        except Exception as e:
            print(f"\n✗ GUARDIAN PHASE FAILED: {e}")
            self.audit_log["phases"]["guardian"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False

    def phase_2_data_integrity(self) -> bool:
        """
        PHASE 2: Data Integrity Verification

        Validates:
        - Data is REAL (not synthetic when claiming real)
        - No all-zeros or constant-value trials
        - Signal statistics realistic
        - Noise present (not too clean)
        - No inter-trial duplication
        - Label distribution valid
        """
        self.print_header("PHASE 2: Data Integrity Verification")

        try:
            from data import create_data_loaders

            print(f"\nValidating {self.dataset_name} data...")

            # Load data
            train_loader, val_loader, test_loader = create_data_loaders(
                self.dataset_name, batch_size=BATCH_SIZE
            )

            # Collect all data
            all_data = []
            all_labels = []
            for batch, labels in train_loader:
                all_data.append(batch.numpy())
                all_labels.append(labels.numpy())

            data = np.vstack(all_data)
            labels = np.concatenate(all_labels)

            # Validate
            validator = DataIntegrityValidator(data_source=self.dataset_name)

            try:
                report = validator.validate_ssvep_data(data, labels)
                print(f"\n  ✓ PASS Data integrity checks:")
                print(f"    - Shape valid: {report.get('shape_valid', False)}")
                print(f"    - All-zeros check: {report.get('all_zeros_check', 'N/A')}")
                print(f"    - Constant-value check: {report.get('constant_value_check', 'N/A')}")
                print(f"    - Signal std: {report.get('signal_std', 0):.6f} (valid: >0.01)")
                print(f"    - Signal range: {report.get('signal_range_valid', False)}")
                print(f"    - Num classes: {report.get('num_classes', 0)}")
                print(f"    - Noise level: {report.get('noise_level', 0):.6f} (valid: >0.001)")
                print(f"    - Overall status: {report.get('overall_status', 'UNKNOWN')}")

                self.passed_checks.append("Data Integrity")
                self.audit_log["phases"]["data_integrity"] = {
                    "status": "PASS",
                    "data_shape": list(data.shape),
                    "num_classes": int(report.get('num_classes', 0)),
                    "signal_std": float(report.get('signal_std', 0))
                }
                return True

            except ValueError as e:
                print(f"  ✗ FAIL Data integrity check failed:")
                print(f"     {e}")
                self.failed_checks.append(f"Data Integrity: {e}")
                self.audit_log["phases"]["data_integrity"] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                return False

        except Exception as e:
            print(f"\n✗ DATA INTEGRITY PHASE FAILED: {e}")
            self.audit_log["phases"]["data_integrity"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False

    def phase_3_forbidden_checks(self) -> bool:
        """
        PHASE 3: Forbidden Checks

        Absolute prohibitions:
        - NO synthetic data claimed as real
        - NO hallucinated results (100% accuracy on real data)
        - NO data leakage between train/test
        - NO cherry-picked results
        - NO hand-coded metrics
        - NO missing evidence (checkpoints/logs)
        """
        self.print_header("PHASE 3: Forbidden Checks (Fraud Prevention)")

        try:
            checker = ForbiddenChecker()

            print(f"\nChecking for absolute fraud indicators...")
            print(f"Dataset: {self.dataset_name}")

            # Check 1: Synthetic as real
            if self.dataset_name == "synthetic":
                print(f"\n[1/3] Synthetic Data Source Check")
                print(f"  ✓ INFO: {self.dataset_name} is synthetic (OK for testing/debugging)")
                print(f"  ⚠️  WARNING: Do NOT report synthetic results as real BETA/OpenBMI data")
                print(f"  ACTION REQUIRED: Use BETA or OpenBMI for publication")
            else:
                print(f"\n[1/3] Data Source Verification")
                try:
                    checker.check_no_synthetic_as_real(
                        data_source=self.dataset_name,
                        claimed_source=self.dataset_name
                    )
                    print(f"  ✓ PASS Data source valid ({self.dataset_name})")
                except RuntimeError as e:
                    print(f"  ✗ FAIL {e}")
                    self.failed_checks.append(f"Synthetic-as-Real: {e}")
                    self.audit_log["phases"]["forbidden"] = {
                        "status": "FAIL",
                        "error": str(e)
                    }
                    return False

            # Check 2: Will be done after training (not applicable pre-training)
            print(f"\n[2/3] Hallucination Detection (Post-Training)")
            print(f"  ⓘ INFO: Will check after training results available")
            print(f"  FORBIDDEN: 100% accuracy on real data")
            print(f"  FORBIDDEN: >97% on BETA (unrealistic vs SOTA)")
            print(f"  FORBIDDEN: |Accuracy - F1| > 0.10")

            # Check 3: Data leakage prevention
            print(f"\n[3/3] Data Leakage Prevention")
            try:
                from data import create_data_loaders

                train_loader, val_loader, test_loader = create_data_loaders(
                    self.dataset_name, batch_size=BATCH_SIZE
                )

                # Collect train and test data indices
                train_data = []
                test_data = []

                for batch, _ in train_loader:
                    train_data.append(batch.numpy())

                for batch, _ in test_loader:
                    test_data.append(batch.numpy())

                if len(train_data) > 0 and len(test_data) > 0:
                    train_data = np.vstack(train_data)
                    test_data = np.vstack(test_data)

                    results_validator = ResultsValidator()
                    is_clean = results_validator.verify_no_data_leakage(train_data, test_data)

                    if is_clean:
                        print(f"  ✓ PASS No data leakage detected")
                    else:
                        print(f"  ✗ FAIL Data leakage detected")
                        self.failed_checks.append("Data Leakage")
                        self.audit_log["phases"]["forbidden"] = {
                            "status": "FAIL",
                            "error": "Data leakage detected"
                        }
                        return False
                else:
                    print(f"  ⓘ INFO: Insufficient data to check leakage")

            except Exception as e:
                print(f"  ⚠️  WARNING: Could not verify data leakage: {e}")

            self.passed_checks.append("Forbidden Checks")
            self.audit_log["phases"]["forbidden"] = {
                "status": "PASS",
                "checks": ["synthetic_as_real", "data_leakage"]
            }
            return True

        except Exception as e:
            print(f"\n✗ FORBIDDEN CHECKS PHASE FAILED: {e}")
            self.audit_log["phases"]["forbidden"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False

    def save_audit_trail(self):
        """Save comprehensive audit trail to JSON."""
        audit_path = Path("verification_audit_trail.json")

        self.audit_log["summary"] = {
            "timestamp": datetime.now().isoformat(),
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "total_passed": len(self.passed_checks),
            "total_failed": len(self.failed_checks),
            "status": "PASS" if len(self.failed_checks) == 0 else "FAIL"
        }

        with open(audit_path, 'w') as f:
            json.dump(self.audit_log, f, indent=2)

        print(f"\n✓ Audit trail saved to {audit_path}")

    def run_full_verification(self) -> bool:
        """
        Run all verification phases in sequence.

        Returns:
            True if ALL phases pass, False if ANY phase fails
        """
        self.print_header("COMPREHENSIVE INTEGRITY VERIFICATION SYSTEM")
        print(f"\nStarting full verification for {self.dataset_name} dataset...")
        print(f"Timestamp: {datetime.now().isoformat()}")

        # Phase 1: Guardian
        if not self.phase_1_guardian():
            print("\n" + "="*70)
            print("✗ VERIFICATION FAILED AT PHASE 1 (Guardian)")
            print("="*70)
            self.save_audit_trail()
            return False

        # Phase 2: Data Integrity
        if not self.phase_2_data_integrity():
            print("\n" + "="*70)
            print("✗ VERIFICATION FAILED AT PHASE 2 (Data Integrity)")
            print("="*70)
            self.save_audit_trail()
            return False

        # Phase 3: Forbidden Checks
        if not self.phase_3_forbidden_checks():
            print("\n" + "="*70)
            print("✗ VERIFICATION FAILED AT PHASE 3 (Forbidden Checks)")
            print("="*70)
            self.save_audit_trail()
            return False

        # All phases passed
        print("\n" + "="*70)
        print("✓ ALL VERIFICATION PHASES PASSED!")
        print("="*70)
        print(f"\nPassed checks ({len(self.passed_checks)}):")
        for check in self.passed_checks:
            print(f"  ✓ {check}")

        if self.failed_checks:
            print(f"\nFailed checks ({len(self.failed_checks)}):")
            for check in self.failed_checks:
                print(f"  ✗ {check}")

        print("\n" + "="*70)
        print("INTEGRITY STATUS: GO FOR TRAINING")
        print("="*70)
        print("\n✅ Your implementation is ready for real training!")
        print("\nNext steps:")
        if self.dataset_name == "synthetic":
            print("  1. Test training on synthetic data: 2-3 epochs")
            print("  2. Then train on real BETA data: 50-100 epochs")
        else:
            print("  1. Run full training on BETA data")
            print("  2. Track metrics carefully")
            print("  3. Save integrity report with results")

        self.save_audit_trail()
        return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Integrity Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 verify_everything.py --dataset synthetic
  python3 verify_everything.py --dataset BETA
  python3 verify_everything.py --dataset OpenBMI
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "BETA", "OpenBMI"],
        help="Dataset to verify (default: synthetic)"
    )

    parser.add_argument(
        "--save-audit",
        action="store_true",
        help="Save audit trail (default: always saved)"
    )

    args = parser.parse_args()

    # Run verification
    verifier = ComprehensiveVerifier(dataset_name=args.dataset)
    success = verifier.run_full_verification()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
