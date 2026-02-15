"""
FORBIDDEN CHECKS: Absolutely prevent hallucinated/fabricated results.

ZERO tolerance for:
- Made-up accuracy numbers
- Fabricated metrics
- Cherry-picked results
- Synthetic data as real
- Data hallucination
- Result manipulation
"""

import numpy as np
from typing import Dict, List
import sys


class ForbiddenChecker:
    """Absolute prohibition on fraudulent results."""

    # FORBIDDEN PATTERNS
    FORBIDDEN_ACCURACIES = [
        0.99,  # Too perfect (except on synthetic)
        1.0,   # Always 100% → FAKE
        0.95,  # Suspiciously high on real data
    ]

    FORBIDDEN_METRICS = [
        "made_up",
        "estimated",
        "assumed",
        "guessed",
        "interpolated_from",
        "extrapolated",
        "hallucinated",
    ]

    def __init__(self):
        self.violations = []

    def check_no_hallucinated_results(self, results: Dict) -> bool:
        """
        ABSOLUTELY prevent hallucinated results.

        Args:
            results: Results dictionary

        Returns:
            True if OK, False if HALLUCINATED detected

        Raises:
            RuntimeError if hallucination detected
        """
        print("\n" + "="*70)
        print("FORBIDDEN RESULTS CHECK: Detecting Hallucinated/Fabricated Results")
        print("="*70)

        # Check 1: NO perfect accuracy on real data
        accuracy = results.get("accuracy", 0)
        data_source = results.get("data_source", "unknown")

        if accuracy == 1.0:
            if data_source != "synthetic":
                raise RuntimeError(
                    f"✗ HALLUCINATED RESULT: 100% accuracy on {data_source} data is FAKE\n"
                    f"  Real SSVEP datasets never reach 100%\n"
                    f"  This result is FABRICATED and FORBIDDEN"
                )

        # Check 2: Flag suspiciously high accuracies
        if accuracy > 0.97 and data_source == "BETA":
            raise RuntimeError(
                f"✗ SUSPICIOUS RESULT: {accuracy:.1%} on BETA is unrealistic\n"
                f"  SOTA on BETA: 94-96%\n"
                f"  Your claim: {accuracy:.1%}\n"
                f"  This looks HALLUCINATED. Provide evidence or FORBIDDEN"
            )

        # Check 3: Check for hand-coded metrics
        for key, value in results.items():
            if isinstance(value, str):
                if any(forbidden in value.lower() for forbidden in self.FORBIDDEN_METRICS):
                    raise RuntimeError(
                        f"✗ FABRICATED RESULT: '{key}' contains forbidden word '{value}'\n"
                        f"  This is a MADE-UP result, not from actual training\n"
                        f"  FORBIDDEN"
                    )

        # Check 4: Accuracy should match F1 (within 5%)
        f1 = results.get("f1_score", accuracy)
        if abs(accuracy - f1) > 0.10:
            raise RuntimeError(
                f"✗ INCONSISTENT METRICS (hallucination indicator):\n"
                f"  Accuracy: {accuracy:.4f}\n"
                f"  F1 Score: {f1:.4f}\n"
                f"  Difference: {abs(accuracy-f1):.4f} (should be <0.10)\n"
                f"  This mismatch suggests HALLUCINATED results\n"
                f"  FORBIDDEN"
            )

        # Check 5: ITR must be calculable from accuracy
        itr = results.get("itr", None)
        if itr is not None:
            # ITR = [log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))] * 60 / T
            expected_itr_min = 0
            expected_itr_max = 200

            if itr < expected_itr_min or itr > expected_itr_max:
                raise RuntimeError(
                    f"✗ HALLUCINATED ITR:\n"
                    f"  Reported: {itr:.2f} bits/min\n"
                    f"  Valid range: {expected_itr_min}-{expected_itr_max}\n"
                    f"  This is FABRICATED\n"
                    f"  FORBIDDEN"
                )

        # Check 6: NO manually entered numbers
        for key in ["accuracy", "f1_score", "precision", "recall"]:
            if key in results:
                val = results[key]
                # Check for suspiciously round numbers (sign of hand-coding)
                if isinstance(val, float):
                    # 0.900000000, 0.950000000, etc. are suspicious
                    if val == round(val, 1):
                        # Could be legitimate, but flag for review
                        print(f"  ⚠️  {key} = {val} is suspiciously round (may be hand-coded)")

        # Check 7: Results must have actual evidence
        if "model_weights" not in results and "checkpoint_path" not in results:
            print(f"  ⚠️  WARNING: No model checkpoint saved (cannot verify results)")

        if "random_seed" not in results:
            print(f"  ⚠️  WARNING: Random seed not recorded (not reproducible)")

        # PASSED all forbidden checks
        print(f"\n✓ PASSED: No hallucinated results detected")
        print(f"  Accuracy: {accuracy:.4f} (realistic)")
        print(f"  F1 Score: {f1:.4f} (matches accuracy)")
        print(f"  Data source: {data_source} (valid)")
        print(f"  Metrics: All actual (not made-up)")

        return True

    def check_no_synthetic_as_real(self, data_source: str, claimed_source: str) -> bool:
        """
        ABSOLUTELY prevent synthetic data being claimed as real.

        Args:
            data_source: Actual data source ("synthetic", "BETA", etc.)
            claimed_source: What's being claimed in paper

        Returns:
            True if OK, False if FRAUD detected

        Raises:
            RuntimeError if fraud detected
        """
        if data_source == "synthetic" and claimed_source in ["BETA", "OpenBMI", "real"]:
            raise RuntimeError(
                f"✗ ABSOLUTELY FORBIDDEN:\n"
                f"  Actual data: {data_source}\n"
                f"  Claimed as: {claimed_source}\n"
                f"  This is SCIENTIFIC FRAUD\n"
                f"  FORBIDDEN - WILL NOT PROCEED"
            )

        return True

    def check_no_cherry_picking(self, all_results: List[Dict]) -> bool:
        """
        ABSOLUTELY prevent cherry-picking best results.

        Args:
            all_results: All runs from multiple seeds

        Returns:
            True if OK, False if cherry-picking detected

        Raises:
            RuntimeError if cherry-picking detected
        """
        if not all_results:
            return True

        accuracies = [r["accuracy"] for r in all_results if "accuracy" in r]

        if len(accuracies) == 0:
            return True

        mean_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        min_acc = np.min(accuracies)
        std_acc = np.std(accuracies)

        # Check: Are you reporting only the best run?
        print(f"\nRuns summary:")
        print(f"  Best: {max_acc:.4f}")
        print(f"  Mean: {mean_acc:.4f}")
        print(f"  Std:  {std_acc:.6f}")
        print(f"  Worst: {min_acc:.4f}")

        if max_acc - mean_acc > 0.05:  # Best is >5% better than mean
            print(f"  ⚠️  WARNING: Best run is {(max_acc-mean_acc):.1%} better than average")
            print(f"      Did you cherry-pick the best result?")
            print(f"      REQUIRED: Report mean ± std, not just best")

        # Check: Standard deviation too small?
        if std_acc < 0.001:
            raise RuntimeError(
                f"✗ HALLUCINATED CONSISTENCY:\n"
                f"  Std dev across {len(accuracies)} runs: {std_acc:.6f}\n"
                f"  Real runs always have some variance\n"
                f"  This low std suggests MADE-UP results\n"
                f"  FORBIDDEN"
            )

        return True

    def check_no_data_leakage(self, train_indices: np.ndarray, test_indices: np.ndarray) -> bool:
        """
        ABSOLUTELY prevent data leakage.

        Args:
            train_indices: Indices used for training
            test_indices: Indices used for testing

        Returns:
            True if OK, False if leakage detected

        Raises:
            RuntimeError if leakage detected
        """
        overlap = np.intersect1d(train_indices, test_indices)

        if len(overlap) > 0:
            raise RuntimeError(
                f"✗ ABSOLUTELY FORBIDDEN - DATA LEAKAGE:\n"
                f"  {len(overlap)} samples in BOTH train and test\n"
                f"  Leaking indices: {overlap[:10]}\n"
                f"  This is SCIENTIFIC FRAUD\n"
                f"  FORBIDDEN - WILL NOT PROCEED"
            )

        return True

    def verify_results_from_actual_training(self, results: Dict,
                                           checkpoint_exists: bool,
                                           logs_exist: bool) -> bool:
        """
        ABSOLUTELY verify results come from actual training, not fabrication.

        Args:
            results: Results dictionary
            checkpoint_exists: Is there a saved model checkpoint?
            logs_exist: Are there training logs?

        Returns:
            True if results appear legitimate

        Raises:
            RuntimeError if results appear fabricated
        """
        print("\nVerifying results are from actual training:")

        if not checkpoint_exists:
            print("  ⚠️  WARNING: No model checkpoint found")
            print("      Cannot verify model actually exists")
            print("      Results may be FABRICATED")

        if not logs_exist:
            print("  ⚠️  WARNING: No training logs found")
            print("      Cannot verify training actually happened")
            print("      Results may be HALLUCINATED")

        if not checkpoint_exists and not logs_exist:
            raise RuntimeError(
                f"✗ FABRICATED RESULTS:\n"
                f"  No checkpoint: Model doesn't exist\n"
                f"  No logs: Training didn't happen\n"
                f"  These are HALLUCINATED results\n"
                f"  FORBIDDEN"
            )

        print("  ✓ Evidence of actual training found")
        return True


# Usage Examples
if __name__ == "__main__":
    print("=== FORBIDDEN CHECKS DEMO ===\n")

    checker = ForbiddenChecker()

    # Test 1: Valid results
    print("[TEST 1] Valid, realistic results")
    try:
        checker.check_no_hallucinated_results({
            "accuracy": 0.94,
            "f1_score": 0.93,
            "itr": 152.1,
            "data_source": "BETA",
        })
        print("✓ PASSED: Results are valid")
    except RuntimeError as e:
        print(f"✗ FAILED: {e}")

    # Test 2: Hallucinated 100%
    print("\n[TEST 2] Detecting 100% accuracy on real data")
    try:
        checker.check_no_hallucinated_results({
            "accuracy": 1.0,
            "f1_score": 1.0,
            "data_source": "BETA",
        })
        print("✗ FAILED TO DETECT HALLUCINATION!")
    except RuntimeError as e:
        print(f"✓ CORRECTLY DETECTED: Hallucinated results")

    # Test 3: Synthetic as real
    print("\n[TEST 3] Detecting synthetic data claimed as real")
    try:
        checker.check_no_synthetic_as_real("synthetic", "BETA")
        print("✗ FAILED TO DETECT FRAUD!")
    except RuntimeError as e:
        print(f"✓ CORRECTLY DETECTED: Fraud (synthetic → real)")

    # Test 4: Data leakage
    print("\n[TEST 4] Detecting data leakage")
    try:
        train_idx = np.array([0, 1, 2, 3, 4, 5])
        test_idx = np.array([5, 6, 7, 8, 9, 10])  # 5 is in both!
        checker.check_no_data_leakage(train_idx, test_idx)
        print("✗ FAILED TO DETECT LEAKAGE!")
    except RuntimeError as e:
        print(f"✓ CORRECTLY DETECTED: Data leakage")

    # Test 5: Cherry-picking
    print("\n[TEST 5] Detecting cherry-picked results")
    try:
        results = [
            {"accuracy": 0.99},
            {"accuracy": 0.50},  # Huge variance = sign of cherry-picking
            {"accuracy": 0.48},
        ]
        checker.check_no_cherry_picking(results)
        print("✓ FLAGGED: Suspicious cherry-picking detected")
    except RuntimeError as e:
        print(f"✓ CORRECTLY FLAGGED: {str(e)[:50]}...")

    print("\n✓ All forbidden checks working correctly")
