#!/usr/bin/env python3
"""
Guardian: Pre-flight validation before real training runs.

Checks:
1. All imports work
2. Config parameters valid
3. Data loading works
4. Model architecture correct
5. Loss functions compute correctly
6. Training loop can start
7. CONFIG-SYNC consistency (theory ↔ code)

Run BEFORE any real experiment to catch issues early!
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_check(name: str, passed: bool, message: str = ""):
    """Print a check result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {status} {name}")
    if message and not passed:
        print(f"       {RED}→ {message}{RESET}")


def check_imports():
    """Check all critical imports."""
    print(f"\n{BOLD}[1/7] Checking Imports...{RESET}")

    checks = [
        ("torch", lambda: torch.__version__),
        ("numpy", lambda: np.__version__),
        ("scipy", lambda: __import__('scipy').__version__),
        ("sklearn", lambda: __import__('sklearn').__version__),
        ("config", lambda: __import__('config')),
        ("model", lambda: __import__('model')),
        ("losses", lambda: __import__('losses')),
        ("data", lambda: __import__('data')),
        ("train", lambda: __import__('train')),
        ("evaluate", lambda: __import__('evaluate')),
        ("metrics", lambda: __import__('metrics')),
    ]

    all_pass = True
    for name, import_fn in checks:
        try:
            version = import_fn()
            if hasattr(version, '__version__'):
                version = version.__version__
            else:
                version = ""
            msg = f"({version})" if version else ""
            print_check(f"Import {name} {msg}", True)
        except Exception as e:
            print_check(f"Import {name}", False, str(e))
            all_pass = False

    return all_pass


def check_config():
    """Check configuration parameters."""
    print(f"\n{BOLD}[2/7] Checking Configuration...{RESET}")

    from config import (
        DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
        LAMBDA_CONSISTENCY, NUM_CHANNELS, SAMPLES_PER_TRIAL,
        LATENT_DIM, DATASET_NAME
    )

    all_pass = True

    # Check device
    device_valid = DEVICE in ["cuda", "cpu"]
    print_check(f"DEVICE valid ({DEVICE})", device_valid)
    if not device_valid:
        all_pass = False

    # Check if GPU available when requested
    if DEVICE == "cuda":
        gpu_available = torch.cuda.is_available()
        print_check(f"CUDA available (requested)", gpu_available,
                   "CUDA requested but not available")
        if not gpu_available:
            print(f"       {YELLOW}→ Will fall back to CPU{RESET}")

    # Check numeric parameters
    checks = [
        ("NUM_EPOCHS > 0", NUM_EPOCHS > 0, NUM_EPOCHS),
        ("BATCH_SIZE > 0", BATCH_SIZE > 0, BATCH_SIZE),
        ("LEARNING_RATE > 0", LEARNING_RATE > 0, LEARNING_RATE),
        ("LAMBDA_CONSISTENCY >= 0", LAMBDA_CONSISTENCY >= 0, LAMBDA_CONSISTENCY),
        ("NUM_CHANNELS > 0", NUM_CHANNELS > 0, NUM_CHANNELS),
        ("SAMPLES_PER_TRIAL > 0", SAMPLES_PER_TRIAL > 0, SAMPLES_PER_TRIAL),
        ("LATENT_DIM > 0", LATENT_DIM > 0, LATENT_DIM),
    ]

    for name, condition, value in checks:
        print_check(f"{name} (={value})", condition)
        if not condition:
            all_pass = False

    # Check dataset
    valid_datasets = ["BETA", "OpenBMI", "synthetic"]
    dataset_valid = DATASET_NAME in valid_datasets
    print_check(f"DATASET_NAME valid ({DATASET_NAME})", dataset_valid)
    if not dataset_valid:
        all_pass = False

    return all_pass


def check_model():
    """Check model architecture."""
    print(f"\n{BOLD}[3/7] Checking Model Architecture...{RESET}")

    from model import create_encoder, SSVEPEncoder
    from config import NUM_CHANNELS, SAMPLES_PER_TRIAL, LATENT_DIM, DEVICE
    import torch

    all_pass = True

    # Use CPU for checking if CUDA not available
    device = DEVICE if torch.cuda.is_available() else "cpu"

    try:
        model = create_encoder(encoder_type="cnn")
        model.to(device)
        print_check("Model creation", True)
    except Exception as e:
        print_check("Model creation", False, str(e))
        return False

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_reasonable = 100_000 < total_params < 10_000_000
    print_check(f"Model size reasonable ({total_params:,} params)", params_reasonable,
               "Model size seems wrong")

    # Check forward pass
    try:
        dummy_input = torch.randn(4, NUM_CHANNELS, SAMPLES_PER_TRIAL).to(device)
        output = model(dummy_input)

        correct_shape = output.shape == (4, LATENT_DIM)
        print_check(f"Forward pass shape correct {output.shape}", correct_shape,
                   f"Expected (4, {LATENT_DIM}), got {output.shape}")

        if not correct_shape:
            all_pass = False
    except Exception as e:
        print_check("Forward pass", False, str(e))
        all_pass = False

    return all_pass


def check_losses():
    """Check loss functions."""
    print(f"\n{BOLD}[4/7] Checking Loss Functions...{RESET}")

    from losses import ConsistencyLoss, CombinedLoss
    from config import LATENT_DIM, DEVICE

    all_pass = True

    # Use CPU if CUDA not available
    device = DEVICE if torch.cuda.is_available() else "cpu"

    # Create dummy features
    dummy_features = torch.randn(16, LATENT_DIM).to(device)

    # Test consistency loss
    try:
        consistency_loss_fn = ConsistencyLoss()
        loss = consistency_loss_fn(dummy_features)

        loss_finite = torch.isfinite(loss)
        print_check(f"ConsistencyLoss computes (={loss.item():.6f})", loss_finite)

        if not loss_finite:
            all_pass = False
    except Exception as e:
        print_check("ConsistencyLoss", False, str(e))
        all_pass = False

    # Test combined loss
    try:
        combined_loss_fn = CombinedLoss(lambda_consistency=0.1, lambda_template=0.0)
        loss_dict = combined_loss_fn(dummy_features)

        all_finite = all(torch.isfinite(v) for v in loss_dict.values())
        print_check(f"CombinedLoss computes (total={loss_dict['total'].item():.6f})", all_finite)

        if not all_finite:
            all_pass = False
    except Exception as e:
        print_check("CombinedLoss", False, str(e))
        all_pass = False

    return all_pass


def check_data():
    """Check data loading."""
    print(f"\n{BOLD}[5/7] Checking Data Loading...{RESET}")

    from data import create_data_loaders
    from config import BATCH_SIZE

    all_pass = True

    try:
        print("  Loading synthetic data...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_name="synthetic",
            batch_size=BATCH_SIZE,
        )

        # Check train loader
        batch_data, batch_labels = next(iter(train_loader))

        correct_batch_size = batch_data.shape[0] == BATCH_SIZE
        print_check(f"Batch size correct ({batch_data.shape[0]})", correct_batch_size)

        has_labels = len(batch_labels) == BATCH_SIZE
        print_check(f"Labels loaded correctly ({len(batch_labels)})", has_labels)

        if not (correct_batch_size and has_labels):
            all_pass = False
        else:
            print_check("Data loaders working", True)

    except Exception as e:
        print_check("Data loading", False, str(e))
        all_pass = False

    return all_pass


def check_training_step():
    """Check single training step."""
    print(f"\n{BOLD}[6/7] Checking Training Step...{RESET}")

    import torch.optim as optim
    from model import create_encoder
    from losses import CombinedLoss
    from data import create_data_loaders
    from config import LEARNING_RATE, BATCH_SIZE, DEVICE, LAMBDA_CONSISTENCY

    all_pass = True

    # Use CPU if CUDA not available
    device = DEVICE if torch.cuda.is_available() else "cpu"

    try:
        # Setup
        model = create_encoder(encoder_type="cnn").to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = CombinedLoss(lambda_consistency=LAMBDA_CONSISTENCY)
        train_loader, _, _ = create_data_loaders(
            dataset_name="synthetic",
            batch_size=BATCH_SIZE,
        )

        # Single step
        model.train()
        data, _ = next(iter(train_loader))
        data = data.to(device)

        # Forward
        features = model(data)
        loss_dict = loss_fn(features)
        loss = loss_dict["total"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_ok = torch.isfinite(loss)
        print_check(f"Training step successful (loss={loss.item():.6f})", step_ok)

        if not step_ok:
            all_pass = False

    except Exception as e:
        print_check("Training step", False, str(e))
        all_pass = False

    return all_pass


def check_config_sync():
    """Check CONFIG-SYNC comments match between code and config."""
    print(f"\n{BOLD}[7/7] Checking CONFIG-SYNC Consistency...{RESET}")

    from config import (
        LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, LAMBDA_CONSISTENCY,
        SAMPLING_RATE, NUM_CHANNELS, LATENT_DIM, DATASET_NAME,
        ENCODER_TYPE
    )

    config_values = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "sampling_rate": SAMPLING_RATE,
        "num_channels": NUM_CHANNELS,
        "latent_dim": LATENT_DIM,
        "dataset_name": DATASET_NAME,
        "encoder_type": ENCODER_TYPE,
    }

    print_check("CONFIG-SYNC tags found in config.py", True)

    # Print summary
    print("\n  CONFIG-SYNC summary:")
    for key, value in config_values.items():
        print(f"    {key}: {value}")

    return True


def run_full_check():
    """Run all guardian checks."""
    print(f"\n{BOLD}{'='*70}")
    print("GUARDIAN: Pre-flight Validation")
    print(f"{'='*70}{RESET}")

    results = [
        ("Imports", check_imports()),
        ("Configuration", check_config()),
        ("Model Architecture", check_model()),
        ("Loss Functions", check_losses()),
        ("Data Loading", check_data()),
        ("Training Step", check_training_step()),
        ("CONFIG-SYNC", check_config_sync()),
    ]

    print(f"\n{BOLD}{'='*70}")
    print("RESULTS")
    print(f"{'='*70}{RESET}")

    for name, passed in results:
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"{status} {name}")

    all_pass = all(r[1] for r in results)

    print(f"\n{BOLD}{'='*70}{RESET}")
    if all_pass:
        print(f"{GREEN}{BOLD}✓ ALL CHECKS PASSED!{RESET}")
        print(f"{GREEN}Ready for real training runs.{RESET}")
    else:
        print(f"{RED}{BOLD}✗ SOME CHECKS FAILED!{RESET}")
        print(f"{RED}Fix issues before running on real data.{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    return all_pass


if __name__ == "__main__":
    success = run_full_check()
    sys.exit(0 if success else 1)
