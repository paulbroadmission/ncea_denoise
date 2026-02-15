#!/usr/bin/env python
"""
Main entry point for Neural Conditional Ensemble Averaging experiments.

Usage:
    python main.py --mode train --dataset synthetic
    python main.py --mode train --dataset BETA
    python main.py --mode test --checkpoint best_model.pt
    python main.py --mode ablation
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LAMBDA_CONSISTENCY,
    DEBUG, CHECKPOINT_ROOT, RESULTS_ROOT
)
from data import create_data_loaders
from model import create_encoder, create_classifier
from train import Trainer
from evaluate import Evaluator
from baselines import TraditionalTRCA, CNNBaseline


def main():
    parser = argparse.ArgumentParser(
        description="Neural Conditional Ensemble Averaging for SSVEP"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "ablation", "compare"],
        default="train",
        help="Execution mode"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["BETA", "OpenBMI", "synthetic"],
        default="synthetic" if DEBUG else "BETA",
        help="Dataset to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint file to load for testing"
    )
    parser.add_argument(
        "--lambda-consistency",
        type=float,
        default=LAMBDA_CONSISTENCY,
        help="Weight for consistency loss"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4 if DEBUG else BATCH_SIZE,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2 if DEBUG else NUM_EPOCHS,
        help="Number of epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=DEVICE,
        help="Device to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (smaller data, fewer epochs)"
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    print("=" * 80)
    print("Neural Conditional Ensemble Averaging for SSVEP")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Lambda consistency: {args.lambda_consistency}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80 + "\n")

    if args.mode == "train":
        train_model(
            dataset=args.dataset,
            lambda_consistency=args.lambda_consistency,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=args.device,
            seed=args.seed,
        )

    elif args.mode == "test":
        test_model(
            dataset=args.dataset,
            checkpoint_path=Path(CHECKPOINT_ROOT) / args.checkpoint,
            device=args.device,
        )

    elif args.mode == "ablation":
        run_ablation_studies(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=args.device,
        )

    elif args.mode == "compare":
        compare_methods(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=args.device,
        )


def train_model(
    dataset: str,
    lambda_consistency: float,
    batch_size: int,
    num_epochs: int,
    device: str,
    seed: int,
):
    """Train the neural conditional ensemble averaging model."""
    print("[1/3] Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=dataset,
        batch_size=batch_size,
    )

    print("[2/3] Creating model...")
    model = create_encoder(encoder_type="cnn")

    print("[3/3] Training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lambda_consistency=lambda_consistency,
        num_epochs=num_epochs,
        device=device,
    )

    history = trainer.train()

    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"  Best epoch: {history['best_epoch']}")

    # Evaluate on test set
    print("\n[Evaluating on test set...]")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
    )
    test_metrics = evaluator.evaluate()

    print(f"Test set metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")


def test_model(
    dataset: str,
    checkpoint_path: Path,
    device: str,
):
    """Test a trained model."""
    print(f"[Loading model from {checkpoint_path}...]")

    _, _, test_loader = create_data_loaders(dataset_name=dataset)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_encoder(encoder_type="cnn")
    model.load_state_dict(checkpoint["model_state"])

    # Evaluate
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
    )
    metrics = evaluator.evaluate()

    print(f"\nTest metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def run_ablation_studies(
    dataset: str,
    batch_size: int,
    num_epochs: int,
    device: str,
):
    """Run ablation studies comparing different lambda values."""
    print("Running ablation studies: varying lambda_consistency\n")

    lambda_values = [0.0, 0.01, 0.1, 0.5, 1.0]
    results = {}

    for lambda_val in lambda_values:
        print(f"\n{'='*60}")
        print(f"Running with lambda_consistency = {lambda_val}")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_name=dataset,
            batch_size=batch_size,
        )

        model = create_encoder(encoder_type="cnn")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lambda_consistency=lambda_val,
            num_epochs=num_epochs,
            device=device,
        )

        history = trainer.train()
        results[lambda_val] = {
            "best_val_accuracy": history["best_val_accuracy"],
            "best_epoch": history["best_epoch"],
        }

    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Lambda':<10} {'Best Val Acc':<15} {'Best Epoch':<10}")
    print("-"*60)
    for lambda_val, metrics in sorted(results.items()):
        print(
            f"{lambda_val:<10.3f} {metrics['best_val_accuracy']:<15.4f} "
            f"{metrics['best_epoch']:<10}"
        )


def compare_methods(
    dataset: str,
    batch_size: int,
    num_epochs: int,
    device: str,
):
    """Compare different methods: TRCA, CNN, and proposed method."""
    print("Comparing methods: TRCA vs CNN vs Proposed\n")

    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=dataset,
        batch_size=batch_size,
    )

    results = {}

    # Method 1: TRCA
    print(f"\n{'='*60}")
    print("Method 1: TRCA (Classical)")
    print(f"{'='*60}")
    trca_model = TraditionalTRCA()
    # trca_metrics = trca_model.evaluate(train_loader, test_loader)
    # results["TRCA"] = trca_metrics
    print("[TRCA evaluation: TODO]")

    # Method 2: CNN (no consistency)
    print(f"\n{'='*60}")
    print("Method 2: CNN Baseline")
    print(f"{'='*60}")
    cnn_model = create_encoder(encoder_type="cnn")
    trainer = Trainer(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lambda_consistency=0.0,  # No consistency term
        num_epochs=num_epochs,
        device=device,
    )
    history = trainer.train()
    results["CNN"] = {"accuracy": history["best_val_accuracy"]}

    # Method 3: Proposed (with consistency)
    print(f"\n{'='*60}")
    print("Method 3: Proposed (Consistency + Manifold)")
    print(f"{'='*60}")
    proposed_model = create_encoder(encoder_type="cnn")
    trainer = Trainer(
        model=proposed_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lambda_consistency=0.1,  # With consistency term
        num_epochs=num_epochs,
        device=device,
    )
    history = trainer.train()
    results["Proposed"] = {"accuracy": history["best_val_accuracy"]}

    # Print comparison
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    print(f"{'Method':<20} {'Accuracy':<15}")
    print("-"*60)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['accuracy']:<15.4f}")


if __name__ == "__main__":
    main()
