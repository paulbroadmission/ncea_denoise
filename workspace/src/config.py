"""
Configuration for Neural Conditional Ensemble Averaging

All hyperparameters and settings defined here.
CONFIG-SYNC comments enable automated checking with LaTeX paper.
"""

import os

# ===== Device Configuration =====
DEVICE = "cuda"  # "cuda" or "cpu"
NUM_WORKERS = 4

# ===== Model Configuration =====
# CONFIG-SYNC: encoder_type = "cnn"
ENCODER_TYPE = "cnn"  # "cnn" or "attention"

# CONFIG-SYNC: latent_dim = 80
LATENT_DIM = 80  # Feature dimension D

# CNN Architecture
CNN_CHANNELS = [1, 32, 32, 64]  # Input channels -> hidden layers
CNN_KERNEL_SIZES = [5, 5, 5]
CNN_STRIDES = [1, 1, 1]
CNN_POOLING = [2, 2, 2]  # Max pool after each conv layer

# ===== Training Configuration =====
# CONFIG-SYNC: learning_rate = 1e-3
LEARNING_RATE = 1e-3

# CONFIG-SYNC: batch_size = 32
BATCH_SIZE = 32

# CONFIG-SYNC: num_epochs = 100
NUM_EPOCHS = 100

# CONFIG-SYNC: lambda_consistency = 0.1
LAMBDA_CONSISTENCY = 0.1  # Weight for L_consistency in total loss

OPTIMIZER = "adam"  # "adam" or "sgd"
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5

# Learning rate schedule
LR_SCHEDULER = "cosine"  # "cosine", "linear", or "constant"
WARMUP_EPOCHS = 5

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_METRIC = "val_accuracy"  # or "val_loss"

# ===== Data Configuration =====
# CONFIG-SYNC: dataset_name = "BETA"
DATASET_NAME = "BETA"  # "BETA", "OpenBMI", or "synthetic"

# For BETA dataset
NUM_SUBJECTS_BETA = 10
NUM_CLASSES_BETA = 12  # 12 SSVEP frequencies
TRIAL_LENGTH = 1  # seconds

# CONFIG-SYNC: sampling_rate = 250
SAMPLING_RATE = 250  # Hz

# Derived: samples per trial
SAMPLES_PER_TRIAL = int(TRIAL_LENGTH * SAMPLING_RATE)  # 250

# CONFIG-SYNC: num_channels = 8
NUM_CHANNELS = 8  # Number of EEG electrodes

# Data preprocessing
BANDPASS_LOW = 5  # Hz
BANDPASS_HIGH = 100  # Hz
NOTCH_FREQ = 50  # Hz (power line frequency)

# Train/val/test split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ===== Loss Configuration =====
LOSS_TYPE = "consistency"  # "consistency" only, as decided

# ===== Evaluation Configuration =====
# CONFIG-SYNC: evaluation_metric = "accuracy"
EVALUATION_METRICS = ["accuracy", "f1_weighted", "itr"]  # Information Transfer Rate

# Confusion matrix visualization
PLOT_CONFUSION_MATRIX = True

# ===== Experiment Configuration =====
NUM_RANDOM_SEEDS = 3  # Run multiple seeds for robustness
RANDOM_SEEDS = [42, 123, 456]

# Cross-subject evaluation
CROSS_SUBJECT_EVAL = True

# ===== Paths =====
PROJECT_ROOT = "/Users/paul/prj/GenAI/vibe/paper_machine/auto-research-fleet"
WORKSPACE_ROOT = os.path.join(PROJECT_ROOT, "workspace")
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")
CHECKPOINT_ROOT = os.path.join(WORKSPACE_ROOT, "checkpoints")
RESULTS_ROOT = os.path.join(WORKSPACE_ROOT, "results")
LOG_ROOT = os.path.join(WORKSPACE_ROOT, "logs")

# Create directories if they don't exist
for d in [DATA_ROOT, CHECKPOINT_ROOT, RESULTS_ROOT, LOG_ROOT]:
    os.makedirs(d, exist_ok=True)

# ===== Logging =====
LOG_LEVEL = "INFO"
LOG_INTERVAL = 10  # Log every N batches

# ===== Ablation Studies =====
ABLATION_CONFIGS = {
    "baseline_cnn": {
        "name": "CNN baseline (no consistency)",
        "lambda_consistency": 0.0,
        "description": "Pure CNN with no consistency regularization"
    },
    "consistency_only": {
        "name": "Consistency only",
        "lambda_consistency": 0.1,
        "description": "L_consistency with default lambda"
    },
    "consistency_light": {
        "name": "Consistency light (λ=0.01)",
        "lambda_consistency": 0.01,
        "description": "Weak consistency regularization"
    },
    "consistency_heavy": {
        "name": "Consistency heavy (λ=1.0)",
        "lambda_consistency": 1.0,
        "description": "Strong consistency regularization"
    },
}

# ===== Baseline Configurations =====
BASELINE_CONFIGS = {
    "trca": {
        "name": "TRCA (from MEEGkit)",
        "description": "Task-Related Component Analysis - classical baseline"
    },
    "cnn": {
        "name": "CNN (no TRCA)",
        "description": "Standard CNN classification without TRCA preprocessing"
    },
    "trca_cnn": {
        "name": "TRCA + CNN (Li et al. 2024)",
        "description": "TRCA preprocessing followed by CNN"
    },
}

# ===== Debug Mode =====
DEBUG = False
DEBUG_BATCH_SIZE = 4
DEBUG_NUM_EPOCHS = 2
DEBUG_NUM_SUBJECTS = 1
