"""
Neural Conditional Ensemble Averaging for SSVEP

Main package for implementing learnable ensemble averaging with consistency
regularization for brain-computer interface applications.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .config import *
from .model import create_encoder, create_classifier, SSVEPEncoder
from .losses import ConsistencyLoss, TemplateLoss, CombinedLoss
from .data import create_data_loaders, SSVEPDataset
from .metrics import compute_accuracy, compute_f1, compute_itr
from .train import Trainer
from .evaluate import Evaluator

__all__ = [
    "create_encoder",
    "create_classifier",
    "SSVEPEncoder",
    "ConsistencyLoss",
    "TemplateLoss",
    "CombinedLoss",
    "create_data_loaders",
    "SSVEPDataset",
    "compute_accuracy",
    "compute_f1",
    "compute_itr",
    "Trainer",
    "Evaluator",
]
