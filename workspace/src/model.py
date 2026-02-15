"""
Neural Network Models for Conditional Ensemble Averaging

Encoder architecture for learning nonlinear representations
for SSVEP signal denoising and alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    CNN_CHANNELS, CNN_KERNEL_SIZES, CNN_POOLING,
    LATENT_DIM, NUM_CHANNELS, SAMPLES_PER_TRIAL
)


class ConvBlock(nn.Module):
    """Single convolutional block: Conv -> BatchNorm -> ReLU -> MaxPool"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pool_size=2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SSVEPEncoder(nn.Module):
    """
    CNN-based encoder for SSVEP signals.

    Input shape: (batch_size, num_channels, samples_per_trial)
    Output shape: (batch_size, latent_dim)

    This encoder learns to:
    1. Denoise the SSVEP signal
    2. Align trials to a common manifold
    3. Extract discriminative features
    """

    def __init__(
        self,
        num_channels=NUM_CHANNELS,
        samples_per_trial=SAMPLES_PER_TRIAL,
        latent_dim=LATENT_DIM,
        cnn_channels=CNN_CHANNELS,
        kernel_sizes=CNN_KERNEL_SIZES,
        pool_sizes=CNN_POOLING,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.samples_per_trial = samples_per_trial
        self.latent_dim = latent_dim

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()

        in_channels = num_channels
        current_length = samples_per_trial

        for i, out_channels in enumerate(cnn_channels[1:]):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 5
            pool_size = pool_sizes[i] if i < len(pool_sizes) else 2

            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                pool_size=pool_size
            )
            self.conv_layers.append(block)

            in_channels = out_channels
            current_length = current_length // pool_size

        # Calculate flattened size after convolutions
        self.flat_size = in_channels * current_length

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, num_channels, samples_per_trial)

        Returns:
            features: Tensor of shape (batch_size, latent_dim)
        """
        # Pass through conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        features = self.fc(x)

        return features

    def get_feature_dim(self):
        """Get the output feature dimension."""
        return self.latent_dim


class SSVEPClassifier(nn.Module):
    """
    Full pipeline: Encoder + Classifier head.

    Used for supervised baseline and comparison.
    """

    def __init__(self, num_classes=12, encoder=None):
        super().__init__()

        if encoder is None:
            encoder = SSVEPEncoder()

        self.encoder = encoder
        self.classifier = nn.Linear(encoder.latent_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, num_channels, samples_per_trial)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
            features: Tensor of shape (batch_size, latent_dim)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features

    def encode(self, x):
        """Get encoder output without classification."""
        return self.encoder(x)


class SimpleShallowNet(nn.Module):
    """
    Shallow network baseline for quick testing.

    Simple 2-layer CNN for debugging and fast iteration.
    """

    def __init__(self, num_channels=NUM_CHANNELS, samples_per_trial=SAMPLES_PER_TRIAL, latent_dim=64):
        super().__init__()

        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Calculate flattened size
        flattened_size = 64 * (samples_per_trial // 4)

        self.fc = nn.Linear(flattened_size, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_encoder(encoder_type="cnn", **kwargs):
    """Factory function to create encoder."""
    if encoder_type == "cnn":
        return SSVEPEncoder(**kwargs)
    elif encoder_type == "shallow":
        return SimpleShallowNet(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def create_classifier(num_classes=12, encoder_type="cnn", **kwargs):
    """Factory function to create full classifier."""
    encoder = create_encoder(encoder_type, **kwargs)
    return SSVEPClassifier(num_classes=num_classes, encoder=encoder)


# Test code
if __name__ == "__main__":
    # Create dummy input
    batch_size = 4
    num_channels = 8
    samples_per_trial = 250

    x = torch.randn(batch_size, num_channels, samples_per_trial)

    # Test encoder
    encoder = SSVEPEncoder()
    features = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Expected latent dim: {LATENT_DIM}")

    # Test classifier
    classifier = SSVEPClassifier(num_classes=12)
    logits, features = classifier(x)
    print(f"\nClassifier logits shape: {logits.shape}")
    print(f"Classifier features shape: {features.shape}")

    # Test shallow network
    shallow = SimpleShallowNet()
    shallow_features = shallow(x)
    print(f"\nShallow net features shape: {shallow_features.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nEncoder total parameters: {total_params:,}")
    print(f"Encoder trainable parameters: {trainable_params:,}")
