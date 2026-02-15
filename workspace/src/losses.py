"""
Loss functions for Neural Conditional Ensemble Averaging

Implements:
1. L_consistency: U-statistic of inter-trial feature distances
2. L_template: Template matching (multiple variants)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    L_consistency: U-statistic of inter-trial distances.

    Minimizes the average pairwise distance between trial features:
        L_consistency = 1/(N(N-1)) * Σ_{h≠k} ||f_θ(y_h) - f_θ(y_k)||²

    This encourages all trials from the same condition to map to similar
    features, implementing manifold regularization.

    Args:
        reduction: 'mean' (default) or 'sum'
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, features):
        """
        Compute consistency loss.

        Args:
            features: Tensor of shape (batch_size, latent_dim)
                    or (batch_size * num_trials, latent_dim)
                    representing encoded features from multiple trials

        Returns:
            loss: Scalar tensor, the consistency loss

        Mathematical form:
            L = (1 / (N(N-1))) * Σ_{h≠k} d(f_h, f_k)
            where d(·,·) is squared Euclidean distance
        """
        batch_size = features.size(0)

        # Compute pairwise distances efficiently using broadcasting
        # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2x_i·x_j
        features_sq = (features ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1)
        distances_sq = features_sq + features_sq.t() - 2 * torch.matmul(features, features.t())

        # Clamp to avoid numerical issues
        distances_sq = torch.clamp(distances_sq, min=1e-8)

        # Set diagonal to 0 (don't compare each sample with itself)
        mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        distances_sq = distances_sq.masked_fill(mask, 0)

        # Sum all pairwise distances
        loss = distances_sq.sum() / (batch_size * (batch_size - 1))

        return loss


class TemplateLoss(nn.Module):
    """
    L_template: Template matching loss.

    Multiple variants to avoid the trivial loss issue.
    """

    def __init__(self, variant="learnable_template", feature_dim=80):
        """
        Args:
            variant: 'learnable_template', 'contrastive', or 'none'
            feature_dim: Dimension of learned template (if learnable)
        """
        super().__init__()
        self.variant = variant

        if variant == "learnable_template":
            # Learnable template reference that evolves during training
            self.template = nn.Parameter(torch.randn(1, feature_dim) * 0.01)

        elif variant == "contrastive":
            # Will use contrastive loss against templates
            pass

        else:
            # 'none': No template loss, only consistency
            pass

    def forward(self, features, epoch=None, variant=None):
        """
        Compute template loss.

        Args:
            features: Tensor of shape (batch_size, latent_dim)
            epoch: Current epoch (for scheduling, optional)
            variant: Override the loss variant for this forward pass

        Returns:
            loss: Scalar tensor, or 0 if variant is 'none'
        """
        if variant is None:
            variant = self.variant

        if variant == "none":
            return torch.tensor(0.0, device=features.device)

        elif variant == "learnable_template":
            # Compute distance from features to learnable template
            # L = 1/N * Σ_k ||f_k - t||²
            template_expanded = self.template.expand(features.size(0), -1)
            distances = F.mse_loss(features, template_expanded, reduction="none").sum(dim=1)
            loss = distances.mean()
            return loss

        elif variant == "contrastive":
            # Contrastive loss: pull all features toward their mean (template)
            template = features.mean(dim=0, keepdim=True)  # (1, latent_dim)
            positive_distance = F.mse_loss(features, template.expand_as(features))

            # We could add a negative term here (e.g., random negatives)
            # For now, just use positive distance
            loss = positive_distance
            return loss

        else:
            raise ValueError(f"Unknown template loss variant: {variant}")

    def get_template(self):
        """Get the current learnable template (if variant is 'learnable_template')."""
        if self.variant == "learnable_template":
            return self.template.data.clone()
        else:
            return None


class CombinedLoss(nn.Module):
    """
    Combined loss: L_total = L_consistency + λ * L_template

    Implemented as L_consistency only per user choice,
    but structure allows easy switching.
    """

    def __init__(self, lambda_consistency=0.1, lambda_template=0.0, template_variant="none"):
        """
        Args:
            lambda_consistency: Weight for consistency loss
            lambda_template: Weight for template loss
            template_variant: Type of template loss ('none', 'learnable_template', etc.)
        """
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.lambda_template = lambda_template
        self.consistency_loss = ConsistencyLoss()
        self.template_loss = TemplateLoss(variant=template_variant)

    def forward(self, features, epoch=None):
        """
        Compute combined loss.

        Args:
            features: Tensor of shape (batch_size, latent_dim)
            epoch: Current epoch (for scheduling)

        Returns:
            loss_dict: Dictionary with 'total', 'consistency', 'template' losses
        """
        l_consistency = self.consistency_loss(features)
        l_template = self.template_loss(features, epoch=epoch)

        l_total = (
            self.lambda_consistency * l_consistency +
            self.lambda_template * l_template
        )

        return {
            "total": l_total,
            "consistency": l_consistency,
            "template": l_template,
        }


# Utility loss functions for supervised baselines

class CrossEntropyWithConsistency(nn.Module):
    """
    Classification loss + consistency regularization.

    For comparing with baselines.
    """

    def __init__(self, num_classes=12, lambda_consistency=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.consistency_loss = ConsistencyLoss()
        self.lambda_consistency = lambda_consistency

    def forward(self, logits, labels, features):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes)
            labels: Tensor of shape (batch_size,)
            features: Tensor of shape (batch_size, latent_dim)

        Returns:
            loss_dict: Dictionary with total loss and component losses
        """
        ce = self.ce_loss(logits, labels)
        consistency = self.consistency_loss(features)
        total = ce + self.lambda_consistency * consistency

        return {
            "total": total,
            "ce": ce,
            "consistency": consistency,
        }


# Test code
if __name__ == "__main__":
    # Test consistency loss
    batch_size = 16
    latent_dim = 80

    features = torch.randn(batch_size, latent_dim)

    # Test 1: Consistency loss
    print("=== Testing Consistency Loss ===")
    consistency_loss_fn = ConsistencyLoss()
    loss = consistency_loss_fn(features)
    print(f"Consistency loss: {loss.item():.6f}")

    # Test 2: Template loss variants
    print("\n=== Testing Template Loss ===")

    # Variant 1: Learnable template
    template_loss_fn = TemplateLoss(variant="learnable_template", feature_dim=latent_dim)
    loss = template_loss_fn(features)
    print(f"Template loss (learnable): {loss.item():.6f}")

    # Variant 2: Contrastive
    template_loss_fn = TemplateLoss(variant="contrastive")
    loss = template_loss_fn(features)
    print(f"Template loss (contrastive): {loss.item():.6f}")

    # Variant 3: None
    template_loss_fn = TemplateLoss(variant="none")
    loss = template_loss_fn(features)
    print(f"Template loss (none): {loss.item():.6f}")

    # Test 3: Combined loss
    print("\n=== Testing Combined Loss ===")
    combined_loss_fn = CombinedLoss(
        lambda_consistency=0.1,
        lambda_template=0.0,
        template_variant="none"
    )
    loss_dict = combined_loss_fn(features)
    print(f"Combined loss: {loss_dict['total'].item():.6f}")
    print(f"  - Consistency: {loss_dict['consistency'].item():.6f}")
    print(f"  - Template: {loss_dict['template'].item():.6f}")

    # Test 4: Consistency + learnable template
    print("\n=== Testing Combined Loss (with Template) ===")
    combined_loss_fn = CombinedLoss(
        lambda_consistency=0.1,
        lambda_template=0.1,
        template_variant="learnable_template"
    )
    loss_dict = combined_loss_fn(features)
    print(f"Combined loss: {loss_dict['total'].item():.6f}")
    print(f"  - Consistency: {loss_dict['consistency'].item():.6f}")
    print(f"  - Template: {loss_dict['template'].item():.6f}")
