"""
Modular noise distribution system for simulating analog hardware effects.

This module implements a Strategy Pattern for noise injection, allowing easy
extension with new noise types by subclassing NoiseDistribution.

Supported noise types:
- Gaussian (thermal noise)
- Uniform (quantization noise)
- Cauchy (heavy-tailed noise for modeling outliers)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Literal
import copy

import torch
import torch.nn as nn


class NoiseDistribution(ABC):
    """Abstract base class for noise distributions."""

    @abstractmethod
    def sample(self, shape: Tuple[int, ...], scale: float, device: torch.device) -> torch.Tensor:
        """
        Generate noise samples.

        Args:
            shape: Shape of the noise tensor to generate
            scale: Scale parameter (interpretation depends on distribution)
            device: Device to create tensor on

        Returns:
            Tensor of noise samples with the specified shape
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the distribution."""
        pass


class GaussianNoise(NoiseDistribution):
    """
    Gaussian (Normal) noise distribution.

    Models thermal noise in analog circuits.
    N(0, σ²) where σ = scale parameter.
    """

    def sample(self, shape: Tuple[int, ...], scale: float, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device) * scale

    @property
    def name(self) -> str:
        return "Gaussian"


class UniformNoise(NoiseDistribution):
    """
    Uniform noise distribution.

    Models quantization noise in digital-to-analog converters.
    U(-a, a) where a = scale parameter.
    """

    def sample(self, shape: Tuple[int, ...], scale: float, device: torch.device) -> torch.Tensor:
        return (torch.rand(shape, device=device) * 2 - 1) * scale

    @property
    def name(self) -> str:
        return "Uniform"


class CauchyNoise(NoiseDistribution):
    """
    Cauchy noise distribution (heavy-tailed).

    Models outlier events and extreme noise spikes in analog hardware.
    Cauchy(0, γ) where γ = scale parameter.

    Warning: Cauchy distribution has undefined mean and variance,
    so even small scale values can produce extreme outliers.
    """

    def sample(self, shape: Tuple[int, ...], scale: float, device: torch.device) -> torch.Tensor:
        # Cauchy samples via inverse CDF: x = tan(π(U - 0.5))
        u = torch.rand(shape, device=device)
        return torch.tan(torch.pi * (u - 0.5)) * scale

    @property
    def name(self) -> str:
        return "Cauchy"


# Registry of available noise types
NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "uniform": UniformNoise,
    "cauchy": CauchyNoise,
}

NoiseType = Literal["gaussian", "uniform", "cauchy"]


def get_noise_distribution(noise_type: NoiseType) -> NoiseDistribution:
    """
    Factory function to get a noise distribution instance.

    Args:
        noise_type: Type of noise distribution ("gaussian", "uniform", "cauchy")

    Returns:
        NoiseDistribution instance

    Raises:
        ValueError: If noise_type is not recognized
    """
    if noise_type not in NOISE_REGISTRY:
        available = ", ".join(NOISE_REGISTRY.keys())
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {available}")
    return NOISE_REGISTRY[noise_type]()


def inject_thermal_noise(
    model: nn.Module,
    noise_level: float,
    noise_type: NoiseType = "gaussian",
    in_place: bool = True
) -> nn.Module:
    """
    Inject noise into all Linear and Conv1D layer weights of a model.

    Simulates thermal drift in analog hardware by adding noise:
        W_noisy = W + noise(0, σ²)

    where σ (sigma) represents the 'Thermal Temperature'.

    Args:
        model: PyTorch model to inject noise into
        noise_level: Scale of noise (σ for Gaussian, bounds for Uniform, γ for Cauchy)
        noise_type: Type of noise distribution to use
        in_place: If True, modify model in place. If False, work on a deep copy.

    Returns:
        Model with noisy weights (same instance if in_place=True, copy otherwise)
    """
    # Import Conv1D for GPT-2 style models
    try:
        from transformers.pytorch_utils import Conv1D
        has_conv1d = True
    except ImportError:
        has_conv1d = False
        Conv1D = None

    if not in_place:
        model = copy.deepcopy(model)

    if noise_level == 0:
        return model

    noise_dist = get_noise_distribution(noise_type)

    with torch.no_grad():
        for name, module in model.named_modules():
            # Check for nn.Linear
            is_linear = isinstance(module, nn.Linear)
            # Check for HuggingFace Conv1D (used by GPT-2)
            is_conv1d = has_conv1d and isinstance(module, Conv1D)

            if is_linear or is_conv1d:
                # Add noise to weights
                noise = noise_dist.sample(
                    module.weight.shape,
                    noise_level,
                    module.weight.device
                )
                module.weight.add_(noise)

                # Optionally add noise to bias if it exists
                if module.bias is not None:
                    bias_noise = noise_dist.sample(
                        module.bias.shape,
                        noise_level,
                        module.bias.device
                    )
                    module.bias.add_(bias_noise)

    return model


def get_model_weight_stats(model: nn.Module) -> dict:
    """
    Get statistics about model weights for analysis.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary with weight statistics (mean, std, min, max)
    """
    all_weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.flatten())
            if module.bias is not None:
                all_weights.append(module.bias.data.flatten())

    if not all_weights:
        return {}

    weights = torch.cat(all_weights)
    return {
        "mean": weights.mean().item(),
        "std": weights.std().item(),
        "min": weights.min().item(),
        "max": weights.max().item(),
        "total_params": weights.numel()
    }
