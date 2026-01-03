#!/usr/bin/env python3
"""
Layer Sensitivity Analysis for Analog Hardware Design

This module analyzes which layers (Attention vs MLP) are more sensitive to noise,
informing hybrid analog/digital chip design decisions.

Key insight: If Attention layers are noise-tolerant but MLP layers aren't,
we can build hybrid chips with Analog Attention + Digital MLP.
"""

import copy
from typing import Literal, List, Dict
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from noise_distributions import get_noise_distribution, NoiseType
from evaluation import compute_perplexity


class LayerType(str, Enum):
    """Types of layers that can be selectively noised."""
    ATTENTION = "attention"
    MLP = "mlp"
    ALL = "all"


def get_layer_patterns(layer_type: LayerType) -> List[str]:
    """
    Get the naming patterns for different layer types in GPT-2.

    GPT-2 architecture naming:
    - Attention: h.{i}.attn.c_attn, h.{i}.attn.c_proj
    - MLP: h.{i}.mlp.c_fc, h.{i}.mlp.c_proj

    Args:
        layer_type: Type of layer to target

    Returns:
        List of substrings that identify the layer type
    """
    patterns = {
        LayerType.ATTENTION: [".attn."],
        LayerType.MLP: [".mlp."],
        LayerType.ALL: [".attn.", ".mlp."]
    }
    return patterns[layer_type]


def inject_noise_by_layer_type(
    model: nn.Module,
    noise_level: float,
    layer_type: LayerType,
    noise_type: NoiseType = "gaussian",
    in_place: bool = True
) -> nn.Module:
    """
    Inject noise only into specific layer types (Attention or MLP).

    Args:
        model: PyTorch model to inject noise into
        noise_level: Scale of noise (σ)
        layer_type: Which layers to inject noise into
        noise_type: Type of noise distribution
        in_place: If True, modify model in place

    Returns:
        Model with selectively noised weights
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
    patterns = get_layer_patterns(layer_type)

    noised_count = 0
    total_count = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            # Check for nn.Linear or HuggingFace Conv1D
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = has_conv1d and isinstance(module, Conv1D)

            if is_linear or is_conv1d:
                total_count += 1

                # Check if this layer matches our target patterns
                should_noise = any(pattern in name for pattern in patterns)

                if should_noise:
                    noised_count += 1

                    # Add noise to weights
                    noise = noise_dist.sample(
                        module.weight.shape,
                        noise_level,
                        module.weight.device
                    )
                    module.weight.add_(noise)

                    # Add noise to bias if exists
                    if module.bias is not None:
                        bias_noise = noise_dist.sample(
                            module.bias.shape,
                            noise_level,
                            module.bias.device
                        )
                        module.bias.add_(bias_noise)

    return model


def count_parameters_by_layer_type(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in Attention vs MLP layers.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary with parameter counts per layer type
    """
    counts = {
        "attention": 0,
        "mlp": 0,
        "other": 0
    }

    for name, param in model.named_parameters():
        if ".attn." in name:
            counts["attention"] += param.numel()
        elif ".mlp." in name:
            counts["mlp"] += param.numel()
        else:
            counts["other"] += param.numel()

    return counts


def run_layer_sensitivity_experiment(
    model_name: str = "distilgpt2",
    noise_levels: np.ndarray = None,
    noise_type: NoiseType = "gaussian",
    device: str = "auto"
) -> Dict[str, List[float]]:
    """
    Run the layer sensitivity experiment.

    Compares perplexity degradation when noise is injected into:
    - Attention layers only
    - MLP layers only
    - All layers (baseline comparison)

    Args:
        model_name: HuggingFace model name
        noise_levels: Array of noise levels to test
        noise_type: Type of noise distribution
        device: Compute device

    Returns:
        Dictionary mapping layer type to list of perplexities
    """
    if noise_levels is None:
        noise_levels = np.linspace(0, 0.05, 10)

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.to(device)
    base_model.eval()

    # Analyze parameter distribution
    param_counts = count_parameters_by_layer_type(base_model)
    total = sum(param_counts.values())
    print(f"\nParameter Distribution:")
    print(f"  Attention: {param_counts['attention']:,} ({100*param_counts['attention']/total:.1f}%)")
    print(f"  MLP:       {param_counts['mlp']:,} ({100*param_counts['mlp']/total:.1f}%)")
    print(f"  Other:     {param_counts['other']:,} ({100*param_counts['other']/total:.1f}%)")

    # Load validation text
    print("\nLoading validation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = ""
    for item in dataset:
        if item["text"].strip():
            text += item["text"] + " "
            if len(text) >= 500:
                break
    validation_text = text[:500].strip()

    # Run experiments for each layer type
    results = {
        "attention_only": [],
        "mlp_only": [],
        "all_layers": []
    }

    layer_configs = [
        ("attention_only", LayerType.ATTENTION),
        ("mlp_only", LayerType.MLP),
        ("all_layers", LayerType.ALL)
    ]

    for config_name, layer_type in layer_configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config_name}")
        print(f"{'='*50}")

        for i, noise_level in enumerate(noise_levels):
            print(f"  σ = {noise_level:.4f}", end=" → ")

            # Create fresh copy
            noisy_model = copy.deepcopy(base_model)
            noisy_model.to(device)

            # Inject noise selectively
            inject_noise_by_layer_type(
                noisy_model, noise_level, layer_type, noise_type, in_place=True
            )

            # Compute perplexity
            ppl = compute_perplexity(noisy_model, tokenizer, validation_text, device)
            results[config_name].append(ppl)
            print(f"Perplexity: {ppl:.2f}")

            del noisy_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results, noise_levels


def plot_layer_sensitivity(
    results: Dict[str, List[float]],
    noise_levels: np.ndarray,
    save_path: str = "images/layer_sensitivity.png"
) -> None:
    """
    Plot the layer sensitivity comparison.

    Args:
        results: Dictionary of perplexity results per layer type
        noise_levels: Noise levels tested
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "attention_only": "#E91E63",  # Pink
        "mlp_only": "#2196F3",        # Blue
        "all_layers": "#9C27B0"       # Purple
    }

    markers = {
        "attention_only": "o",
        "mlp_only": "s",
        "all_layers": "^"
    }

    labels = {
        "attention_only": "Attention Layers Only",
        "mlp_only": "MLP Layers Only",
        "all_layers": "All Layers"
    }

    for config_name, perplexities in results.items():
        ax.plot(
            noise_levels, perplexities,
            marker=markers[config_name],
            color=colors[config_name],
            linewidth=2,
            markersize=8,
            label=labels[config_name]
        )

    ax.set_xlabel('Noise Level (σ)', fontsize=14)
    ax.set_ylabel('Perplexity (log scale)', fontsize=14)
    ax.set_title('Layer Sensitivity Analysis\nAttention vs MLP Noise Tolerance',
                 fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xlim(0, noise_levels[-1])

    # Add insight annotation
    attn_final = results["attention_only"][-1]
    mlp_final = results["mlp_only"][-1]

    if attn_final < mlp_final:
        insight = "→ Attention layers are MORE noise-tolerant"
    else:
        insight = "→ MLP layers are MORE noise-tolerant"

    ax.text(0.5, 0.02, insight, transform=ax.transAxes,
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer Sensitivity Analysis for Analog Hardware Design"
    )
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--max_noise", type=float, default=0.05)
    parser.add_argument("--n_levels", type=int, default=10)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="images/layer_sensitivity.png")

    args = parser.parse_args()

    noise_levels = np.linspace(0, args.max_noise, args.n_levels)

    print("🧠 Layer Sensitivity Analysis")
    print("=" * 50)
    print("Comparing noise tolerance: Attention vs MLP layers")
    print("=" * 50)

    results, noise_levels = run_layer_sensitivity_experiment(
        model_name=args.model,
        noise_levels=noise_levels,
        noise_type=args.noise_type,
        device=args.device
    )

    plot_layer_sensitivity(results, noise_levels, args.output)

    # Print summary
    print("\n" + "=" * 50)
    print("LAYER SENSITIVITY SUMMARY")
    print("=" * 50)

    # Find the noise level where each configuration crosses perplexity threshold
    threshold = 100
    for config_name, perplexities in results.items():
        cross_idx = next((i for i, p in enumerate(perplexities) if p > threshold), -1)
        if cross_idx > 0:
            print(f"{config_name}: Crosses PPL={threshold} at σ ≈ {noise_levels[cross_idx]:.4f}")
        else:
            print(f"{config_name}: Never crosses PPL={threshold} in tested range")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
