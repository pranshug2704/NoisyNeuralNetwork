#!/usr/bin/env python3
"""
Generate comparison graphs for all noise types.

This script runs simulations for Gaussian, Uniform, and Cauchy noise
distributions and generates comparison plots for the README.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from noise_distributions import inject_thermal_noise, get_model_weight_stats
from evaluation import compute_perplexity


# Configuration
MODEL_NAME = "distilgpt2"
NOISE_LEVELS = np.linspace(0, 0.1, 10)
NOISE_TYPES = ["gaussian", "uniform", "cauchy"]
VALIDATION_TEXT_LENGTH = 500


def load_validation_text(max_chars: int = VALIDATION_TEXT_LENGTH) -> str:
    """Load WikiText-2 validation text."""
    print("Loading WikiText-2 validation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = ""
    for item in dataset:
        if item["text"].strip():
            text += item["text"] + " "
            if len(text) >= max_chars:
                break
    return text[:max_chars].strip()


def run_noise_comparison():
    """Run simulations for all noise types and generate plots."""

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.to(device)
    base_model.eval()

    # Load validation text
    validation_text = load_validation_text()

    # Store results for all noise types
    results = {noise_type: [] for noise_type in NOISE_TYPES}

    for noise_type in NOISE_TYPES:
        print(f"\n{'='*50}")
        print(f"Testing {noise_type.upper()} noise")
        print(f"{'='*50}")

        for i, noise_level in enumerate(NOISE_LEVELS):
            print(f"  Level {i+1}/{len(NOISE_LEVELS)}: σ = {noise_level:.4f}", end=" → ")

            # Create fresh copy
            noisy_model = copy.deepcopy(base_model)
            noisy_model.to(device)

            # Inject noise
            inject_thermal_noise(noisy_model, noise_level, noise_type, in_place=True)

            # Compute perplexity
            ppl = compute_perplexity(noisy_model, tokenizer, validation_text, device)
            results[noise_type].append(ppl)
            print(f"Perplexity: {ppl:.2f}")

            del noisy_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def generate_individual_plots(results):
    """Generate individual plots for each noise type."""

    colors = {"gaussian": "#2196F3", "uniform": "#4CAF50", "cauchy": "#FF5722"}

    for noise_type, perplexities in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(NOISE_LEVELS, perplexities, 'o-',
                color=colors[noise_type], linewidth=2, markersize=8)

        ax.set_xlabel('Noise Level (σ)', fontsize=14)
        ax.set_ylabel('Perplexity', fontsize=14)
        ax.set_title(f'Impact of {noise_type.capitalize()} Noise on GPT-2 Perplexity',
                     fontsize=16, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, NOISE_LEVELS[-1])

        # Add threshold annotation
        threshold_idx = next((i for i, p in enumerate(perplexities) if p > 1000), -1)
        if threshold_idx > 0:
            ax.axvline(x=NOISE_LEVELS[threshold_idx], color='red',
                      linestyle='--', alpha=0.5, label='Breakdown threshold')
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'images/noise_vs_perplexity_{noise_type}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: images/noise_vs_perplexity_{noise_type}.png")
        plt.close()


def generate_comparison_plot(results):
    """Generate a combined comparison plot."""

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"gaussian": "#2196F3", "uniform": "#4CAF50", "cauchy": "#FF5722"}
    markers = {"gaussian": "o", "uniform": "s", "cauchy": "^"}

    for noise_type, perplexities in results.items():
        ax.plot(NOISE_LEVELS, perplexities,
                marker=markers[noise_type],
                color=colors[noise_type],
                linewidth=2,
                markersize=8,
                label=f'{noise_type.capitalize()} Noise')

    ax.set_xlabel('Noise Level (σ)', fontsize=14)
    ax.set_ylabel('Perplexity (log scale)', fontsize=14)
    ax.set_title('Comparison of Noise Distributions on LLM Performance\n(distilgpt2)',
                 fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xlim(0, NOISE_LEVELS[-1])

    # Add annotations
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.005, 120, 'Acceptable threshold', fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.savefig('images/noise_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: images/noise_comparison.png")
    plt.close()


def main():
    print("🧠 Noisy Neural Network - Graph Generation")
    print("=" * 50)

    # Run simulations
    results = run_noise_comparison()

    # Generate plots
    print("\nGenerating individual plots...")
    generate_individual_plots(results)

    print("\nGenerating comparison plot...")
    generate_comparison_plot(results)

    print("\n✅ All graphs generated successfully!")
    print("Files saved in: images/")


if __name__ == "__main__":
    main()
