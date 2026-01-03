#!/usr/bin/env python3
"""
Thermal Noise LLM Simulation

This script simulates how analog hardware noise (thermal drift) affects a
Large Language Model's performance. It demonstrates the relationship between
weight-level noise and the model's output 'creativity' (entropy) and
'accuracy' (perplexity).

Usage:
    python thermal_noise_simulation.py [--model MODEL] [--noise_type TYPE]

Example:
    python thermal_noise_simulation.py --model distilgpt2 --noise_type gaussian
"""

import argparse
import copy
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from noise_distributions import inject_thermal_noise, get_model_weight_stats, NoiseType
from evaluation import compute_perplexity, generate_text_samples, compute_output_entropy


# Configuration
DEFAULT_MODEL = "distilgpt2"
DEFAULT_PROMPT = "The future of artificial intelligence is"
DEFAULT_NOISE_LEVELS = np.linspace(0, 0.1, 10)
N_SAMPLES_PER_LEVEL = 3
VALIDATION_TEXT_LENGTH = 500  # Characters from WikiText


def load_validation_text(max_chars: int = VALIDATION_TEXT_LENGTH) -> str:
    """
    Load a snippet of WikiText-2 for perplexity evaluation.

    Args:
        max_chars: Maximum number of characters to load

    Returns:
        Text snippet for validation
    """
    print("Loading WikiText-2 validation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate text until we have enough
    text = ""
    for item in dataset:
        if item["text"].strip():
            text += item["text"] + " "
            if len(text) >= max_chars:
                break

    return text[:max_chars].strip()


def run_simulation(
    model_name: str = DEFAULT_MODEL,
    noise_type: NoiseType = "gaussian",
    noise_levels: np.ndarray = DEFAULT_NOISE_LEVELS,
    prompt: str = DEFAULT_PROMPT,
    n_samples: int = N_SAMPLES_PER_LEVEL,
    device: str = "auto"
) -> Tuple[List[float], List[float], List[List[str]]]:
    """
    Run the thermal noise simulation across multiple noise levels.

    Args:
        model_name: HuggingFace model name to use
        noise_type: Type of noise distribution
        noise_levels: Array of noise levels to test
        prompt: Text prompt for generation
        n_samples: Number of text samples per noise level
        device: Device to run on ("auto", "cuda", "mps", or "cpu")

    Returns:
        Tuple of (perplexities, entropies, text_samples)
    """
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

    # Get baseline weight statistics
    base_stats = get_model_weight_stats(base_model)
    print(f"Model weight stats - Mean: {base_stats['mean']:.4f}, Std: {base_stats['std']:.4f}")
    print(f"Total parameters in Linear layers: {base_stats['total_params']:,}")

    # Load validation text
    validation_text = load_validation_text()
    print(f"Validation text length: {len(validation_text)} characters")

    # Store results
    perplexities = []
    entropies = []
    all_samples = []

    print(f"\n{'='*60}")
    print(f"Starting simulation with {noise_type} noise")
    print(f"Testing {len(noise_levels)} noise levels: {noise_levels[0]:.4f} to {noise_levels[-1]:.4f}")
    print(f"{'='*60}\n")

    for i, noise_level in enumerate(noise_levels):
        print(f"\n--- Noise Level {i+1}/{len(noise_levels)}: σ = {noise_level:.4f} ---")

        # Create a fresh copy of the model for each noise level
        noisy_model = copy.deepcopy(base_model)
        noisy_model.to(device)

        # Inject noise
        inject_thermal_noise(noisy_model, noise_level, noise_type, in_place=True)

        if noise_level > 0:
            noisy_stats = get_model_weight_stats(noisy_model)
            print(f"After noise - Mean: {noisy_stats['mean']:.4f}, Std: {noisy_stats['std']:.4f}")

        # Compute perplexity
        ppl = compute_perplexity(noisy_model, tokenizer, validation_text, device)
        perplexities.append(ppl)
        print(f"Perplexity: {ppl:.2f}")

        # Compute entropy
        entropy = compute_output_entropy(noisy_model, tokenizer, prompt, device)
        entropies.append(entropy)
        print(f"Avg. Entropy: {entropy:.2f} bits")

        # Generate text samples
        samples = generate_text_samples(
            noisy_model, tokenizer, prompt, device, n_samples=n_samples
        )
        all_samples.append(samples)

        print(f"\nGenerated Samples:")
        for j, sample in enumerate(samples, 1):
            # Truncate for display
            display_sample = sample[:200] + "..." if len(sample) > 200 else sample
            print(f"  [{j}] {display_sample}")

        # Clean up
        del noisy_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return perplexities, entropies, all_samples


def plot_results(
    noise_levels: np.ndarray,
    perplexities: List[float],
    entropies: List[float],
    noise_type: str,
    save_path: str = "noise_vs_perplexity.png"
) -> None:
    """
    Create visualization of noise level vs. perplexity and entropy.

    Args:
        noise_levels: Array of noise levels tested
        perplexities: Perplexity at each noise level
        entropies: Entropy at each noise level
        noise_type: Type of noise used (for title)
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Noise Level vs Perplexity
    ax1.plot(noise_levels, perplexities, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title(f'Noise Level vs. Perplexity\n({noise_type.capitalize()} Noise)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for perplexity

    # Add annotation for key points
    min_idx = np.argmin(perplexities)
    ax1.annotate(f'Min: {perplexities[min_idx]:.1f}',
                 xy=(noise_levels[min_idx], perplexities[min_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=10, ha='left')

    # Plot 2: Noise Level vs Entropy
    finite_entropies = [e if e != float('inf') else max([x for x in entropies if x != float('inf')], default=0) * 1.5
                        for e in entropies]
    ax2.plot(noise_levels, finite_entropies, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level (σ)', fontsize=12)
    ax2.set_ylabel('Average Entropy (bits)', fontsize=12)
    ax2.set_title(f'Noise Level vs. Output Entropy\n({noise_type.capitalize()} Noise)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def print_summary(
    noise_levels: np.ndarray,
    perplexities: List[float],
    all_samples: List[List[str]]
) -> None:
    """Print a summary table of results."""
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    print(f"{'Noise Level':^15} | {'Perplexity':^15} | {'Sample Quality':^30}")
    print("-"*70)

    for i, (noise, ppl, samples) in enumerate(zip(noise_levels, perplexities, all_samples)):
        # Assess quality based on perplexity
        if ppl < 50:
            quality = "✓ Coherent"
        elif ppl < 200:
            quality = "~ Creative Jitter"
        elif ppl < 1000:
            quality = "✗ Degraded"
        else:
            quality = "✗✗ Gibberish"

        print(f"{noise:^15.4f} | {ppl:^15.2f} | {quality:^30}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate thermal noise effects on LLM performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "uniform", "cauchy"],
        help="Type of noise distribution (default: gaussian)"
    )
    parser.add_argument(
        "--max_noise",
        type=float,
        default=0.1,
        help="Maximum noise level to test (default: 0.1)"
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=10,
        help="Number of noise levels to test (default: 10)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt for text generation (default: '{DEFAULT_PROMPT}')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on (default: auto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="noise_vs_perplexity.png",
        help="Output path for the plot (default: noise_vs_perplexity.png)"
    )

    args = parser.parse_args()

    # Define noise levels
    noise_levels = np.linspace(0, args.max_noise, args.n_levels)

    # Run simulation
    perplexities, entropies, all_samples = run_simulation(
        model_name=args.model,
        noise_type=args.noise_type,
        noise_levels=noise_levels,
        prompt=args.prompt,
        device=args.device
    )

    # Plot results
    plot_results(noise_levels, perplexities, entropies, args.noise_type, args.output)

    # Print summary
    print_summary(noise_levels, perplexities, all_samples)

    print("\n✓ Simulation complete!")


if __name__ == "__main__":
    main()
