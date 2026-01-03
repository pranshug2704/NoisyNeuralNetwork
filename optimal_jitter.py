#!/usr/bin/env python3
"""
Optimal Jitter Search: Finding the "Goldilocks Zone"

This module searches for the optimal noise level (σ) where the model exhibits:
- Higher diversity/creativity than baseline
- Acceptable perplexity (before the "cliff")

This represents the sweet spot for analog hardware where thermal noise
actually improves output diversity without causing model breakdown.
"""

import copy
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from noise_distributions import inject_thermal_noise, NoiseType
from evaluation import (
    compute_perplexity,
    generate_text_samples,
    compute_distinct_n,
    compute_repetition_ratio,
    compute_vocabulary_richness
)


def evaluate_at_noise_level(
    model,
    tokenizer,
    noise_level: float,
    prompt: str,
    validation_text: str,
    device: torch.device,
    n_samples: int = 10,
    noise_type: NoiseType = "gaussian"
) -> Dict:
    """
    Evaluate model metrics at a specific noise level.

    Args:
        model: Base model (will be copied, not modified)
        tokenizer: Tokenizer
        noise_level: Noise level σ to test
        prompt: Generation prompt
        validation_text: Text for perplexity evaluation
        device: Compute device
        n_samples: Number of text samples to generate
        noise_type: Type of noise distribution

    Returns:
        Dictionary with perplexity, diversity metrics, and samples
    """
    # Create noisy copy
    noisy_model = copy.deepcopy(model)
    noisy_model.to(device)
    inject_thermal_noise(noisy_model, noise_level, noise_type, in_place=True)

    # Compute perplexity
    ppl = compute_perplexity(noisy_model, tokenizer, validation_text, device)

    # Generate samples
    samples = generate_text_samples(
        noisy_model, tokenizer, prompt, device,
        n_samples=n_samples, max_new_tokens=50
    )

    # Compute diversity metrics
    richness = compute_vocabulary_richness(samples)
    repetition = compute_repetition_ratio(samples, n=3)

    # Clean up
    del noisy_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "noise_level": noise_level,
        "perplexity": ppl,
        "distinct_1": richness["distinct_1"],
        "distinct_2": richness["distinct_2"],
        "distinct_3": richness["distinct_3"],
        "type_token_ratio": richness["type_token_ratio"],
        "repetition_ratio": repetition,
        "samples": samples
    }


def find_goldilocks_zone(
    model_name: str = "distilgpt2",
    noise_range: Tuple[float, float] = (0.001, 0.03),
    n_points: int = 20,
    prompt: str = "The future of artificial intelligence is",
    perplexity_threshold: float = 200,
    noise_type: NoiseType = "gaussian",
    device: str = "auto"
) -> Dict:
    """
    Search for the optimal noise level (Goldilocks zone).

    The Goldilocks zone is where:
    - diversity > baseline_diversity (more creative)
    - perplexity < threshold (still logical)

    Args:
        model_name: HuggingFace model name
        noise_range: (min_noise, max_noise) to search
        n_points: Number of noise levels to test
        prompt: Generation prompt
        perplexity_threshold: Maximum acceptable perplexity
        noise_type: Type of noise distribution
        device: Compute device

    Returns:
        Dictionary with all results and identified optimal zone
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

    # Load model
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.to(device)
    base_model.eval()

    # Load validation text
    print("Loading validation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = ""
    for item in dataset:
        if item["text"].strip():
            text += item["text"] + " "
            if len(text) >= 500:
                break
    validation_text = text[:500].strip()

    # Include 0 (baseline) in noise levels
    noise_levels = np.concatenate([[0], np.linspace(*noise_range, n_points)])

    print(f"\n{'='*60}")
    print("OPTIMAL JITTER SEARCH")
    print(f"Searching noise range: {noise_range[0]:.4f} to {noise_range[1]:.4f}")
    print(f"{'='*60}")

    results = []

    for i, noise_level in enumerate(noise_levels):
        print(f"\n[{i+1}/{len(noise_levels)}] σ = {noise_level:.5f}")

        metrics = evaluate_at_noise_level(
            base_model, tokenizer, noise_level, prompt,
            validation_text, device, n_samples=10, noise_type=noise_type
        )
        results.append(metrics)

        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Distinct-2: {metrics['distinct_2']:.4f}")
        print(f"  Repetition: {metrics['repetition_ratio']:.4f}")

    # Analyze results to find Goldilocks zone
    baseline = results[0]
    baseline_diversity = baseline["distinct_2"]

    goldilocks_candidates = []

    for r in results[1:]:  # Skip baseline
        is_more_diverse = r["distinct_2"] > baseline_diversity
        is_acceptable_ppl = r["perplexity"] < perplexity_threshold
        is_not_repetitive = r["repetition_ratio"] < 0.5

        if is_more_diverse and is_acceptable_ppl and is_not_repetitive:
            # Score: higher diversity, lower perplexity is better
            score = r["distinct_2"] / (r["perplexity"] / baseline["perplexity"])
            goldilocks_candidates.append({
                **r,
                "goldilocks_score": score
            })

    # Find optimal
    optimal = None
    if goldilocks_candidates:
        optimal = max(goldilocks_candidates, key=lambda x: x["goldilocks_score"])

    return {
        "results": results,
        "baseline": baseline,
        "optimal": optimal,
        "goldilocks_candidates": goldilocks_candidates,
        "noise_levels": noise_levels
    }


def plot_goldilocks_zone(
    results: Dict,
    save_path: str = "images/goldilocks_zone.png"
) -> None:
    """
    Plot the Goldilocks zone analysis with dual y-axis.

    Args:
        results: Results dictionary from find_goldilocks_zone
        save_path: Path to save the plot
    """
    data = results["results"]
    baseline = results["baseline"]
    optimal = results["optimal"]

    noise_levels = [d["noise_level"] for d in data]
    perplexities = [d["perplexity"] for d in data]
    distinct_2 = [d["distinct_2"] for d in data]
    repetition = [d["repetition_ratio"] for d in data]

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    # Plot perplexity (log scale)
    line1 = ax1.semilogy(noise_levels, perplexities, 'b-o',
                         linewidth=2, markersize=6, label='Perplexity')
    ax1.set_xlabel('Noise Level (σ)', fontsize=14)
    ax1.set_ylabel('Perplexity (log scale)', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot Distinct-2 diversity
    line2 = ax2.plot(noise_levels, distinct_2, 'g-s',
                     linewidth=2, markersize=6, label='Distinct-2 (Diversity)')
    ax2.set_ylabel('Distinct-2 Score', color='green', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='green')

    # Mark baseline diversity
    ax2.axhline(y=baseline["distinct_2"], color='green', linestyle='--',
                alpha=0.5, label='Baseline Diversity')

    # Highlight Goldilocks zone
    if results["goldilocks_candidates"]:
        goldilocks_x = [c["noise_level"] for c in results["goldilocks_candidates"]]
        ax1.axvspan(min(goldilocks_x), max(goldilocks_x),
                   alpha=0.2, color='gold', label='Goldilocks Zone')

    # Mark optimal point
    if optimal:
        ax1.axvline(x=optimal["noise_level"], color='red', linestyle='-',
                   linewidth=2, alpha=0.7)
        ax1.annotate(f'Optimal σ={optimal["noise_level"]:.4f}\n'
                    f'PPL={optimal["perplexity"]:.1f}\n'
                    f'Diverse={optimal["distinct_2"]:.3f}',
                    xy=(optimal["noise_level"], optimal["perplexity"]),
                    xytext=(20, 20), textcoords='offset points',
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_title('Optimal Jitter Search: Finding the Goldilocks Zone\n'
                  'Where Creativity Increases Without Model Breakdown',
                  fontsize=16, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def print_goldilocks_summary(results: Dict) -> None:
    """Print summary of the Goldilocks zone analysis."""
    baseline = results["baseline"]
    optimal = results["optimal"]
    candidates = results["goldilocks_candidates"]

    print("\n" + "=" * 60)
    print("GOLDILOCKS ZONE ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nBaseline (σ=0):")
    print(f"  Perplexity:  {baseline['perplexity']:.2f}")
    print(f"  Distinct-2:  {baseline['distinct_2']:.4f}")
    print(f"  Repetition:  {baseline['repetition_ratio']:.4f}")

    if candidates:
        print(f"\n✓ Found {len(candidates)} noise levels in the Goldilocks Zone!")
        print(f"\nOptimal noise level:")
        print(f"  σ = {optimal['noise_level']:.5f}")
        print(f"  Perplexity:  {optimal['perplexity']:.2f} (+{100*(optimal['perplexity']/baseline['perplexity']-1):.1f}%)")
        print(f"  Distinct-2:  {optimal['distinct_2']:.4f} (+{100*(optimal['distinct_2']/baseline['distinct_2']-1):.1f}%)")
        print(f"  Repetition:  {optimal['repetition_ratio']:.4f}")

        print(f"\nSample text at optimal σ:")
        for i, sample in enumerate(optimal['samples'][:2], 1):
            print(f"  [{i}] {sample[:150]}...")
    else:
        print("\n✗ No Goldilocks zone found in the tested range.")
        print("  Try adjusting the noise_range or perplexity_threshold.")

    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Find the optimal noise level (Goldilocks Zone)"
    )
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--search_range", type=float, nargs=2, default=[0.001, 0.03])
    parser.add_argument("--n_points", type=int, default=15)
    parser.add_argument("--perplexity_threshold", type=float, default=200)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="images/goldilocks_zone.png")

    args = parser.parse_args()

    print("🎯 Optimal Jitter Search")
    print("=" * 60)
    print("Finding the Goldilocks Zone: More Creative, Still Logical")
    print("=" * 60)

    results = find_goldilocks_zone(
        model_name=args.model,
        noise_range=tuple(args.search_range),
        n_points=args.n_points,
        perplexity_threshold=args.perplexity_threshold,
        noise_type=args.noise_type,
        device=args.device
    )

    plot_goldilocks_zone(results, args.output)
    print_goldilocks_summary(results)

    print("\n✓ Optimal jitter search complete!")


if __name__ == "__main__":
    main()
