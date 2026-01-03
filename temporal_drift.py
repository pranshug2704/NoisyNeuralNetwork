#!/usr/bin/env python3
"""
Temporal Noise (Weight Drift) Simulation

Simulates memristor-like weight drift where noise level σ increases over time
during text generation. This models how analog hardware degrades during a
"thinking session" as the material heats up or cools down.

Key question: Does the model degrade gracefully (like a tired human) or
suddenly snap at a critical point?
"""

import copy
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from noise_distributions import get_noise_distribution, NoiseType


class DriftSchedule(str, Enum):
    """Types of noise drift schedules."""
    LINEAR = "linear"       # σ increases linearly
    EXPONENTIAL = "exponential"  # σ increases exponentially
    STEP = "step"           # σ jumps at specific points
    SINE = "sine"           # σ oscillates (thermal cycling)


def get_drift_schedule(
    schedule: DriftSchedule,
    start_sigma: float,
    end_sigma: float,
    total_steps: int
) -> Callable[[int], float]:
    """
    Create a noise schedule function.

    Args:
        schedule: Type of drift schedule
        start_sigma: Initial noise level
        end_sigma: Final noise level
        total_steps: Total number of steps

    Returns:
        Function that maps step number to noise level
    """
    def linear(step: int) -> float:
        progress = step / max(total_steps - 1, 1)
        return start_sigma + (end_sigma - start_sigma) * progress

    def exponential(step: int) -> float:
        progress = step / max(total_steps - 1, 1)
        # Exponential interpolation
        if start_sigma == 0:
            return end_sigma * (np.exp(progress * 3) - 1) / (np.exp(3) - 1)
        return start_sigma * (end_sigma / start_sigma) ** progress

    def step_func(step: int) -> float:
        # Three distinct phases
        if step < total_steps // 3:
            return start_sigma
        elif step < 2 * total_steps // 3:
            return (start_sigma + end_sigma) / 2
        else:
            return end_sigma

    def sine(step: int) -> float:
        # Oscillating with upward trend
        progress = step / max(total_steps - 1, 1)
        base = start_sigma + (end_sigma - start_sigma) * progress
        oscillation = 0.3 * (end_sigma - start_sigma) * np.sin(4 * np.pi * progress)
        return max(0, base + oscillation)

    schedules = {
        DriftSchedule.LINEAR: linear,
        DriftSchedule.EXPONENTIAL: exponential,
        DriftSchedule.STEP: step_func,
        DriftSchedule.SINE: sine,
    }

    return schedules[schedule]


def inject_noise_at_level(
    model: nn.Module,
    noise_level: float,
    noise_type: NoiseType = "gaussian"
) -> None:
    """
    Inject noise into model weights in-place.

    Args:
        model: Model to inject noise into
        noise_level: Current noise level σ
        noise_type: Type of noise distribution
    """
    try:
        from transformers.pytorch_utils import Conv1D
        has_conv1d = True
    except ImportError:
        has_conv1d = False
        Conv1D = None

    if noise_level == 0:
        return

    noise_dist = get_noise_distribution(noise_type)

    with torch.no_grad():
        for name, module in model.named_modules():
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = has_conv1d and isinstance(module, Conv1D)

            if is_linear or is_conv1d:
                noise = noise_dist.sample(
                    module.weight.shape,
                    noise_level,
                    module.weight.device
                )
                module.weight.add_(noise)


@dataclass
class DriftResult:
    """Results from a single generation step."""
    step: int
    sigma: float
    token_id: int
    token_text: str
    log_prob: float
    cumulative_text: str


def generate_with_drift(
    model_name: str = "distilgpt2",
    prompt: str = "The future of artificial intelligence is",
    total_tokens: int = 100,
    start_sigma: float = 0.0,
    end_sigma: float = 0.02,
    schedule: DriftSchedule = DriftSchedule.LINEAR,
    noise_type: NoiseType = "gaussian",
    device: str = "auto"
) -> Tuple[List[DriftResult], str]:
    """
    Generate text with time-varying noise (weight drift).

    Args:
        model_name: HuggingFace model name
        prompt: Starting prompt
        total_tokens: Number of tokens to generate
        start_sigma: Initial noise level
        end_sigma: Final noise level
        schedule: Type of drift schedule
        noise_type: Type of noise distribution
        device: Compute device

    Returns:
        List of DriftResults and final generated text
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.to(device)
    base_model.eval()

    # Create noise schedule
    noise_schedule = get_drift_schedule(schedule, start_sigma, end_sigma, total_tokens)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    results = []
    generated_text = prompt

    print(f"\n{'='*60}")
    print(f"TEMPORAL DRIFT SIMULATION")
    print(f"Generating {total_tokens} tokens with σ: {start_sigma} → {end_sigma}")
    print(f"Schedule: {schedule.value}")
    print(f"{'='*60}\n")

    # Store original weights to compute incremental noise
    prev_sigma = 0.0

    for step in range(total_tokens):
        current_sigma = noise_schedule(step)

        # Calculate incremental noise (difference from previous step)
        delta_sigma = current_sigma - prev_sigma

        if delta_sigma > 0:
            # Only add the delta noise (incremental drift)
            # For realistic drift, we add small increments each step
            inject_noise_at_level(base_model, delta_sigma, noise_type)

        prev_sigma = current_sigma

        # Generate next token
        with torch.no_grad():
            outputs = base_model(input_ids)
            logits = outputs.logits[:, -1, :]

            # Handle NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n⚠️ Model collapsed at step {step} (σ = {current_sigma:.4f})")
                break

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Get log probability
            log_prob = torch.log(probs[0, next_token[0, 0]]).item()

            # Decode token
            token_text = tokenizer.decode(next_token[0])

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        result = DriftResult(
            step=step,
            sigma=current_sigma,
            token_id=next_token[0, 0].item(),
            token_text=token_text,
            log_prob=log_prob,
            cumulative_text=generated_text
        )
        results.append(result)

        # Progress indicator
        if step % 20 == 0 or step == total_tokens - 1:
            print(f"Step {step:3d}/{total_tokens}: σ = {current_sigma:.4f}, "
                  f"log_prob = {log_prob:.2f}")

    return results, generated_text


def analyze_drift_results(results: List[DriftResult]) -> Dict:
    """
    Analyze the drift results to determine degradation pattern.

    Args:
        results: List of DriftResults from generation

    Returns:
        Analysis dictionary with degradation metrics
    """
    if not results:
        return {}

    steps = [r.step for r in results]
    sigmas = [r.sigma for r in results]
    log_probs = [r.log_prob for r in results]

    # Smooth log_probs with rolling average
    window = min(10, len(log_probs))
    smoothed = np.convolve(log_probs, np.ones(window)/window, mode='valid')

    # Find degradation point (where log_prob drops significantly)
    baseline_prob = np.mean(log_probs[:10]) if len(log_probs) >= 10 else log_probs[0]

    # Look for cliff (sudden drop)
    cliff_step = None
    for i in range(len(log_probs)):
        if log_probs[i] < baseline_prob - 2.0:  # 2 nats drop
            cliff_step = i
            break

    # Compute degradation rate
    if len(log_probs) > 1:
        degradation_rate = (log_probs[-1] - log_probs[0]) / len(log_probs)
    else:
        degradation_rate = 0

    # Determine degradation type
    if cliff_step and cliff_step < len(log_probs) * 0.7:
        degradation_type = "CLIFF (Sudden Collapse)"
    elif abs(degradation_rate) < 0.01:
        degradation_type = "STABLE (No significant degradation)"
    else:
        degradation_type = "GRADUAL (Like a tired human)"

    return {
        "total_steps": len(results),
        "final_sigma": sigmas[-1] if sigmas else 0,
        "baseline_log_prob": baseline_prob,
        "final_log_prob": log_probs[-1] if log_probs else 0,
        "degradation_rate": degradation_rate,
        "cliff_step": cliff_step,
        "degradation_type": degradation_type
    }


def plot_drift_analysis(
    results: List[DriftResult],
    analysis: Dict,
    save_path: str = "images/temporal_drift.png"
) -> None:
    """
    Plot the temporal drift analysis.

    Args:
        results: List of DriftResults
        analysis: Analysis dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = [r.step for r in results]
    sigmas = [r.sigma for r in results]
    log_probs = [r.log_prob for r in results]

    # Plot 1: Noise level over time
    ax1 = axes[0, 0]
    ax1.plot(steps, sigmas, 'b-', linewidth=2)
    ax1.set_xlabel('Token Position', fontsize=12)
    ax1.set_ylabel('Noise Level (σ)', fontsize=12)
    ax1.set_title('Noise Level Schedule', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log probability over time
    ax2 = axes[0, 1]
    ax2.plot(steps, log_probs, 'g-', alpha=0.5, linewidth=1, label='Raw')

    # Add smoothed line
    window = min(10, len(log_probs))
    if window > 1:
        smoothed = np.convolve(log_probs, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(log_probs)), smoothed, 'r-',
                linewidth=2, label='Smoothed')

    ax2.set_xlabel('Token Position', fontsize=12)
    ax2.set_ylabel('Log Probability', fontsize=12)
    ax2.set_title('Token Confidence Over Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark cliff if found
    if analysis.get('cliff_step'):
        ax2.axvline(x=analysis['cliff_step'], color='red', linestyle='--',
                   label=f'Cliff at step {analysis["cliff_step"]}')

    # Plot 3: σ vs Log Probability (phase diagram)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(sigmas, log_probs, c=steps, cmap='viridis',
                          alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax3, label='Token Position')
    ax3.set_xlabel('Noise Level (σ)', fontsize=12)
    ax3.set_ylabel('Log Probability', fontsize=12)
    ax3.set_title('Phase Diagram: Noise vs Confidence', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
TEMPORAL DRIFT ANALYSIS

Degradation Type: {analysis.get('degradation_type', 'N/A')}

Metrics:
  • Total Tokens: {analysis.get('total_steps', 0)}
  • Final σ: {analysis.get('final_sigma', 0):.4f}
  • Baseline Log Prob: {analysis.get('baseline_log_prob', 0):.2f}
  • Final Log Prob: {analysis.get('final_log_prob', 0):.2f}
  • Degradation Rate: {analysis.get('degradation_rate', 0):.4f}/token

Cliff Point: {f"Step {analysis['cliff_step']}" if analysis.get('cliff_step') else "None detected"}
"""

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Temporal Weight Drift: Does the Model Degrade Gracefully?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Temporal Noise (Weight Drift) Simulation"
    )
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--total_tokens", type=int, default=100)
    parser.add_argument("--start_sigma", type=float, default=0.0)
    parser.add_argument("--end_sigma", type=float, default=0.02)
    parser.add_argument("--schedule", type=str, default="linear",
                        choices=["linear", "exponential", "step", "sine"])
    parser.add_argument("--prompt", type=str,
                        default="The future of artificial intelligence is")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="images/temporal_drift.png")

    args = parser.parse_args()

    print("⏰ Temporal Weight Drift Simulation")
    print("=" * 60)
    print("Simulating memristor-like weight drift during generation")
    print("=" * 60)

    results, final_text = generate_with_drift(
        model_name=args.model,
        prompt=args.prompt,
        total_tokens=args.total_tokens,
        start_sigma=args.start_sigma,
        end_sigma=args.end_sigma,
        schedule=DriftSchedule(args.schedule),
        device=args.device
    )

    analysis = analyze_drift_results(results)

    plot_drift_analysis(results, analysis, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("TEMPORAL DRIFT SUMMARY")
    print("=" * 60)
    print(f"Degradation Type: {analysis.get('degradation_type', 'N/A')}")
    print(f"Final σ: {analysis.get('final_sigma', 0):.4f}")
    print(f"Degradation Rate: {analysis.get('degradation_rate', 0):.4f}/token")

    if analysis.get('cliff_step'):
        sigma_at_cliff = results[analysis['cliff_step']].sigma
        print(f"\n⚠️ CLIFF DETECTED at step {analysis['cliff_step']} (σ = {sigma_at_cliff:.4f})")
    else:
        print("\n✓ Graceful degradation (no sudden collapse)")

    print(f"\nGenerated text preview:")
    print(f"  {final_text[:200]}...")

    print("\n✓ Temporal drift simulation complete!")


if __name__ == "__main__":
    main()
