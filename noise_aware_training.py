#!/usr/bin/env python3
"""
Noise-Aware Fine-tuning (Quantization-Aware Training for Analog Hardware)

This module implements training with noise injection during the forward pass,
teaching the model to store information in a more redundant, "holographic" way
that is immune to analog drift.

Key insight: If we tell the model "expect your weights to be shaky" during training,
it learns representations that are robust to noise at inference time.
"""

import copy
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

from noise_distributions import get_noise_distribution, NoiseType
from evaluation import compute_perplexity


@dataclass
class TrainingConfig:
    """Configuration for noise-aware training."""
    noise_level: float = 0.01
    noise_type: NoiseType = "gaussian"
    learning_rate: float = 5e-5
    epochs: int = 3
    batch_size: int = 4
    max_length: int = 128
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_samples: Optional[int] = None  # Limit samples for testing


class NoisyLinear(nn.Module):
    """
    Wrapper that adds noise to a Linear layer during forward pass.
    Used for noise-aware training.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        noise_level: float,
        noise_type: NoiseType = "gaussian"
    ):
        super().__init__()
        self.original_layer = original_layer
        self.noise_level = noise_level
        self.noise_dist = get_noise_distribution(noise_type)
        self.training_mode = True  # Only inject noise during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training_mode and self.noise_level > 0:
            # Create noisy weights for this forward pass only
            with torch.no_grad():
                weight_noise = self.noise_dist.sample(
                    self.original_layer.weight.shape,
                    self.noise_level,
                    self.original_layer.weight.device
                )

            noisy_weight = self.original_layer.weight + weight_noise

            if self.original_layer.bias is not None:
                bias_noise = self.noise_dist.sample(
                    self.original_layer.bias.shape,
                    self.noise_level,
                    self.original_layer.bias.device
                )
                noisy_bias = self.original_layer.bias + bias_noise
            else:
                noisy_bias = None

            return nn.functional.linear(x, noisy_weight, noisy_bias)
        else:
            return self.original_layer(x)

    def set_training_mode(self, mode: bool):
        """Enable/disable noise injection."""
        self.training_mode = mode


def wrap_model_with_noise(
    model: nn.Module,
    noise_level: float,
    noise_type: NoiseType = "gaussian"
) -> nn.Module:
    """
    Wrap all Linear layers in the model with NoisyLinear wrappers.

    Args:
        model: Model to wrap
        noise_level: Noise level for training
        noise_type: Type of noise distribution

    Returns:
        Model with wrapped linear layers
    """
    wrapped_count = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            # Wrap the linear layer
            wrapped = NoisyLinear(module, noise_level, noise_type)
            setattr(parent, attr_name, wrapped)
            wrapped_count += 1

    print(f"Wrapped {wrapped_count} Linear layers with noise injection")
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Remove NoisyLinear wrappers from the model.

    Args:
        model: Model with wrapped layers

    Returns:
        Model with original Linear layers
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, NoisyLinear):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            setattr(parent, attr_name, module.original_layer)

    return model


def set_model_noise_mode(model: nn.Module, training_mode: bool):
    """Enable/disable noise injection for all NoisyLinear layers."""
    for module in model.modules():
        if isinstance(module, NoisyLinear):
            module.set_training_mode(training_mode)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        for text in texts:
            if len(text.strip()) > 20:  # Skip very short texts
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return {
            "input_ids": item["input_ids"].squeeze(),
            "attention_mask": item["attention_mask"].squeeze(),
            "labels": item["input_ids"].squeeze()
        }


def train_noise_aware(
    model_name: str = "distilgpt2",
    config: TrainingConfig = None,
    device: str = "auto"
) -> Dict:
    """
    Train a model with noise-aware training.

    Args:
        model_name: HuggingFace model name
        config: Training configuration
        device: Compute device

    Returns:
        Dictionary with trained model and training metrics
    """
    if config is None:
        config = TrainingConfig()

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Load training data
    print("Loading training data (WikiText-2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if item["text"].strip()]

    if config.max_samples:
        texts = texts[:config.max_samples]

    train_dataset = TextDataset(texts, tokenizer, config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    print(f"Training samples: {len(train_dataset)}")

    # Wrap model with noise injection
    print(f"\nWrapping model with noise level σ = {config.noise_level}")
    model = wrap_model_with_noise(model, config.noise_level, config.noise_type)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    training_losses = []
    model.train()

    print(f"\n{'='*50}")
    print(f"NOISE-AWARE TRAINING")
    print(f"Epochs: {config.epochs}, Noise Level: {config.noise_level}")
    print(f"{'='*50}\n")

    global_step = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass (noise is injected automatically by NoisyLinear)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += outputs.loss.item()
            progress_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

    # Unwrap model for clean weights
    model = unwrap_model(model)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "training_losses": training_losses,
        "config": config
    }


def compare_robustness(
    baseline_model: nn.Module,
    noise_trained_model: nn.Module,
    tokenizer,
    noise_levels: np.ndarray,
    device: torch.device
) -> Dict:
    """
    Compare noise robustness between baseline and noise-trained models.

    Args:
        baseline_model: Original model (not noise-trained)
        noise_trained_model: Model trained with noise injection
        tokenizer: Tokenizer for both models
        noise_levels: Noise levels to test
        device: Compute device

    Returns:
        Dictionary with comparison results
    """
    # Load validation text
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = ""
    for item in dataset:
        if item["text"].strip():
            text += item["text"] + " "
            if len(text) >= 500:
                break
    validation_text = text[:500].strip()

    results = {
        "baseline": [],
        "noise_trained": []
    }

    from noise_distributions import inject_thermal_noise

    print("\nComparing robustness across noise levels...")

    for noise_level in tqdm(noise_levels, desc="Testing"):
        # Test baseline
        baseline_copy = copy.deepcopy(baseline_model)
        baseline_copy.to(device)
        inject_thermal_noise(baseline_copy, noise_level, "gaussian", in_place=True)
        baseline_ppl = compute_perplexity(baseline_copy, tokenizer, validation_text, device)
        results["baseline"].append(baseline_ppl)
        del baseline_copy

        # Test noise-trained
        trained_copy = copy.deepcopy(noise_trained_model)
        trained_copy.to(device)
        inject_thermal_noise(trained_copy, noise_level, "gaussian", in_place=True)
        trained_ppl = compute_perplexity(trained_copy, tokenizer, validation_text, device)
        results["noise_trained"].append(trained_ppl)
        del trained_copy

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def plot_robustness_comparison(
    results: Dict,
    noise_levels: np.ndarray,
    save_path: str = "images/noise_aware_training.png"
) -> None:
    """
    Plot robustness comparison between baseline and noise-trained models.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.semilogy(noise_levels, results["baseline"], 'b-o',
                linewidth=2, markersize=8, label='Baseline Model')
    ax.semilogy(noise_levels, results["noise_trained"], 'g-s',
                linewidth=2, markersize=8, label='Noise-Aware Trained')

    ax.set_xlabel('Noise Level (σ)', fontsize=14)
    ax.set_ylabel('Perplexity (log scale)', fontsize=14)
    ax.set_title('Noise-Aware Training: Improved Robustness\n'
                 'Model trained with noise learns fault-tolerant representations',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(0, noise_levels[-1])

    # Calculate improvement
    baseline_cliff_idx = next((i for i, p in enumerate(results["baseline"]) if p > 1000), -1)
    trained_cliff_idx = next((i for i, p in enumerate(results["noise_trained"]) if p > 1000), -1)

    if baseline_cliff_idx > 0 and trained_cliff_idx > 0:
        improvement = noise_levels[trained_cliff_idx] / noise_levels[baseline_cliff_idx]
        ax.text(0.5, 0.02,
                f'Noise tolerance improved by {improvement:.1f}x!',
                transform=ax.transAxes, fontsize=12, ha='center', style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Noise-Aware Fine-tuning for Analog Hardware Robustness"
    )
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--noise_level", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="images/noise_aware_training.png")

    args = parser.parse_args()

    print("🧬 Noise-Aware Fine-tuning")
    print("=" * 60)
    print("Training model to be robust to analog hardware noise")
    print("=" * 60)

    config = TrainingConfig(
        noise_level=args.noise_level,
        epochs=args.epochs,
        max_samples=args.max_samples
    )

    # Train noise-aware model
    result = train_noise_aware(args.model, config, args.device)
    noise_trained_model = result["model"]
    tokenizer = result["tokenizer"]

    # Load fresh baseline for comparison
    print("\nLoading baseline model for comparison...")
    baseline_model = AutoModelForCausalLM.from_pretrained(args.model)

    device = next(noise_trained_model.parameters()).device
    baseline_model.to(device)
    baseline_model.eval()

    # Compare robustness
    noise_levels = np.linspace(0, 0.05, 10)
    comparison = compare_robustness(
        baseline_model,
        noise_trained_model,
        tokenizer,
        noise_levels,
        device
    )

    plot_robustness_comparison(comparison, noise_levels, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("NOISE-AWARE TRAINING SUMMARY")
    print("=" * 60)
    print(f"Training noise level: σ = {config.noise_level}")
    print(f"Training epochs: {config.epochs}")
    print(f"Final training loss: {result['training_losses'][-1]:.4f}")

    # Calculate improvement
    for i, sigma in enumerate(noise_levels):
        if comparison["baseline"][i] > 100 and comparison["noise_trained"][i] < 100:
            print(f"\nAt σ = {sigma:.4f}:")
            print(f"  Baseline PPL:      {comparison['baseline'][i]:.2f}")
            print(f"  Noise-Trained PPL: {comparison['noise_trained'][i]:.2f}")
            print(f"  Improvement:       {comparison['baseline'][i]/comparison['noise_trained'][i]:.1f}x better!")
            break

    print("\n✓ Noise-aware training complete!")


if __name__ == "__main__":
    main()
