#!/usr/bin/env python3
"""
Noisy Neural Network - Advanced Interactive Demo

Features:
1. Anatomical Toggles (Attention vs MLP)
2. Live Probability Distribution Chart
3. Thermal Drift Mode (Simulated Runaway)
4. Analog Efficiency Estimator
5. Side-by-Side Digital vs Analog Comparison
6. 4-bit Digital Quantization vs Analog Noise
7. CSV Experiment Export
8. Entropy over Time Visualization
"""

import copy
import csv
import io
from typing import Tuple, List, Dict
import warnings
from datetime import datetime

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from noise_distributions import get_noise_distribution, NoiseType

warnings.filterwarnings("ignore")

# Global model cache and experiment log
MODEL_CACHE = {}
EXPERIMENT_LOG = []


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = "distilgpt2"):
    """Load and cache the model."""
    if model_name not in MODEL_CACHE:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(get_device())
        model.eval()

        MODEL_CACHE[model_name] = (model, tokenizer)
        print(f"✓ {model_name} loaded on {get_device()}")

    return MODEL_CACHE[model_name]


def quantize_weights_4bit(model: nn.Module) -> None:
    """
    Apply 4-bit quantization to model weights (simulated).
    Rounds weights to 16 discrete levels.
    """
    try:
        from transformers.pytorch_utils import Conv1D
        has_conv1d = True
    except ImportError:
        has_conv1d = False
        Conv1D = None

    n_levels = 16  # 4-bit = 2^4 = 16 levels

    with torch.no_grad():
        for name, module in model.named_modules():
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = has_conv1d and isinstance(module, Conv1D)

            if is_linear or is_conv1d:
                w = module.weight
                # Scale to [-1, 1] range
                w_min, w_max = w.min(), w.max()
                w_norm = (w - w_min) / (w_max - w_min + 1e-8)
                # Quantize to n_levels
                w_quant = torch.round(w_norm * (n_levels - 1)) / (n_levels - 1)
                # Scale back
                w_new = w_quant * (w_max - w_min) + w_min
                module.weight.copy_(w_new)


def inject_noise_selective(
    model: nn.Module,
    noise_level: float,
    noise_type: str,
    apply_attention: bool,
    apply_mlp: bool
) -> None:
    """Inject noise selectively into Attention and/or MLP layers."""
    # Handle 4-bit quantization separately
    if noise_type == "quantize_4bit":
        quantize_weights_4bit(model)
        return

    try:
        from transformers.pytorch_utils import Conv1D
        has_conv1d = True
    except ImportError:
        has_conv1d = False
        Conv1D = None

    if noise_level == 0 or (not apply_attention and not apply_mlp):
        return

    noise_dist = get_noise_distribution(noise_type)

    with torch.no_grad():
        for name, module in model.named_modules():
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = has_conv1d and isinstance(module, Conv1D)

            if is_linear or is_conv1d:
                is_attn = ".attn." in name
                is_mlp_layer = ".mlp." in name

                should_noise = (is_attn and apply_attention) or (is_mlp_layer and apply_mlp)

                if should_noise:
                    noise = noise_dist.sample(
                        module.weight.shape,
                        noise_level,
                        module.weight.device
                    )
                    module.weight.add_(noise)


def calculate_energy_savings(
    noise_level: float,
    noise_type: str,
    apply_attention: bool,
    apply_mlp: bool
) -> Tuple[float, str]:
    """Estimate energy savings from analog computation."""
    if noise_level == 0 and noise_type != "quantize_4bit":
        return 0.0, "0% Savings (Traditional Digital)"

    # 4-bit quantization has different savings profile
    if noise_type == "quantize_4bit":
        savings = 45 if apply_mlp else 25
        return savings, f"{savings}% Savings (4-bit Digital Quantization)"

    base_savings = 0
    if apply_mlp:
        base_savings += 40
    if apply_attention:
        base_savings += 20

    noise_multiplier = {"gaussian": 1.0, "uniform": 0.8, "cauchy": 0.5}.get(noise_type, 0.5)
    noise_bonus = min(noise_level * 1000, 30)

    total_savings = min(base_savings * noise_multiplier + noise_bonus, 85)

    if total_savings > 60:
        status = f"{total_savings:.0f}% Savings (Highly Efficient Analog)"
    elif total_savings > 30:
        status = f"{total_savings:.0f}% Savings (Hybrid Digital-Analog)"
    else:
        status = f"{total_savings:.0f}% Savings (Mostly Digital)"

    return total_savings, status


def create_probability_chart(probs: List[Tuple[str, float]], title: str = "Token Probabilities"):
    """Create a bar chart of top token probabilities."""
    fig, ax = plt.subplots(figsize=(5, 2.5))

    if not probs:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        tokens = [p[0][:12] for p in probs]
        values = [p[1] * 100 for p in probs]

        colors = ['#4CAF50' if v > 50 else '#FFC107' if v > 20 else '#F44336' for v in values]

        bars = ax.barh(range(len(tokens)), values, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel('Probability (%)', fontsize=9)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()

        for bar, val in zip(bars, values):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=8)

    ax.set_title(title, fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig


def create_entropy_chart(entropies: List[float], sigmas: List[float]):
    """Create entropy over time chart for thermal drift."""
    fig, ax = plt.subplots(figsize=(5, 2.5))

    if not entropies:
        ax.text(0.5, 0.5, "Enable Thermal Runaway", ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        steps = list(range(len(entropies)))

        ax.plot(steps, entropies, 'b-', linewidth=2, label='Entropy')
        ax.fill_between(steps, entropies, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(steps, sigmas, 'r--', linewidth=1, alpha=0.7, label='sigma')
        ax2.set_ylabel('sigma', color='red', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_xlabel('Token Position', fontsize=9)
        ax.set_ylabel('Entropy (bits)', fontsize=9)
        ax.set_title('Entropy Over Time', fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    return fig


def generate_with_features(
    prompt: str,
    noise_level: float,
    noise_type: str,
    apply_attention: bool,
    apply_mlp: bool,
    thermal_drift: bool,
    drift_rate: float,
    max_tokens: int,
    temperature: float
) -> Tuple[str, str, str, str, str, plt.Figure, plt.Figure]:
    """Generate text with all advanced features."""
    global EXPERIMENT_LOG

    base_model, tokenizer = load_model("distilgpt2")
    device = get_device()

    # Generate baseline (clean)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        baseline_outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(0.1, temperature),
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)

    # Ensure we have baseline text
    if not baseline_text.strip():
        baseline_text = prompt + " [Generation produced empty output]"

    # Create noisy model
    noisy_model = copy.deepcopy(base_model)
    noisy_model.to(device)

    # Track data
    last_token_probs = []
    entropies = []
    sigmas = []

    # Generate with noise (token by token)
    input_ids = inputs.input_ids.clone()
    current_sigma = noise_level

    try:
        for step in range(max_tokens):
            # Apply thermal drift
            if thermal_drift and step > 0:
                current_sigma = min(noise_level + step * drift_rate, 0.05)

            sigmas.append(current_sigma)

            # For drift, recreate model each step; for static, only once
            if step == 0 or thermal_drift:
                if thermal_drift and step > 0:
                    noisy_model = copy.deepcopy(base_model)
                    noisy_model.to(device)

                inject_noise_selective(
                    noisy_model, current_sigma, noise_type,
                    apply_attention, apply_mlp
                )

            # Get logits
            with torch.no_grad():
                outputs = noisy_model(input_ids)
                logits = outputs.logits[:, -1, :]

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    # Model collapsed - fill remaining with marker
                    entropies.extend([0.0] * (max_tokens - step))
                    sigmas.extend([current_sigma] * (max_tokens - step - 1))
                    break

                logits = logits / max(0.1, temperature)
                probs = F.softmax(logits, dim=-1)

                # Calculate entropy
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1).item()
                entropies.append(entropy)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Track last token probabilities at midpoint
                if step == max_tokens // 2:
                    top_probs, top_indices = torch.topk(probs[0], k=5)
                    last_token_probs = [
                        (tokenizer.decode([idx.item()]).strip() or "[space]", p.item())
                        for idx, p in zip(top_indices, top_probs)
                    ]

        noisy_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Ensure we have noisy text
        if not noisy_text.strip():
            noisy_text = prompt + " [Generation produced empty output - try lower noise]"

    except Exception as e:
        noisy_text = f"{prompt}... [GENERATION FAILED: {str(e)[:50]}]"
        last_token_probs = [("Error", 1.0)]
        if not entropies:
            entropies = [0.0]
            sigmas = [current_sigma]

    # Clean up
    del noisy_model

    # Determine quality
    effective_sigma = current_sigma if thermal_drift else noise_level

    if noise_type == "quantize_4bit":
        quality = "🔢 QUANTIZED - 4-bit digital precision"
        heat_bar = "🟦" * 10
    elif effective_sigma == 0 or (not apply_attention and not apply_mlp):
        quality = "✅ COHERENT - Baseline output"
        heat_bar = "🟢" * 10
    elif effective_sigma < 0.005:
        quality = "✅ COHERENT - Slight variations"
        heat_bar = "🟢" * 8 + "🟡" * 2
    elif effective_sigma < 0.01:
        quality = "⚡ CREATIVE JITTER - Interesting variations"
        heat_bar = "🟢" * 5 + "🟡" * 3 + "🟠" * 2
    elif effective_sigma < 0.02:
        quality = "⚠️ DEGRADED - Repetitive patterns"
        heat_bar = "🟡" * 4 + "🟠" * 4 + "🔴" * 2
    elif effective_sigma < 0.03:
        quality = "❌ BREAKDOWN - Severe degradation"
        heat_bar = "🟠" * 3 + "🔴" * 7
    else:
        quality = "💀 GIBBERISH - Model collapse"
        heat_bar = "🔴" * 10

    # Energy savings
    _, energy_status = calculate_energy_savings(
        noise_level, noise_type, apply_attention, apply_mlp
    )

    # Create charts
    chart_title = f"Token Probs @ step {max_tokens//2}"
    prob_chart = create_probability_chart(last_token_probs, chart_title)
    entropy_chart = create_entropy_chart(entropies if thermal_drift else [], sigmas if thermal_drift else [])

    # Log experiment
    avg_entropy = np.mean(entropies) if entropies else 0
    EXPERIMENT_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt[:50],
        "sigma": noise_level,
        "noise_type": noise_type,
        "apply_attention": apply_attention,
        "apply_mlp": apply_mlp,
        "thermal_drift": thermal_drift,
        "avg_entropy": round(avg_entropy, 3),
        "quality": quality.split(" - ")[0],
        "baseline_text": baseline_text[:100],
        "noisy_text": noisy_text[:100]
    })

    return baseline_text, noisy_text, quality, heat_bar, energy_status, prob_chart, entropy_chart


def export_experiment_log():
    """Export experiment log as CSV."""
    global EXPERIMENT_LOG

    if not EXPERIMENT_LOG:
        return None

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=EXPERIMENT_LOG[0].keys())
    writer.writeheader()
    writer.writerows(EXPERIMENT_LOG)

    # Create temporary file
    csv_content = output.getvalue()
    output.close()

    # Save to temp file
    filename = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w') as f:
        f.write(csv_content)

    return filename


def create_demo():
    """Create the advanced Gradio demo interface."""

    load_model("distilgpt2")

    css = """
    .heat-bar { font-size: 20px; letter-spacing: 2px; }
    .quality-indicator {
        font-size: 14px; font-weight: bold; padding: 6px;
        border-radius: 6px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .output-text {
        font-family: 'Courier New', monospace; background: #0d0d0d;
        border: 1px solid #333; border-radius: 6px; padding: 8px; font-size: 12px;
    }
    .energy-display {
        font-size: 12px; font-weight: bold; color: #4CAF50;
        background: #1a3a1a; padding: 6px; border-radius: 6px;
    }
    """

    with gr.Blocks(css=css, title="🧠 Noisy Neural Network", theme=gr.themes.Base()) as demo:

        gr.Markdown("""
        # 🧠 Noisy Neural Network
        **Interactive Analog Hardware Noise Simulator** | [GitHub](https://github.com/pranshug2704/NoisyNeuralNetwork)
        """)

        with gr.Row():
            # LEFT COLUMN: Controls
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Input Prompt",
                    value="The future of artificial intelligence is",
                    lines=2
                )

                noise_slider = gr.Slider(
                    minimum=0.0, maximum=0.05, value=0.0, step=0.001,
                    label="🌡️ Voltage / Heat Level (σ)"
                )

                noise_type = gr.Radio(
                    choices=["gaussian", "uniform", "cauchy", "quantize_4bit"],
                    value="gaussian", label="Noise Type",
                    info="Analog (gaussian/uniform/cauchy) vs Digital (4-bit)"
                )

                gr.Markdown("#### 🧬 Anatomical Targeting")
                with gr.Row():
                    apply_attention = gr.Checkbox(label="Attention", value=True)
                    apply_mlp = gr.Checkbox(label="MLP", value=True)

                thermal_drift = gr.Checkbox(
                    label="🔥 Thermal Runaway",
                    value=False,
                    info="Simulate chip overheating over time"
                )
                drift_rate = gr.Slider(
                    minimum=0.0001, maximum=0.005, value=0.001, step=0.0001,
                    label="Thermal Drift Rate (σ/token)",
                    info="How fast the chip heats up"
                )

                with gr.Row():
                    max_tokens = gr.Slider(20, 60, value=35, step=5, label="Tokens")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temp")

                generate_btn = gr.Button("🔥 Generate", variant="primary")

                energy_display = gr.Textbox(
                    label="⚡ Energy Efficiency",
                    value="0% Savings (Digital)",
                    interactive=False,
                    elem_classes=["energy-display"]
                )

            # RIGHT COLUMN: Outputs
            with gr.Column(scale=2):
                heat_bar = gr.Textbox(
                    label="Heat", value="🟢" * 10,
                    interactive=False, elem_classes=["heat-bar"]
                )

                quality = gr.Textbox(
                    label="Quality",
                    value="✅ COHERENT - Baseline",
                    interactive=False, elem_classes=["quality-indicator"]
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🖥️ Digital Baseline**")
                        baseline_output = gr.Textbox(
                            lines=4, interactive=False,
                            elem_classes=["output-text"]
                        )
                    with gr.Column():
                        gr.Markdown("**🔌 Analog/Quantized**")
                        noisy_output = gr.Textbox(
                            lines=4, interactive=False,
                            elem_classes=["output-text"]
                        )

                with gr.Row():
                    prob_chart = gr.Plot(label="Token Probabilities")
                    entropy_chart = gr.Plot(label="Entropy Over Time")

        with gr.Accordion("📖 About the Physics", open=False):
            gr.Markdown("""
            | Noise Type | What It Simulates |
            |------------|-------------------|
            | **Gaussian** | Thermal noise in analog circuits (most realistic) |
            | **Uniform** | DAC quantization errors |
            | **Cauchy** | Heavy-tailed outlier events |
            | **4-bit Quantize** | Digital low-precision (like GPTQ) |

            **Key Insight**: MLP layers tolerate ~2x more noise than Attention layers!
            """)

        # Event handlers
        generate_btn.click(
            fn=generate_with_features,
            inputs=[prompt, noise_slider, noise_type, apply_attention,
                   apply_mlp, thermal_drift, drift_rate, max_tokens, temperature],
            outputs=[baseline_output, noisy_output, quality, heat_bar,
                    energy_display, prob_chart, entropy_chart]
        )



        def update_energy(sigma, noise_type, attn, mlp):
            _, status = calculate_energy_savings(sigma, noise_type, attn, mlp)
            return status

        for control in [noise_slider, apply_attention, apply_mlp, noise_type]:
            control.change(
                fn=update_energy,
                inputs=[noise_slider, noise_type, apply_attention, apply_mlp],
                outputs=[energy_display]
            )

    return demo


if __name__ == "__main__":
    print("🧠 Starting Advanced Noisy Neural Network Demo...")
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
