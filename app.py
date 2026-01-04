#!/usr/bin/env python3
"""
Noisy Neural Network - Advanced Interactive Demo

Features:
1. Anatomical Toggles (Attention vs MLP)
2. Live Probability Distribution Chart
3. Thermal Drift Mode (Simulated Runaway)
4. Analog Efficiency Estimator
5. Side-by-Side Digital vs Analog Comparison
"""

import copy
from typing import Tuple, List, Dict
import warnings

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

# Global model cache
MODEL_CACHE = {}


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


def inject_noise_selective(
    model: nn.Module,
    noise_level: float,
    noise_type: str,
    apply_attention: bool,
    apply_mlp: bool
) -> None:
    """Inject noise selectively into Attention and/or MLP layers."""
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
                # Determine layer type
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
    """
    Estimate energy savings from analog computation.
    Higher noise tolerance = more analog = more savings.
    """
    if noise_level == 0:
        return 0.0, "0% Savings (Traditional Digital)"

    # Base savings from using analog
    base_savings = 0

    # MLP layers are 2x larger than attention, so more savings
    if apply_mlp:
        base_savings += 40  # MLP is the big win
    if apply_attention:
        base_savings += 20  # Attention is smaller

    # Gaussian noise (thermal) is the most realistic analog simulation
    noise_multiplier = {"gaussian": 1.0, "uniform": 0.8, "cauchy": 0.5}.get(noise_type, 0.5)

    # Higher noise tolerance = better efficiency
    noise_bonus = min(noise_level * 1000, 30)  # Cap at 30%

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
    fig, ax = plt.subplots(figsize=(6, 3))

    if not probs:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        tokens = [p[0][:15] for p in probs]  # Truncate long tokens
        values = [p[1] * 100 for p in probs]

        colors = ['#4CAF50' if v > 50 else '#FFC107' if v > 20 else '#F44336' for v in values]

        bars = ax.barh(range(len(tokens)), values, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=10)
        ax.set_xlabel('Probability (%)', fontsize=10)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()

        # Add percentage labels
        for bar, val in zip(bars, values):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=9)

    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def generate_with_features(
    prompt: str,
    noise_level: float,
    noise_type: str,
    apply_attention: bool,
    apply_mlp: bool,
    thermal_drift: bool,
    max_tokens: int,
    temperature: float
) -> Tuple[str, str, str, str, str, str, plt.Figure]:
    """
    Generate text with all advanced features.

    Returns:
        - baseline_text: Clean baseline output
        - noisy_text: Analog simulation output
        - quality: Quality assessment
        - heat_bar: Visual heat indicator
        - energy_status: Energy savings estimate
        - top_probs_chart: Matplotlib figure
    """
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

    # Create noisy model
    noisy_model = copy.deepcopy(base_model)
    noisy_model.to(device)

    # Track probabilities for last token
    last_token_probs = []

    # Generate with noise (token by token for thermal drift + probability tracking)
    input_ids = inputs.input_ids.clone()
    current_sigma = noise_level

    try:
        for step in range(max_tokens):
            # Apply thermal drift
            if thermal_drift and step > 0:
                current_sigma = min(noise_level + step * 0.001, 0.05)

            # Re-inject noise for thermal drift, or just once for static
            if step == 0 or thermal_drift:
                if thermal_drift and step > 0:
                    # For drift, we need fresh model each step
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
                    break

                # Apply temperature
                logits = logits / max(0.1, temperature)
                probs = F.softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Track last token's top probabilities
                if step == max_tokens - 1 or step == max_tokens // 2:
                    top_probs, top_indices = torch.topk(probs[0], k=5)
                    last_token_probs = [
                        (tokenizer.decode([idx.item()]).strip() or "[empty]", p.item())
                        for idx, p in zip(top_indices, top_probs)
                    ]

        noisy_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    except Exception as e:
        noisy_text = f"{prompt}... [GENERATION FAILED: {str(e)[:50]}]"
        last_token_probs = [("Error", 1.0)]

    # Clean up
    del noisy_model

    # Determine quality indicator
    effective_sigma = current_sigma if thermal_drift else noise_level

    if effective_sigma == 0 or (not apply_attention and not apply_mlp):
        quality = "✅ COHERENT - Baseline output"
        heat_bar = "🟢" * 10
    elif effective_sigma < 0.005:
        quality = "✅ COHERENT - Slight variations"
        heat_bar = "🟢" * 8 + "🟡" * 2
    elif effective_sigma < 0.01:
        quality = "⚡ CREATIVE JITTER - Interesting variations"
        heat_bar = "🟢" * 5 + "🟡" * 3 + "🟠" * 2
    elif effective_sigma < 0.02:
        quality = "⚠️ DEGRADED - Repetitive patterns emerging"
        heat_bar = "🟡" * 4 + "🟠" * 4 + "🔴" * 2
    elif effective_sigma < 0.03:
        quality = "❌ BREAKDOWN - Severe degradation"
        heat_bar = "🟠" * 3 + "🔴" * 7
    else:
        quality = "💀 GIBBERISH - Complete model collapse"
        heat_bar = "🔴" * 10

    # Calculate energy savings
    _, energy_status = calculate_energy_savings(
        noise_level, noise_type, apply_attention, apply_mlp
    )

    # Create probability chart
    chart_title = "Token Probabilities (Last Token)"
    if thermal_drift:
        chart_title = f"Probabilities @ σ={current_sigma:.3f}"
    prob_chart = create_probability_chart(last_token_probs, chart_title)

    return baseline_text, noisy_text, quality, heat_bar, energy_status, prob_chart


def create_demo():
    """Create the advanced Gradio demo interface."""

    # Preload model
    load_model("distilgpt2")

    css = """
    .heat-bar { font-size: 24px; letter-spacing: 2px; }
    .quality-indicator {
        font-size: 16px; font-weight: bold; padding: 8px;
        border-radius: 8px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .output-text {
        font-family: 'Courier New', monospace; background: #0d0d0d;
        border: 1px solid #333; border-radius: 8px; padding: 10px;
    }
    .energy-display {
        font-size: 14px; font-weight: bold; color: #4CAF50;
        background: #1a3a1a; padding: 8px; border-radius: 8px;
    }
    """

    with gr.Blocks(css=css, title="🧠 Noisy Neural Network", theme=gr.themes.Base()) as demo:

        gr.Markdown("""
        # 🧠 Noisy Neural Network
        ## Interactive Analog Hardware Noise Simulator

        **Turn up the heat and watch the AI's mind melt!**
        Simulate thermal noise in analog AI chips and observe real-time degradation.

        [GitHub](https://github.com/pranshug2704/NoisyNeuralNetwork)
        """)

        with gr.Row():
            # LEFT COLUMN: Controls
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Controls")

                prompt = gr.Textbox(
                    label="Input Prompt",
                    value="The future of artificial intelligence is",
                    lines=2
                )

                noise_slider = gr.Slider(
                    minimum=0.0, maximum=0.05, value=0.0, step=0.001,
                    label="🌡️ Voltage / Heat Level (σ)",
                    info="0 = Cool baseline, 0.05 = Overheating"
                )

                gr.Markdown("#### 🧬 Anatomical Targeting")
                with gr.Row():
                    apply_attention = gr.Checkbox(label="Apply to Attention", value=True)
                    apply_mlp = gr.Checkbox(label="Apply to MLP", value=True)

                noise_type = gr.Radio(
                    choices=["gaussian", "uniform", "cauchy"],
                    value="gaussian", label="Noise Distribution"
                )

                thermal_drift = gr.Checkbox(
                    label="🔥 Simulate Thermal Runaway",
                    value=False,
                    info="σ increases by +0.001 per token (chip overheating)"
                )

                with gr.Row():
                    max_tokens = gr.Slider(20, 80, value=40, step=10, label="Max Tokens")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")

                generate_btn = gr.Button("🔥 Generate with Noise", variant="primary")

                # Energy Savings Display
                energy_display = gr.Textbox(
                    label="⚡ Analog Efficiency",
                    value="0% Savings (Traditional Digital)",
                    interactive=False,
                    elem_classes=["energy-display"]
                )

            # RIGHT COLUMN: Outputs
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Output Comparison")

                heat_bar = gr.Textbox(
                    label="Heat Indicator", value="🟢" * 10,
                    interactive=False, elem_classes=["heat-bar"]
                )

                quality = gr.Textbox(
                    label="Quality Assessment",
                    value="✅ COHERENT - Baseline output",
                    interactive=False, elem_classes=["quality-indicator"]
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🖥️ Digital Baseline (Perfect)**")
                        baseline_output = gr.Textbox(
                            lines=5, interactive=False,
                            elem_classes=["output-text"]
                        )
                    with gr.Column():
                        gr.Markdown("**🔌 Analog Simulation (Noisy)**")
                        noisy_output = gr.Textbox(
                            lines=5, interactive=False,
                            elem_classes=["output-text"]
                        )

                gr.Markdown("### 📈 Token Probability Distribution")
                prob_chart = gr.Plot(label="Top 5 Token Probabilities")

        # Collapsible documentation
        with gr.Accordion("📖 About the Physics", open=False):
            gr.Markdown("""
            ### How It Works

            This demo injects noise directly into neural network **weights**:

            $$W_{noisy} = W_{original} + \\mathcal{N}(0, \\sigma^2)$$

            | Control | What It Does |
            |---------|--------------|
            | **σ (Sigma)** | Noise magnitude - higher = more weight perturbation |
            | **Attention Toggle** | Apply noise to Query/Key/Value projections |
            | **MLP Toggle** | Apply noise to feed-forward layers (2x larger!) |
            | **Thermal Runaway** | σ increases +0.001 per token (simulates heating) |

            ### Key Insights

            - **MLP is MORE noise-tolerant** than Attention (hardware implication!)
            - **Goldilocks Zone** at σ=0.001: slightly MORE creative output
            - **Entropy Paradox**: High noise → model becomes MORE confident (wrong!)
            """)

        # Event handlers
        generate_btn.click(
            fn=generate_with_features,
            inputs=[prompt, noise_slider, noise_type, apply_attention,
                   apply_mlp, thermal_drift, max_tokens, temperature],
            outputs=[baseline_output, noisy_output, quality, heat_bar,
                    energy_display, prob_chart]
        )

        # Auto-update energy display on control changes
        def update_energy(sigma, noise_type, attn, mlp):
            _, status = calculate_energy_savings(sigma, noise_type, attn, mlp)
            return status

        for control in [noise_slider, apply_attention, apply_mlp]:
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
