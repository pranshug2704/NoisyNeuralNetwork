#!/usr/bin/env python3
"""
Noisy Neural Network - Interactive Demo

A Gradio interface to visualize how analog hardware noise affects LLM output.
Turn the "Voltage/Heat Slider" and watch the text break in real-time!
"""

import copy
from typing import Tuple
import warnings

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from noise_distributions import inject_thermal_noise, NoiseType

warnings.filterwarnings("ignore")


# Global model cache (loaded once)
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


def generate_with_noise(
    prompt: str,
    noise_level: float,
    noise_type: str,
    max_tokens: int,
    temperature: float
) -> Tuple[str, str, str]:
    """
    Generate text with specified noise level.

    Returns:
        Tuple of (generated_text, quality_indicator, noise_emoji_bar)
    """
    base_model, tokenizer = load_model("distilgpt2")
    device = get_device()

    # Create noisy copy
    noisy_model = copy.deepcopy(base_model)
    noisy_model.to(device)

    # Inject noise
    inject_thermal_noise(noisy_model, noise_level, noise_type, in_place=True)

    # Generate text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            outputs = noisy_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.1, temperature),
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        generated_text = f"{prompt}... [GENERATION FAILED: Model collapsed due to extreme noise]"

    # Clean up
    del noisy_model

    # Determine quality indicator
    if noise_level == 0:
        quality = "✅ COHERENT - Baseline output"
        heat_bar = "🟢" * 10
    elif noise_level < 0.005:
        quality = "✅ COHERENT - Slight variations"
        heat_bar = "🟢" * 8 + "🟡" * 2
    elif noise_level < 0.01:
        quality = "⚡ CREATIVE JITTER - Interesting variations"
        heat_bar = "🟢" * 5 + "🟡" * 3 + "🟠" * 2
    elif noise_level < 0.02:
        quality = "⚠️ DEGRADED - Repetitive patterns emerging"
        heat_bar = "🟡" * 4 + "🟠" * 4 + "🔴" * 2
    elif noise_level < 0.03:
        quality = "❌ BREAKDOWN - Severe degradation"
        heat_bar = "🟠" * 3 + "🔴" * 7
    else:
        quality = "💀 GIBBERISH - Complete model collapse"
        heat_bar = "🔴" * 10

    return generated_text, quality, heat_bar


def create_demo():
    """Create the Gradio demo interface."""

    # Preload model
    load_model("distilgpt2")

    # Custom CSS for styling
    css = """
    .heat-bar {
        font-size: 24px;
        letter-spacing: 2px;
    }
    .quality-indicator {
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .output-text {
        font-family: 'Courier New', monospace;
        background: #0d0d0d;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
    }
    """

    with gr.Blocks(css=css, title="🧠 Noisy Neural Network", theme=gr.themes.Base()) as demo:

        gr.Markdown("""
        # 🧠 Noisy Neural Network
        ## Interactive Analog Hardware Noise Simulator

        **Turn up the heat and watch the AI's mind melt!**

        This demo simulates how thermal noise in analog AI chips affects language model output.
        Move the slider to increase "Voltage/Heat" and observe real-time degradation.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Controls")

                prompt = gr.Textbox(
                    label="Input Prompt",
                    value="The future of artificial intelligence is",
                    lines=2,
                    placeholder="Enter your prompt here..."
                )

                noise_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.05,
                    value=0.0,
                    step=0.001,
                    label="🌡️ Voltage / Heat Level (σ)",
                    info="0 = Cool baseline, 0.05 = Overheating"
                )

                noise_type = gr.Radio(
                    choices=["gaussian", "uniform", "cauchy"],
                    value="gaussian",
                    label="Noise Distribution",
                    info="Gaussian = thermal, Uniform = quantization, Cauchy = outliers"
                )

                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=10,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.1,
                        label="Temperature"
                    )

                generate_btn = gr.Button("🔥 Generate with Noise", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Output")

                heat_bar = gr.Textbox(
                    label="Heat Indicator",
                    value="🟢" * 10,
                    interactive=False,
                    elem_classes=["heat-bar"]
                )

                quality = gr.Textbox(
                    label="Quality Assessment",
                    value="✅ COHERENT - Baseline output",
                    interactive=False,
                    elem_classes=["quality-indicator"]
                )

                output = gr.Textbox(
                    label="Generated Text",
                    lines=8,
                    interactive=False,
                    elem_classes=["output-text"]
                )

        gr.Markdown("""
        ---
        ### 📖 Controls Guide

        | Control | What It Does | Recommended Values |
        |---------|--------------|-------------------|
        | **🌡️ Voltage/Heat (σ)** | Simulates thermal noise magnitude. Higher = more weight perturbation. | 0-0.005 for coherent, 0.01-0.02 for creative, >0.03 for breakdown |
        | **Noise Distribution** | Type of random noise injected into weights | **Gaussian**: thermal noise (most realistic). **Uniform**: quantization errors. **Cauchy**: extreme outliers |
        | **Max Tokens** | How many words/tokens to generate | 50 is good for demos, increase for longer text |
        | **Temperature** | Controls randomness in token selection (separate from noise) | 0.8 is balanced, lower = deterministic, higher = creative |

        ---
        ### 🔬 The Science

        This demo injects noise directly into the neural network **weights** using:

        $$W_{noisy} = W_{original} + \\mathcal{N}(0, \\sigma^2)$$

        Where σ (sigma) represents the "Thermal Temperature" of analog hardware.

        | Noise Level | Effect | Hardware Analogy |
        |-------------|--------|------------------|
        | σ = 0.000 | Perfect output | Cool silicon |
        | σ = 0.001 | **Goldilocks Zone** - slightly MORE creative! | Warm processor |
        | σ = 0.010 | Repetitive patterns emerge | Hot chip |
        | σ = 0.020 | Major degradation | Overheating |
        | σ = 0.050 | Complete gibberish | Thermal runaway 🔥 |

        ---
        ### 🔗 Links

        **GitHub**: [pranshug2704/NoisyNeuralNetwork](https://github.com/pranshug2704/NoisyNeuralNetwork)

        Built with 🧠 by exploring how analog AI chips handle noise.
        """)

        # Event handlers
        generate_btn.click(
            fn=generate_with_noise,
            inputs=[prompt, noise_slider, noise_type, max_tokens, temperature],
            outputs=[output, quality, heat_bar]
        )

        # Auto-generate on slider change
        noise_slider.change(
            fn=generate_with_noise,
            inputs=[prompt, noise_slider, noise_type, max_tokens, temperature],
            outputs=[output, quality, heat_bar]
        )

    return demo


if __name__ == "__main__":
    print("🧠 Starting Noisy Neural Network Demo...")
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
