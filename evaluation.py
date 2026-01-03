"""
Evaluation utilities for measuring LLM performance under noise.

This module provides functions for:
- Computing perplexity on validation text
- Generating text samples to qualitatively assess output quality
"""

from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: torch.device,
    stride: int = 512,
    max_length: Optional[int] = None
) -> float:
    """
    Compute perplexity of the model on a given text.

    Uses a sliding window approach to handle texts longer than the model's
    context window, with overlap for better estimation.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        text: Text to compute perplexity on
        device: Device to run inference on
        stride: Stride for sliding window (overlap = max_length - stride)
        max_length: Maximum sequence length (defaults to model's max length)

    Returns:
        Perplexity score (lower is better, 1.0 is perfect prediction)
    """
    model.eval()

    if max_length is None:
        max_length = model.config.n_positions if hasattr(model.config, 'n_positions') else 1024

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # Number of tokens to compute loss on

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()

        # Mask out tokens we've already computed loss for (the overlap region)
        target_chunk[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)

            # Handle potential NaN/Inf in loss
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                # Return a very high perplexity to indicate model breakdown
                return float('inf')

            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Average NLL and compute perplexity
    total_nll = torch.stack(nlls).sum()
    total_tokens = prev_end_loc

    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(avg_nll).item()

    # Cap extremely high perplexity values
    return min(perplexity, 1e10)


def generate_text_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: torch.device,
    n_samples: int = 3,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True
) -> List[str]:
    """
    Generate multiple text samples from a prompt.

    Args:
        model: Language model to generate from
        tokenizer: Tokenizer for the model
        prompt: Starting prompt for generation
        device: Device to run inference on
        n_samples: Number of samples to generate
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling threshold
        do_sample: Whether to sample or use greedy decoding

    Returns:
        List of generated text samples (including the prompt)
    """
    model.eval()

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    samples = []
    for _ in range(n_samples):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append(generated_text)

        except Exception as e:
            # If generation fails (e.g., due to extreme noise), return error message
            samples.append(f"{prompt}... [GENERATION FAILED: {str(e)[:50]}]")

    return samples


def compute_output_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: torch.device,
    n_next_tokens: int = 10
) -> float:
    """
    Compute average entropy of next-token predictions.

    Higher entropy indicates more uncertainty/randomness in predictions,
    which can be interpreted as increased "creativity" or degraded confidence.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        device: Device to run inference on
        n_next_tokens: Number of next-token predictions to average over

    Returns:
        Average entropy in bits (log base 2)
    """
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    entropies = []

    with torch.no_grad():
        for i in range(n_next_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token's logits

            # Handle NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return float('inf')

            # Compute probability distribution
            probs = F.softmax(logits, dim=-1)

            # Compute entropy: H = -sum(p * log2(p))
            # Add small epsilon to avoid log(0)
            log_probs = torch.log2(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            entropies.append(entropy.item())

            # Sample next token and append
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return sum(entropies) / len(entropies) if entropies else 0.0


# =============================================================================
# Diversity Metrics for Optimal Jitter Analysis
# =============================================================================

def get_ngrams(text: str, n: int) -> List[tuple]:
    """
    Extract n-grams from text.

    Args:
        text: Input text
        n: Size of n-grams

    Returns:
        List of n-gram tuples
    """
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Compute Distinct-n metric: ratio of unique n-grams to total n-grams.

    Higher values indicate more diverse/creative output.
    Used to find the "Goldilocks zone" where noise increases creativity
    without causing repetition/gibberish.

    Args:
        texts: List of generated text samples
        n: Size of n-grams (typically 1, 2, or 3)

    Returns:
        Distinct-n score (0 to 1, higher is more diverse)
    """
    all_ngrams = []
    for text in texts:
        all_ngrams.extend(get_ngrams(text, n))

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def compute_repetition_ratio(texts: List[str], n: int = 3) -> float:
    """
    Compute repetition ratio: how often n-grams are repeated.

    Higher values indicate more repetitive (degraded) output.

    Args:
        texts: List of generated text samples
        n: Size of n-grams to check for repetition

    Returns:
        Repetition ratio (0 to 1, lower is better)
    """
    all_ngrams = []
    for text in texts:
        all_ngrams.extend(get_ngrams(text, n))

    if not all_ngrams:
        return 0.0

    unique_count = len(set(all_ngrams))
    total_count = len(all_ngrams)

    # Repetition = 1 - (unique / total)
    return 1.0 - (unique_count / total_count)


def compute_vocabulary_richness(texts: List[str]) -> dict:
    """
    Compute multiple vocabulary richness metrics.

    Args:
        texts: List of generated text samples

    Returns:
        Dictionary with type-token ratio, unique words count, etc.
    """
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    if not all_words:
        return {"type_token_ratio": 0.0, "unique_words": 0, "total_words": 0}

    unique_words = set(all_words)

    return {
        "type_token_ratio": len(unique_words) / len(all_words),
        "unique_words": len(unique_words),
        "total_words": len(all_words),
        "distinct_1": compute_distinct_n(texts, 1),
        "distinct_2": compute_distinct_n(texts, 2),
        "distinct_3": compute_distinct_n(texts, 3)
    }
