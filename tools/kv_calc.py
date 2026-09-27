#!/usr/bin/env python3
"""KV cache sizing calculator.

Computes the size of the transformer key/value cache -- in bytes per token and
in total for a given (tokens, batch, dtype) -- for three attention families:

  * "gqa"    Plain multi-head / grouped-query / multi-query attention. Every
             layer caches K and V for every token that has been seen. Bytes
             per token per layer = 2 * num_kv_heads * head_dim * bytes_per_element
             (the classic formula, e.g. kipply 2022, "Transformer Inference
             Arithmetic"; Pope et al. 2022, "Efficiently Scaling Transformer
             Inference", arXiv:2211.05102).

  * "mla"    DeepSeek-style Multi-head Latent Attention. Instead of per-head
             K/V, each layer caches only a shared low-rank latent vector
             (dimension kv_lora_rank) plus a shared decoupled RoPE key
             (dimension qk_rope_head_dim). Per DeepSeek-V2 (Liu et al. 2024,
             "DeepSeek-V2", arXiv:2405.04434, Section 2.1.2-2.1.3 and Table 1):
             KV cache per token = (d_c + d_h^R) * l elements, where d_c is
             kv_lora_rank, d_h^R is qk_rope_head_dim, and l is the layer
             count. Notably this does NOT depend on the number of attention
             heads and there is no separate factor of 2 for K vs V, because
             both are reconstructed from the same cached latent.

  * "hybrid" Models that mix full-attention layers (K/V grows with every
             token, as in "gqa") with sliding-window layers whose cache is
             capped at a fixed window size once the sequence exceeds it
             (e.g. Mistral 7B's uniform sliding window, or Gemma 2/3's and
             gpt-oss's interleaved local/global layers). Bytes per layer per
             cached position use the same 2 * num_kv_heads * head_dim formula
             as "gqa"; what differs is how many positions are cached: full
             layers cache min(tokens, tokens) = tokens positions, sliding
             layers cache min(tokens, window) positions.

Built-in model table
---------------------
Architecture parameters below were read directly from each model's public
Hugging Face `config.json` (the exact URL fetched is given in each entry's
`config_url`, and reproduced in a comment next to the entry). Two families
(Meta's Llama-3/3.1 and Google's Gemma-2/3) gate the original repository
behind a license click-through, which blocks anonymous/anonymous-friendly
fetches; for those, the config was read from a public mirror re-upload
(NousResearch for Llama, unsloth for Gemma). The architecture-relevant
fields these mirrors report (head counts, head_dim, layer count, sliding
window) match what is documented for the gated originals, but the mirror
files are not byte-for-byte identical to Meta's/Google's own release
copies: unsloth's gemma-3-27b-it config is explicitly flagged
`"unsloth_fixed": true` (unsloth's own marker for a patched config), and
its gemma-2-9b-it config carries extra fields a stock release would not
(`_name_or_path`, `unsloth_version`, a duplicated `sliding_window_size`
alongside `sliding_window`). This is noted per entry below. Gemma 2's 1:1
local:global layer ratio is not itself a config.json field (transformers
hardcodes it for the gemma2 model type); it is taken from Google's own
Gemma 3 technical report (arXiv:2503.19786, Sec. 2), which states it
explicitly when contrasting Gemma 2 against Gemma 3's 5:1 ratio.

`max_context` records each model's own documented or configured maximum
context length (usually config.json's `max_position_embeddings`, or a
paper's stated trained/evaluated length when that differs -- see the
mistral-7b-v0.1 entry). It is independent of the KV-cache-bytes formula:
`kv_cache_bytes()` will compute a cache size for any `tokens` value
requested, but flags the result (`exceeds_max_context`) when `tokens`
exceeds this figure, since that is an extrapolation past what the model is
documented to support, not a realizable deployment configuration on its
own. Four of the eleven built-in models do not reach 131,072 (128K)
tokens this way: Llama 3 8B and Gemma 2 9B (max_position_embeddings=8192),
Qwen3-8B (max_position_embeddings=40960, rope_scaling=null -- no YaRN or
other extension is configured), and Mistral 7B v0.1 (context_len=8192 per
its own paper's Table 1, arXiv:2310.06825; its config.json's
max_position_embeddings=32768 is a separate, looser positional-embedding
ceiling the model was not documented as trained or evaluated at).

`native_dtype` records the KV-cache/activation dtype this tool assumes for
each model, which is not necessarily the released checkpoint's own weight
storage format. DeepSeek-V3 and Kimi-K2-Instruct's public config.json
files carry a `quantization_config` block with `quant_method: "fp8"`
(block-scaled, `weight_block_size: [128, 128]`) for their *weights*, yet
both are modeled here as bf16 KV cache. gpt-oss-20b's own config.json
clarifies the general pattern: its `quantization_config` (`mxfp4`) lists
`modules_to_not_convert`, which explicitly excludes `model.layers.*.self_attn`
from weight quantization -- i.e. OpenAI's own config documents that the
attention path is not run in the same low-precision format as the rest of
the weights. DeepSeek-V3/Kimi-K2's config carries no equivalent exclusion
list, so whether their FP8 weight quantization extends to the attention/KV
path is not established from the config alone; `native_dtype="bf16"` for
these two is this tool's modeling assumption, not a verified fact about
either checkpoint's runtime KV dtype.

This tool is dependency-free (Python 3 standard library only).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Units and dtypes
# ---------------------------------------------------------------------------

KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3

# Bytes per cached element for each KV cache storage dtype. fp8/int8 KV cache
# quantization is a real, shipped serving feature (e.g. vLLM's "Quantized KV
# Cache", https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)
# -- it is independent of the dtype the model's weights are stored/trained in.
BYTES_PER_ELEMENT: Dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
}


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass
class Model:
    key: str
    name: str
    org: str
    arch: str  # "gqa" | "mla" | "hybrid"
    layers: int
    config_url: str
    native_dtype: str = "bf16"
    note: str = ""
    # This model's own documented/configured maximum context length (see
    # the module docstring's "max_context" paragraph). None means unknown;
    # every built-in model below sets it.
    max_context: Optional[int] = None

    # "gqa" and "hybrid" fields (bytes/layer/position = 2 * num_kv_heads * head_dim * s)
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # "hybrid"-only fields
    window: Optional[int] = None
    # Every `full_attn_period`-th layer (1-indexed) is full attention; the
    # rest are sliding-window. None/0 means there are no full-attention
    # layers at all (the whole model is uniformly sliding-window, as in
    # Mistral 7B v0.1). Ignored if `layer_pattern` is given.
    full_attn_period: Optional[int] = None
    # Optional explicit per-layer override, e.g. copied verbatim from a
    # config.json "layer_types" field. Each entry is "full" or "sliding".
    layer_pattern: Optional[List[str]] = None

    # "mla"-only fields (bytes/layer/position = (kv_lora_rank + qk_rope_head_dim) * s)
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    def layer_types(self) -> List[str]:
        """Return a length-`layers` list of "full"/"sliding", for hybrid models."""
        if self.layer_pattern is not None:
            if len(self.layer_pattern) != self.layers:
                raise ValueError(
                    f"{self.key}: layer_pattern has {len(self.layer_pattern)} "
                    f"entries, expected {self.layers}"
                )
            return list(self.layer_pattern)
        if not self.full_attn_period:
            return ["sliding"] * self.layers
        return [
            "full" if (i + 1) % self.full_attn_period == 0 else "sliding"
            for i in range(self.layers)
        ]


# gpt-oss-20b's config.json spells out its layer types explicitly (alternating,
# starting with sliding_attention); reproduced verbatim rather than derived
# from a period, since we have the exact list.
_GPT_OSS_20B_LAYER_TYPES = [
    "sliding" if t == "sliding_attention" else "full"
    for t in (
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
    )
]

MODELS: Dict[str, Model] = {
    # --- Plain GQA ----------------------------------------------------------
    "llama3-8b": Model(
        key="llama3-8b",
        name="Llama 3 8B",
        org="Meta",
        arch="gqa",
        layers=32,
        num_kv_heads=8,
        head_dim=128,
        native_dtype="bf16",
        # Source: https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/config.json
        # (gated; read via public mirror -- see module docstring)
        # Mirror fetched: https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json
        config_url="https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json",
        max_context=8192,
        note=(
            "num_attention_heads=32, num_key_value_heads=8, hidden_size=4096 -> "
            "head_dim=128. max_position_embeddings=8192 (config field, "
            "rope_scaling=null -- no extension configured); the 131,072-token "
            "(128K) rows in this tool's comparison tables are an extrapolation "
            "past this documented length."
        ),
    ),
    "llama31-70b": Model(
        key="llama31-70b",
        name="Llama 3.1 70B",
        org="Meta",
        arch="gqa",
        layers=80,
        num_kv_heads=8,
        head_dim=128,
        native_dtype="bf16",
        # Source: https://huggingface.co/meta-llama/Llama-3.1-70B/raw/main/config.json
        # (gated; read via public mirror -- see module docstring)
        config_url="https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/raw/main/config.json",
        max_context=131072,
        note=(
            "num_attention_heads=64, num_key_value_heads=8, hidden_size=8192 -> "
            "head_dim=128. max_position_embeddings=131072 via a llama3-type RoPE "
            "scaling (factor=8 from an original 8192), so 128K-token sizing here "
            "is within the model's documented context."
        ),
    ),
    "qwen2.5-7b": Model(
        key="qwen2.5-7b",
        name="Qwen2.5 7B",
        org="Alibaba",
        arch="gqa",
        layers=28,
        num_kv_heads=4,
        head_dim=128,
        native_dtype="bf16",
        config_url="https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json",
        max_context=131072,
        note=(
            "num_attention_heads=28, num_key_value_heads=4, hidden_size=3584 -> "
            "head_dim=128. config.json sets use_sliding_window=false, so despite "
            "a sliding_window field being present (131072, equal to "
            "max_position_embeddings), every layer is full attention."
        ),
    ),
    "qwen3-8b": Model(
        key="qwen3-8b",
        name="Qwen3 8B",
        org="Alibaba",
        arch="gqa",
        layers=36,
        num_kv_heads=8,
        head_dim=128,
        native_dtype="bf16",
        config_url="https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json",
        max_context=40960,
        note=(
            "num_attention_heads=32, head_dim=128 (explicit config field); no "
            "sliding window. max_position_embeddings=40960 with rope_scaling=null "
            "-- YaRN or another extension method is not enabled in this config, "
            "so the 131,072-token (128K) rows in this tool's comparison tables "
            "are an extrapolation past this documented length."
        ),
    ),
    # --- Multi-head Latent Attention (MLA) -----------------------------------
    "deepseek-v2": Model(
        key="deepseek-v2",
        name="DeepSeek-V2",
        org="DeepSeek-AI",
        arch="mla",
        layers=60,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        native_dtype="bf16",
        config_url="https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json",
        max_context=163840,
        note=(
            "236B total / 21B activated (MoE). MLA per-token cache = "
            "(kv_lora_rank + qk_rope_head_dim) * layers elements, independent of "
            "num_attention_heads=128 -- see deepseek-2024-mla, Table 1. "
            "max_position_embeddings=163840 via a YaRN RoPE scaling (factor=40 "
            "from an original 4096) baked into this config, not a separate "
            "extension a user must enable."
        ),
    ),
    "deepseek-v3": Model(
        key="deepseek-v3",
        name="DeepSeek-V3",
        org="DeepSeek-AI",
        arch="mla",
        layers=61,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        native_dtype="bf16",
        config_url="https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json",
        max_context=163840,
        note=(
            "671B total / 37B activated (MoE). Same MLA cache shape as "
            "DeepSeek-V2; max_position_embeddings=163840 (same YaRN scaling "
            "pattern as V2). The public checkpoint's config.json carries a "
            "quantization_config (quant_method=fp8, block-scaled) for its "
            "*weights*; native_dtype=bf16 here is this tool's assumption about "
            "the KV-cache/activation dtype, not a claim about the released "
            "checkpoint's weight storage format -- see module docstring."
        ),
    ),
    "kimi-k2": Model(
        key="kimi-k2",
        name="Kimi K2 Instruct",
        org="Moonshot AI",
        arch="mla",
        layers=61,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        native_dtype="bf16",
        config_url="https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json",
        max_context=131072,
        note=(
            "1T total / 32B activated (MoE); architecture is DeepSeek-V3's "
            "DeepseekV3ForCausalLM with the same MLA cache shape (kv_lora_rank=512, "
            "qk_rope_head_dim=64). max_position_embeddings=131072 via YaRN "
            "(factor=32 from an original 4096). Like DeepSeek-V3, the public "
            "config.json carries an fp8 quantization_config for its weights; "
            "native_dtype=bf16 here is this tool's KV-cache/activation-dtype "
            "assumption, not a claim about the checkpoint's weight format -- see "
            "module docstring."
        ),
    ),
    # --- Sliding-window / hybrid ---------------------------------------------
    "mistral-7b-v0.1": Model(
        key="mistral-7b-v0.1",
        name="Mistral 7B v0.1",
        org="Mistral AI",
        arch="hybrid",
        layers=32,
        num_kv_heads=8,
        head_dim=128,
        window=4096,
        full_attn_period=None,  # every layer is sliding-window; no global layers
        native_dtype="bf16",
        config_url="https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json",
        max_context=8192,
        note=(
            "All 32 layers use a uniform 4096-token sliding window (no global "
            "layers) -- unlike Gemma/gpt-oss's local+global hybrids, this is a "
            "pure sliding-window model, so its KV cache is bounded and stops "
            "growing once tokens > 4096. Some serving stacks disable the window "
            "(run full attention) beyond the training length for quality reasons; "
            "this tool models the architecture as configured, not any given "
            "serving engine's override. max_context=8192 follows the model's own "
            "paper (arXiv:2310.06825, Table 1: context_len=8192), not "
            "config.json's max_position_embeddings=32768, which is a separate, "
            "looser positional-embedding ceiling the model was not documented as "
            "trained or evaluated at. The paper separately computes a "
            "~131K-token theoretical multi-layer receptive field from "
            "window x layers (4096 x 32); that describes how far information "
            "can propagate through 32 stacked sliding-window layers, not a "
            "supported input length -- it is not evidence for max_context=131072."
        ),
    ),
    "gemma2-9b": Model(
        key="gemma2-9b",
        name="Gemma 2 9B",
        org="Google",
        arch="hybrid",
        layers=42,
        num_kv_heads=8,
        head_dim=256,
        window=4096,
        full_attn_period=2,  # 1:1 local:global, per the Gemma 3 report (below)
        native_dtype="bf16",
        # Source: https://huggingface.co/google/gemma-2-9b/raw/main/config.json
        # (gated; read via public mirror -- see module docstring; this mirror's
        # config.json also carries _name_or_path, unsloth_version and a
        # duplicated sliding_window_size not present in a stock release)
        config_url="https://huggingface.co/unsloth/gemma-2-9b-it/raw/main/config.json",
        max_context=8192,
        note=(
            "num_attention_heads=16, num_key_value_heads=8, head_dim=256 (config "
            "field, larger than hidden_size/num_attention_heads=224). The 1:1 "
            "local:global layer ratio is not itself a config.json field for "
            "gemma2 (hardcoded in transformers); confirmed by Google's Gemma 3 "
            "technical report (arXiv:2503.19786, Sec. 2), which states '1:1 is "
            "used in Gemma 2 models' when introducing Gemma 3's 5:1 ratio. "
            "max_position_embeddings=8192 (config field, no RoPE extension), so "
            "the 131,072-token (128K) rows in this tool's comparison tables are "
            "an extrapolation past this documented length."
        ),
    ),
    "gemma3-27b": Model(
        key="gemma3-27b",
        name="Gemma 3 27B",
        org="Google",
        arch="hybrid",
        layers=62,
        num_kv_heads=16,
        head_dim=128,
        window=1024,
        full_attn_period=6,  # 5 local : 1 global, per config's sliding_window_pattern
        native_dtype="bf16",
        # Source: https://huggingface.co/google/gemma-3-27b-it/raw/main/config.json
        # (gated; read via public mirror -- see module docstring; this specific
        # mirror file is explicitly flagged "unsloth_fixed": true, i.e. unsloth's
        # own marker that it patched the config relative to Google's release;
        # fields live under the top-level "text_config" object for this
        # multimodal model)
        config_url="https://huggingface.co/unsloth/gemma-3-27b-it/raw/main/config.json",
        max_context=131072,
        note=(
            "text_config: num_attention_heads=32, num_key_value_heads=16, "
            "head_dim=128, sliding_window=1024, sliding_window_pattern=6 (5 local "
            "layers per global layer), max_position_embeddings=131072. Per "
            "Google's Gemma 3 technical report (arXiv:2503.19786, Sec. 2), this "
            "ratio and the smaller 1024-token window (down from Gemma 2's 4096) "
            "are explicitly designed to curb long-context KV cache growth. The "
            "paper's own ablation (Sec. 5.2, Fig. 5), on a 2B text-only model at "
            "a 32K-token context, shows a 'global only' baseline at ~60% "
            "KV-cache memory overhead (relative to model weights) falling to "
            "<15% with a 1:3 local:global ratio and sw=1024 -- the closest "
            "ablated setting to, but not identical to, the 5:1 ratio actually "
            "shipped (including this 27B model); the paper does not report a "
            "single overhead percentage for the shipped 5:1 configuration at "
            "32K (Fig. 6 shows only a qualitative overhead-vs-context-length "
            "curve for it)."
        ),
    ),
    "gpt-oss-20b": Model(
        key="gpt-oss-20b",
        name="gpt-oss-20b",
        org="OpenAI",
        arch="hybrid",
        layers=24,
        num_kv_heads=8,
        head_dim=64,
        window=128,
        layer_pattern=_GPT_OSS_20B_LAYER_TYPES,
        native_dtype="bf16",
        config_url="https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json",
        max_context=131072,
        note=(
            "config.json's layer_types alternates sliding_attention/full_attention "
            "1:1 across all 24 layers (reproduced verbatim here), with a notably "
            "short 128-token sliding window -- the smallest per-layer window in "
            "this table, though not the most aggressive overall cache reduction: "
            "Mistral 7B v0.1's all-sliding design (no full-attention layers at "
            "all) yields both a smaller total cache at any given length and zero "
            "marginal growth past its window, at the cost of retaining no "
            "long-range token-level context in any layer, versus gpt-oss-20b's "
            "12 full-attention layers which keep the cache growing linearly. "
            "num_attention_heads=64, num_key_value_heads=8, head_dim=64 "
            "(hidden_size=2880, so head_dim is not hidden_size/num_attention_heads). "
            "max_position_embeddings=131072. The public checkpoint's config.json "
            "carries a quantization_config (quant_method=mxfp4) whose "
            "modules_to_not_convert list explicitly excludes "
            "model.layers.*.self_attn from weight quantization -- i.e. OpenAI's "
            "own config documents that attention is not run in mxfp4, consistent "
            "with native_dtype=bf16 here (see module docstring)."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Core sizing logic
# ---------------------------------------------------------------------------


def _dtype_bytes(dtype: str) -> float:
    try:
        return BYTES_PER_ELEMENT[dtype]
    except KeyError:
        raise ValueError(
            f"unknown dtype {dtype!r}; choose from {sorted(BYTES_PER_ELEMENT)}"
        ) from None


def kv_cache_bytes(model: Model, tokens: int, batch: int = 1, dtype: Optional[str] = None) -> dict:
    """Compute KV cache sizing for `model` at a given cached-token count.

    Returns a dict with:
      dtype, bytes_per_element        -- the dtype actually used and its width
      per_token_avg_bytes             -- total_bytes_per_sequence / tokens
                                          (exactly constant for "gqa"/"mla"; an
                                          average, not a marginal rate, for
                                          "hybrid" once tokens > window)
      marginal_bytes_per_token        -- bytes added per *additional* token once
                                          any sliding windows are already full
                                          (equals per_token_avg_bytes for
                                          "gqa"/"mla"; for "hybrid" this is the
                                          steady-state/asymptotic rate, driven
                                          only by the full-attention layers --
                                          zero if a model has none, as with
                                          Mistral 7B v0.1)
      total_bytes_per_sequence        -- bytes for one sequence of this length
      total_bytes                     -- total_bytes_per_sequence * batch
      full_layers, sliding_layers     -- layer counts (hybrid only; else None)
      max_context                     -- model.max_context, copied through for
                                          convenience (may be None)
      exceeds_max_context             -- True if `tokens` is beyond
                                          model.max_context, i.e. this result
                                          is an extrapolation past what the
                                          model is documented/configured to
                                          support (False if max_context is
                                          None, i.e. unknown)
    """
    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    if batch < 1:
        raise ValueError("batch must be >= 1")

    dtype = dtype or model.native_dtype
    s = _dtype_bytes(dtype)
    full_layers = sliding_layers = None

    if model.arch == "mla":
        if model.kv_lora_rank is None or model.qk_rope_head_dim is None:
            raise ValueError(f"{model.key}: mla model missing kv_lora_rank/qk_rope_head_dim")
        per_layer = (model.kv_lora_rank + model.qk_rope_head_dim) * s
        per_token = model.layers * per_layer
        total_per_seq = per_token * tokens
        marginal = per_token

    elif model.arch == "gqa":
        if model.num_kv_heads is None or model.head_dim is None:
            raise ValueError(f"{model.key}: gqa model missing num_kv_heads/head_dim")
        per_layer = 2 * model.num_kv_heads * model.head_dim * s
        per_token = model.layers * per_layer
        total_per_seq = per_token * tokens
        marginal = per_token

    elif model.arch == "hybrid":
        if model.num_kv_heads is None or model.head_dim is None or not model.window:
            raise ValueError(f"{model.key}: hybrid model missing num_kv_heads/head_dim/window")
        per_layer = 2 * model.num_kv_heads * model.head_dim * s
        types = model.layer_types()
        full_layers = types.count("full")
        sliding_layers = types.count("sliding")
        capped_tokens = min(tokens, model.window)
        total_per_seq = per_layer * (full_layers * tokens + sliding_layers * capped_tokens)
        marginal = per_layer * full_layers
        per_token = (total_per_seq / tokens) if tokens else 0.0

    else:
        raise ValueError(f"{model.key}: unknown arch {model.arch!r}")

    return {
        "model": model.key,
        "dtype": dtype,
        "bytes_per_element": s,
        "tokens": tokens,
        "batch": batch,
        "per_token_avg_bytes": per_token,
        "marginal_bytes_per_token": marginal,
        "total_bytes_per_sequence": total_per_seq,
        "total_bytes": total_per_seq * batch,
        "full_layers": full_layers,
        "sliding_layers": sliding_layers,
        "max_context": model.max_context,
        "exceeds_max_context": model.max_context is not None and tokens > model.max_context,
    }


# ---------------------------------------------------------------------------
# Formatting / CLI
# ---------------------------------------------------------------------------


def fmt_bytes(n: float) -> str:
    if n >= GIB:
        return f"{n / GIB:.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:.2f} MiB"
    if n >= KIB:
        return f"{n / KIB:.2f} KiB"
    return f"{n:.0f} B"


def parse_tokens(text: str) -> int:
    """Parse a token count, accepting binary-prefix shorthand (128k, 1m, 2g)."""
    t = text.strip().lower()
    mult = 1
    if t.endswith("k"):
        mult, t = KIB, t[:-1]
    elif t.endswith("m"):
        mult, t = MIB, t[:-1]
    elif t.endswith("g"):
        mult, t = GIB, t[:-1]
    return int(round(float(t) * mult))


def arch_summary(m: Model) -> str:
    if m.arch == "mla":
        return f"MLA (kv_lora_rank={m.kv_lora_rank}, qk_rope_head_dim={m.qk_rope_head_dim})"
    if m.arch == "gqa":
        return f"GQA (num_kv_heads={m.num_kv_heads}, head_dim={m.head_dim})"
    if m.arch == "hybrid":
        types = m.layer_types()
        full, sliding = types.count("full"), types.count("sliding")
        return (
            f"Hybrid (num_kv_heads={m.num_kv_heads}, head_dim={m.head_dim}, "
            f"window={m.window}, full:sliding layers={full}:{sliding})"
        )
    return m.arch


def print_model_list(markdown: bool) -> None:
    if not markdown:
        for m in MODELS.values():
            print(f"{m.key:16s} {m.name} ({m.org}) -- {arch_summary(m)}")
            ctx = f"{m.max_context:,}" if m.max_context is not None else "unknown"
            print(
                f"{'':16s} layers={m.layers} native_dtype={m.native_dtype} "
                f"max_context={ctx} source={m.config_url}"
            )
            if m.note:
                print(f"{'':16s} note: {m.note}")
        return
    print("| Key | Model | Org | Layers | Architecture | Native dtype | Max context |")
    print("|---|---|---|---|---|---|---|")
    for m in MODELS.values():
        ctx = f"{m.max_context:,}" if m.max_context is not None else "unknown"
        print(f"| `{m.key}` | {m.name} | {m.org} | {m.layers} | {arch_summary(m)} | {m.native_dtype} | {ctx} |")


def build_rows(tokens: int, batch: int, dtype: Optional[str], only: Optional[str]):
    keys = [only] if only else list(MODELS.keys())
    rows = []
    for key in keys:
        m = MODELS[key]
        rows.append((m, kv_cache_bytes(m, tokens, batch, dtype=dtype)))
    return rows


def print_markdown_table(rows, tokens: int, batch: int) -> None:
    print(f"KV cache sizing at {tokens:,} tokens, batch {batch}\n")
    print("| Model | Org | Architecture | Layers | dtype | Bytes/token (avg) | Total per sequence | Total (batch) |")
    print("|---|---|---|---|---|---|---|---|")
    flagged = []
    for m, r in rows:
        marker = " †" if r["exceeds_max_context"] else ""
        print(
            f"| {m.name}{marker} | {m.org} | {m.arch.upper()} | {m.layers} | {r['dtype']} "
            f"| {fmt_bytes(r['per_token_avg_bytes'])} "
            f"| {fmt_bytes(r['total_bytes_per_sequence'])} "
            f"| {fmt_bytes(r['total_bytes'])} |"
        )
        if r["exceeds_max_context"]:
            flagged.append(f"{m.name} (max_context={m.max_context:,})")
    if flagged:
        print(
            f"\n† {tokens:,} cached tokens exceeds this model's documented or "
            f"configured maximum context length -- these rows extrapolate the "
            f"sizing formula past what the model is known to support, not a "
            f"realizable deployment configuration on their own: {'; '.join(flagged)}."
        )


def print_plain_table(rows) -> None:
    for m, r in rows:
        print(f"{m.key} ({m.name}, {r['dtype']}):")
        print(f"  bytes/token (avg):        {r['per_token_avg_bytes']:,.1f}  ({fmt_bytes(r['per_token_avg_bytes'])})")
        if r["full_layers"] is not None:
            print(
                f"  marginal bytes/token:     {r['marginal_bytes_per_token']:,.1f} "
                f"({fmt_bytes(r['marginal_bytes_per_token'])}) "
                f"[full_layers={r['full_layers']}, sliding_layers={r['sliding_layers']}]"
            )
        print(f"  total per sequence:       {fmt_bytes(r['total_bytes_per_sequence'])}")
        print(f"  total (batch={r['batch']}):        {fmt_bytes(r['total_bytes'])}")
        if r["exceeds_max_context"]:
            print(
                f"  ** {r['tokens']:,} tokens exceeds max_context="
                f"{m.max_context:,} for this model -- extrapolated past its "
                f"documented/configured context length **"
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute transformer KV cache size (bytes/token and totals) for open-weight LLMs.",
    )
    parser.add_argument("--model", choices=sorted(MODELS), help="Model key. Omit to compute all built-in models.")
    parser.add_argument(
        "--tokens", default="4096",
        help="Cached context length; accepts binary-prefix shorthand (4096, 128k, 1m). Default: 4096.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size (independent sequences). Default: 1.")
    parser.add_argument(
        "--dtype", choices=sorted(BYTES_PER_ELEMENT), default=None,
        help=(
            "KV cache element dtype. Default: each model's native_dtype, i.e. "
            "the assumed KV-cache/activation dtype, not necessarily the "
            "released checkpoint's own weight storage format (see --list or "
            "the module docstring, e.g. for DeepSeek-V3/Kimi-K2)."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List built-in models and their architecture parameters.")
    parser.add_argument("--markdown", action="store_true", help="Print a GitHub-flavored markdown table.")
    args = parser.parse_args(argv)

    if args.list:
        print_model_list(markdown=args.markdown)
        return 0

    try:
        tokens = parse_tokens(str(args.tokens))
    except ValueError:
        parser.error(f"invalid --tokens value: {args.tokens!r}")
        return 2

    try:
        rows = build_rows(tokens, args.batch, args.dtype, args.model)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if args.markdown:
        print_markdown_table(rows, tokens, args.batch)
    else:
        print_plain_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
