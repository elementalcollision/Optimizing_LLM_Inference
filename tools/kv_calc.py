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
fetches; for those, the config was read from a byte-identical public mirror
re-upload (NousResearch for Llama, unsloth for Gemma), which is noted in the
entry. Gemma 2's 1:1 local:global layer ratio is not itself a config.json
field (transformers hardcodes it for the gemma2 model type); it is taken
from Google's own Gemma 3 technical report (arXiv:2503.19786), which states
it explicitly when contrasting Gemma 2 against Gemma 3's 5:1 ratio.

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
        # (gated; read via public mirror, byte-identical fields)
        # Mirror fetched: https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json
        config_url="https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json",
        note="num_attention_heads=32, num_key_value_heads=8, hidden_size=4096 -> head_dim=128.",
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
        # (gated; read via public mirror, byte-identical fields)
        config_url="https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/raw/main/config.json",
        note="num_attention_heads=64, num_key_value_heads=8, hidden_size=8192 -> head_dim=128.",
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
        note=(
            "num_attention_heads=28, num_key_value_heads=4, hidden_size=3584 -> "
            "head_dim=128. config.json sets use_sliding_window=false, so despite "
            "a sliding_window field being present, every layer is full attention."
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
        note="num_attention_heads=32, head_dim=128 (explicit config field); no sliding window.",
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
        note=(
            "236B total / 21B activated (MoE). MLA per-token cache = "
            "(kv_lora_rank + qk_rope_head_dim) * layers elements, independent of "
            "num_attention_heads=128 -- see deepseek-2024-mla, Table 1."
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
        note="671B total / 37B activated (MoE). Same MLA cache shape as DeepSeek-V2.",
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
        note=(
            "1T total / 32B activated (MoE); architecture is DeepSeek-V3's "
            "DeepseekV3ForCausalLM with the same MLA cache shape (kv_lora_rank=512, "
            "qk_rope_head_dim=64)."
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
        note=(
            "All 32 layers use a uniform 4096-token sliding window (no global "
            "layers) -- unlike Gemma/gpt-oss's local+global hybrids, this is a "
            "pure sliding-window model, so its KV cache is bounded and stops "
            "growing once tokens > 4096. Some serving stacks disable the window "
            "(run full attention) beyond the training length for quality reasons; "
            "this tool models the architecture as configured, not any given "
            "serving engine's override."
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
        # (gated; read via public mirror, byte-identical fields)
        config_url="https://huggingface.co/unsloth/gemma-2-9b-it/raw/main/config.json",
        note=(
            "num_attention_heads=16, num_key_value_heads=8, head_dim=256 (config "
            "field, larger than hidden_size/num_attention_heads=224). The 1:1 "
            "local:global layer ratio is not itself a config.json field for "
            "gemma2 (hardcoded in transformers); confirmed by Google's Gemma 3 "
            "technical report (arXiv:2503.19786), which states '1:1 is used in "
            "Gemma 2 models' when introducing Gemma 3's 5:1 ratio."
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
        # (gated; read via public mirror, byte-identical fields; fields live
        # under the top-level "text_config" object for this multimodal model)
        config_url="https://huggingface.co/unsloth/gemma-3-27b-it/raw/main/config.json",
        note=(
            "text_config: num_attention_heads=32, num_key_value_heads=16, "
            "head_dim=128, sliding_window=1024, sliding_window_pattern=6 (5 local "
            "layers per global layer). Per Google's Gemma 3 technical report "
            "(arXiv:2503.19786, Sec. 5), this ratio and the smaller 1024-token "
            "window (down from Gemma 2's 4096) are explicitly designed to curb "
            "long-context KV cache growth, reducing overhead from ~60% to <15% "
            "of model-weight memory at a 32K-token context in their measurements."
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
        note=(
            "config.json's layer_types alternates sliding_attention/full_attention "
            "1:1 across all 24 layers (reproduced verbatim here), with a notably "
            "short 128-token sliding window -- the most aggressive window/ratio "
            "combination in this table. num_attention_heads=64, "
            "num_key_value_heads=8, head_dim=64 (hidden_size=2880, so head_dim is "
            "not hidden_size/num_attention_heads)."
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
            print(f"{'':16s} layers={m.layers} native_dtype={m.native_dtype} source={m.config_url}")
            if m.note:
                print(f"{'':16s} note: {m.note}")
        return
    print("| Key | Model | Org | Layers | Architecture | Native dtype |")
    print("|---|---|---|---|---|---|")
    for m in MODELS.values():
        print(f"| `{m.key}` | {m.name} | {m.org} | {m.layers} | {arch_summary(m)} | {m.native_dtype} |")


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
    for m, r in rows:
        print(
            f"| {m.name} | {m.org} | {m.arch.upper()} | {m.layers} | {r['dtype']} "
            f"| {fmt_bytes(r['per_token_avg_bytes'])} "
            f"| {fmt_bytes(r['total_bytes_per_sequence'])} "
            f"| {fmt_bytes(r['total_bytes'])} |"
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
        help="KV cache element dtype. Default: each model's native dtype.",
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
