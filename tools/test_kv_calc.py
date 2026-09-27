#!/usr/bin/env python3
"""Unit tests for kv_calc.py. Run with: python3 tools/test_kv_calc.py"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kv_calc  # noqa: E402


class TestGQAFormula(unittest.TestCase):
    def test_llama3_8b_bytes_per_token(self):
        """32 layers, 8 KV heads, head_dim 128, BF16 -> 131072 bytes/token."""
        m = kv_calc.MODELS["llama3-8b"]
        self.assertEqual(m.layers, 32)
        self.assertEqual(m.num_kv_heads, 8)
        self.assertEqual(m.head_dim, 128)
        r = kv_calc.kv_cache_bytes(m, tokens=1, batch=1, dtype="bf16")
        self.assertEqual(r["per_token_avg_bytes"], 131072)
        self.assertEqual(r["marginal_bytes_per_token"], 131072)

    def test_llama3_8b_one_mebitoken_is_128_gib(self):
        """1 Mi (2**20 = 1,048,576) tokens of Llama-3-8B == exactly 128 GiB."""
        m = kv_calc.MODELS["llama3-8b"]
        tokens = 1024 * 1024  # 1,048,576
        r = kv_calc.kv_cache_bytes(m, tokens=tokens, batch=1, dtype="bf16")
        self.assertEqual(r["total_bytes"], 128 * kv_calc.GIB)
        self.assertAlmostEqual(r["total_bytes"] / kv_calc.GIB, 128.0, places=6)

    def test_batch_multiplies_total_not_per_token(self):
        m = kv_calc.MODELS["llama3-8b"]
        r1 = kv_calc.kv_cache_bytes(m, tokens=4096, batch=1, dtype="bf16")
        r8 = kv_calc.kv_cache_bytes(m, tokens=4096, batch=8, dtype="bf16")
        self.assertEqual(r1["per_token_avg_bytes"], r8["per_token_avg_bytes"])
        self.assertEqual(r8["total_bytes"], r1["total_bytes"] * 8)

    def test_dtype_override_scales_linearly(self):
        m = kv_calc.MODELS["llama3-8b"]
        bf16 = kv_calc.kv_cache_bytes(m, tokens=4096, dtype="bf16")
        fp8 = kv_calc.kv_cache_bytes(m, tokens=4096, dtype="fp8")
        fp32 = kv_calc.kv_cache_bytes(m, tokens=4096, dtype="fp32")
        self.assertEqual(fp8["total_bytes"], bf16["total_bytes"] / 2)
        self.assertEqual(fp32["total_bytes"], bf16["total_bytes"] * 2)

    def test_qwen25_7b_gqa_shape(self):
        m = kv_calc.MODELS["qwen2.5-7b"]
        r = kv_calc.kv_cache_bytes(m, tokens=1, dtype="bf16")
        expected = m.layers * 2 * m.num_kv_heads * m.head_dim * 2
        self.assertEqual(r["per_token_avg_bytes"], expected)


class TestMLAFormula(unittest.TestCase):
    def test_deepseek_v2_per_token_formula(self):
        """MLA bytes/token = layers * (kv_lora_rank + qk_rope_head_dim) * bytes_per_element,
        independent of num_attention_heads (DeepSeek-V2, arXiv:2405.04434, Table 1)."""
        m = kv_calc.MODELS["deepseek-v2"]
        r = kv_calc.kv_cache_bytes(m, tokens=1, dtype="bf16")
        expected = m.layers * (m.kv_lora_rank + m.qk_rope_head_dim) * 2
        self.assertEqual(r["per_token_avg_bytes"], expected)

    def test_mla_much_smaller_than_equivalent_gqa(self):
        """Sanity check against the paper's own claim: MLA's cache is far below
        what plain per-head GQA/MHA would cost for a similarly sized model."""
        deepseek_v2 = kv_calc.MODELS["deepseek-v2"]
        mla = kv_calc.kv_cache_bytes(deepseek_v2, tokens=1, dtype="bf16")
        # A same-layer-count GQA model with the full 128 attention heads and a
        # typical head_dim=128 (DeepSeek-V2's own num_attention_heads=128).
        hypothetical_gqa_bytes = deepseek_v2.layers * 2 * 128 * 128 * 2
        self.assertLess(mla["per_token_avg_bytes"], hypothetical_gqa_bytes / 20)

    def test_deepseek_v3_and_kimi_k2_share_mla_shape(self):
        v3 = kv_calc.MODELS["deepseek-v3"]
        k2 = kv_calc.MODELS["kimi-k2"]
        self.assertEqual(v3.kv_lora_rank, k2.kv_lora_rank)
        self.assertEqual(v3.qk_rope_head_dim, k2.qk_rope_head_dim)


class TestHybridSlidingWindow(unittest.TestCase):
    def test_all_sliding_caps_at_window(self):
        """Mistral 7B v0.1: uniform sliding window, no global layers. Growth
        must stop once tokens exceed the window."""
        m = kv_calc.MODELS["mistral-7b-v0.1"]
        below = kv_calc.kv_cache_bytes(m, tokens=2048, dtype="bf16")
        at_window = kv_calc.kv_cache_bytes(m, tokens=4096, dtype="bf16")
        beyond = kv_calc.kv_cache_bytes(m, tokens=100_000, dtype="bf16")
        self.assertEqual(m.full_attn_period, None)
        self.assertEqual(at_window["total_bytes"], beyond["total_bytes"])
        self.assertGreater(at_window["total_bytes"], below["total_bytes"])
        self.assertEqual(beyond["marginal_bytes_per_token"], 0)

    def test_gemma2_one_to_one_ratio(self):
        m = kv_calc.MODELS["gemma2-9b"]
        types = m.layer_types()
        self.assertEqual(len(types), m.layers)
        self.assertEqual(types.count("full"), m.layers // 2)
        self.assertEqual(types.count("sliding"), m.layers // 2)

    def test_gemma3_five_to_one_ratio(self):
        m = kv_calc.MODELS["gemma3-27b"]
        types = m.layer_types()
        full = types.count("full")
        sliding = types.count("sliding")
        self.assertEqual(full + sliding, m.layers)
        # 5 local layers for every global layer (approximately, since 62 is
        # not a multiple of 6): sliding count is roughly 5x full count.
        self.assertAlmostEqual(sliding / full, 5, delta=1.0)

    def test_gpt_oss_explicit_layer_pattern_alternates(self):
        m = kv_calc.MODELS["gpt-oss-20b"]
        types = m.layer_types()
        self.assertEqual(types[0], "sliding")
        self.assertEqual(types[1], "full")
        self.assertEqual(types.count("full"), 12)
        self.assertEqual(types.count("sliding"), 12)

    def test_hybrid_total_below_context_window_equals_gqa_shape(self):
        """When tokens <= window, a hybrid model's total must equal the plain
        "every layer is full attention" GQA total (no layer is capped yet)."""
        m = kv_calc.MODELS["gemma2-9b"]
        tokens = 1000  # well under the 4096-token window
        hybrid = kv_calc.kv_cache_bytes(m, tokens=tokens, dtype="bf16")
        per_layer = 2 * m.num_kv_heads * m.head_dim * 2
        uncapped_equivalent = m.layers * per_layer * tokens
        self.assertEqual(hybrid["total_bytes"], uncapped_equivalent)


class TestCLIParsing(unittest.TestCase):
    def test_parse_tokens_shorthand(self):
        self.assertEqual(kv_calc.parse_tokens("4096"), 4096)
        self.assertEqual(kv_calc.parse_tokens("128k"), 131072)
        self.assertEqual(kv_calc.parse_tokens("1m"), 1048576)
        self.assertEqual(kv_calc.parse_tokens("1M"), 1048576)

    def test_fmt_bytes_units(self):
        self.assertEqual(kv_calc.fmt_bytes(512), "512 B")
        self.assertIn("KiB", kv_calc.fmt_bytes(2048))
        self.assertIn("MiB", kv_calc.fmt_bytes(5 * kv_calc.MIB))
        self.assertIn("GiB", kv_calc.fmt_bytes(5 * kv_calc.GIB))

    def test_main_markdown_all_models_smoke(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kv_calc.main(["--tokens", "128k", "--batch", "1", "--dtype", "bf16", "--markdown"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("| Model | Org | Architecture", out)
        for m in kv_calc.MODELS.values():
            self.assertIn(m.name, out)

    def test_main_single_model(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kv_calc.main(["--model", "llama3-8b", "--tokens", "1", "--dtype", "bf16"])
        self.assertEqual(rc, 0)
        self.assertIn("131,072.0", buf.getvalue())

    def test_main_list(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kv_calc.main(["--list"])
        self.assertEqual(rc, 0)
        self.assertIn("llama3-8b", buf.getvalue())
        self.assertIn("deepseek-v2", buf.getvalue())


class TestAllBuiltinModelsAreWellFormed(unittest.TestCase):
    def test_every_model_computes_without_error_at_several_lengths(self):
        for key, m in kv_calc.MODELS.items():
            for tokens in (0, 1, 4096, 131072):
                with self.subTest(model=key, tokens=tokens):
                    r = kv_calc.kv_cache_bytes(m, tokens=tokens, batch=1)
                    self.assertGreaterEqual(r["total_bytes"], 0)
                    self.assertTrue(m.config_url.startswith("https://huggingface.co/"))


if __name__ == "__main__":
    unittest.main()
