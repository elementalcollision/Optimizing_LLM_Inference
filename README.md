# Optimizing LLM Inference with Key-Value Caching

A whitepaper on how the key-value (KV) cache shapes LLM serving, current to September 2026. It covers:

- how the cache affects time to first token (TTFT) and time per output token (TPOT);
- how serving engines manage it in GPU memory, and how attention architectures, compression and cross-request reuse make it smaller or cheaper;
- what happens when it is offloaded to CPU memory and SSDs, moved between machines, or kept in shared storage;
- how to measure and price its effects;
- the security risks of shared caches, and the cache as an editable model state.

**Read it:** [Optimizing LLM Inference with Key-Value Caching.md](Optimizing%20LLM%20Inference%20with%20Key-Value%20Caching.md)

## How it was checked

Every section was drafted from sources opened at the time of writing. Two independent reviewers then checked it: one confirmed that each citation supports its sentence, and the other recomputed the numbers and looked for counter-evidence. After that, the assembled document went through a whole-document review for overlap, cross-section consistency, flow, completeness and a spot check of high-stakes figures.

Vendor benchmark numbers are labelled as vendor-reported, with their setup. Fast-moving facts (context windows, API pricing, product availability) are dated.

## Layout

| Path | Purpose |
| --- | --- |
| `Optimizing LLM Inference with Key-Value Caching.md` | The rendered whitepaper. Generated; do not edit by hand. |
| `src/whitepaper.md` | The source. Cites with keys such as `[@vllm-sosp; @kivi]`. |
| `references/references.json` | One entry per source: authors, title, venue, year, URL. |
| `references/references.bib` | BibTeX export of the same entries. Generated. |
| `tools/render.py` | Numbers citations, builds the contents list and reference list, and regenerates the `.bib`. |
| `tools/kv_calc.py` | KV cache sizing calculator for GQA/MQA, MLA and hybrid sliding-window models (Appendix A). |
| `tools/test_kv_calc.py` | Tests for the calculator. |

## Editing

1. Edit `src/whitepaper.md`. Cite a source with `[@key]`, or several with `[@a; @b]`.
2. Add any new source to `references/references.json` under the key you used.
3. Run `python3 tools/render.py` (Python 3.8+, no dependencies) and commit the source, the data and the rendered output together.

The renderer stops on an unknown key, so a citation can never point at a missing entry.

## Sizing a cache

```sh
python3 tools/kv_calc.py --list                                  # built-in models
python3 tools/kv_calc.py --model llama3-8b --tokens 131072       # one model
python3 tools/kv_calc.py --markdown --tokens 131072 --dtype fp8  # table for all models
python3 tools/test_kv_calc.py                                    # tests
```

## License

MIT. See [LICENSE](LICENSE).
