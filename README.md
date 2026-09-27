# Optimizing LLM Inference with Key-Value Caching

A whitepaper on how the key-value (KV) cache shapes LLM serving: its effect on time to first token (TTFT) and time per output token (TPOT), how serving engines manage it in GPU memory, and what happens when it is offloaded to CPU memory, SSDs and shared storage.

**Read it:** [Optimizing LLM Inference with Key-Value Caching.md](Optimizing%20LLM%20Inference%20with%20Key-Value%20Caching.md)

## Layout

| Path | Purpose |
| --- | --- |
| `Optimizing LLM Inference with Key-Value Caching.md` | The rendered whitepaper. Generated; do not edit by hand. |
| `src/whitepaper.md` | The source. Cites with keys such as `[@vllm-sosp; @kivi]`. |
| `references/references.json` | One entry per source: authors, title, venue, year, URL. |
| `references/references.bib` | BibTeX export of the same entries. Generated. |
| `tools/render.py` | Numbers citations, builds the reference list and regenerates the `.bib`. |

## Editing

1. Edit `src/whitepaper.md`. Cite a source with `[@key]`, or several with `[@a; @b]`.
2. Add any new source to `references/references.json` under the key you used.
3. Run `python3 tools/render.py` (Python 3.8+, no dependencies) and commit the source, the data and the rendered output together.

The renderer stops on an unknown key, so a citation can never point at a missing entry.

## License

MIT. See [LICENSE](LICENSE).
