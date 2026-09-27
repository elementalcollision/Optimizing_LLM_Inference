#!/usr/bin/env python3
"""Render the whitepaper from its keyed source.

The source (src/whitepaper.md) cites with Pandoc-style keys, e.g. [@vllm-sosp; @kivi].
This script numbers citations in order of first appearance, links each number to its
entry, appends the reference list, and regenerates references/references.bib from
references/references.json. Run it after every edit to the source or the reference data:

    python3 tools/render.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "whitepaper.md"
OUT = ROOT / "Optimizing LLM Inference with Key-Value Caching.md"
REFS = ROOT / "references" / "references.json"
BIB = ROOT / "references" / "references.bib"

CITE = re.compile(r"\[(@[\w.-]+(?:\s*;\s*@[\w.-]+)*)\]")
TOC_MARKER = "<!-- toc -->"


def slug(heading):
    """GitHub's anchor for a heading: lowercase, drop punctuation, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def toc(markdown):
    """A contents list of the level-2 headings that follow the marker."""
    after = markdown.split(TOC_MARKER, 1)[1]
    lines = []
    for heading in re.findall(r"^## (.+)$", after, flags=re.M):
        lines.append(f"- [{heading}](#{slug(heading)})")
    lines.append("- [References](#references)")
    return "\n".join(lines)


def authors(names):
    if len(names) > 6:
        return ", ".join(names[:3]) + ", et al."
    if len(names) > 1:
        return ", ".join(names[:-1]) + ", and " + names[-1]
    return names[0]


def entry(ref):
    names = authors(ref["author"])
    parts = [names if names.endswith(".") else names + ".", f"“{ref['title']}.”", f"*{ref['container']}*"]
    tail = f"{ref['year']}." if ref.get("year") else "n.d."
    line = " ".join(parts) + f", {tail} <{ref['url']}>"
    if ref.get("note"):
        line += f" {ref['note']}"
    return line


def bibtex(key, ref):
    kind = {"conference": "inproceedings", "preprint": "misc", "report": "techreport"}.get(ref["type"], "misc")
    venue = {"inproceedings": "booktitle", "techreport": "institution"}.get(kind, "howpublished")
    fields = [
        ("author", " and ".join(ref["author"])),
        ("title", "{" + ref["title"] + "}"),
        (venue, ref["container"]),
        ("year", str(ref["year"]) if ref.get("year") else None),
        ("url", ref["url"]),
        ("note", ref.get("note")),
    ]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main():
    refs = json.loads(REFS.read_text())
    text = SRC.read_text()
    order = []

    def number(match):
        keys = [k.strip()[1:] for k in match.group(1).split(";")]
        links = []
        for key in keys:
            if key not in refs:
                sys.exit(f"unknown citation key: {key}")
            if key not in order:
                order.append(key)
            n = order.index(key) + 1
            links.append(f"[{n}](#ref-{n})")
        return "[" + ", ".join(links) + "]"

    body = CITE.sub(number, text).rstrip() + "\n"
    if TOC_MARKER in body:
        body = body.replace(TOC_MARKER, toc(body), 1)
    listing = ["", "## References", ""]
    for n, key in enumerate(order, 1):
        listing.append(f'<a id="ref-{n}"></a>{n}. {entry(refs[key])}')
        listing.append("")
    OUT.write_text(body + "\n".join(listing).rstrip() + "\n")
    BIB.write_text("\n".join(bibtex(k, refs[k]) for k in sorted(refs)))

    unused = sorted(set(refs) - set(order))
    print(f"rendered {len(order)} cited references to {OUT.name}")
    if unused:
        print("uncited entries in references.json (kept in the .bib only): " + ", ".join(unused))


if __name__ == "__main__":
    main()
