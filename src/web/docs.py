"""Render a Markdown file of ``docs/`` to HTML with a table of contents."""

import re
from pathlib import Path

from markdown_it import MarkdownIt

DOCS = {
    "agent": ("User guide", "AGENT.md"),
}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_doc(docs_dir, name):
    """``(title, html, toc)``; toc is a list of ``(level, id, text)`` for the h2 / h3 headings."""
    title, filename = DOCS[name]
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse((Path(docs_dir) / filename).read_text())
    toc, seen = [], set()
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag in ("h2", "h3"):
            text = tokens[i + 1].content
            ident = base = slug(text) or "section"
            n = 2
            while ident in seen:
                ident, n = f"{base}-{n}", n + 1
            seen.add(ident)
            tok.attrSet("id", ident)
            toc.append((int(tok.tag[1]), ident, text))
    return title, md.renderer.render(tokens, md.options, {}), toc
