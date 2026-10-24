"""
make_pdfs.py
============
Converts outputs/research_paper.md and outputs/executive_summary.md into styled PDFs.

Blockquote callouts (the headline/verdict boxes) get distinct styling so the "no signal"
warning is visually impossible to miss when skimming, which is the whole point of putting
it in a callout in the first place.

    python make_pdfs.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

import config

CSS = """
@page {
    size: A4;
    margin: 22mm 18mm;
    @bottom-center { content: counter(page); font-size: 9px; color: #888; }
}
body {
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #1a1a1a;
    max-width: 100%;
}
h1 {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 20pt;
    border-bottom: 3px solid #2c3e50;
    padding-bottom: 8px;
    margin-top: 0;
    color: #1a1a1a;
}
h2 {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 14pt;
    color: #2c3e50;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 26px;
}
h3 {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 11.5pt;
    color: #34495e;
    margin-top: 18px;
}
p { margin: 8px 0; text-align: justify; }
strong { color: #1a1a1a; }
code {
    background: #f0f0f0; padding: 1px 4px; border-radius: 3px;
    font-family: "Courier New", monospace; font-size: 9pt;
}
pre {
    background: #f5f5f5; padding: 10px; border-radius: 4px;
    font-size: 8.5pt; overflow-x: auto; border: 1px solid #ddd;
}
table {
    border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt;
}
th {
    background: #2c3e50; color: white; padding: 6px 8px; text-align: left;
    font-family: "Helvetica", "Arial", sans-serif; font-size: 8.5pt;
}
td { padding: 5px 8px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) { background: #f8f9fa; }
hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }

/* Blockquote callouts - the headline / verdict boxes */
blockquote {
    margin: 14px 0; padding: 10px 16px;
    border-left: 5px solid #c0392b;
    background: #fdf2f2;
    border-radius: 0 4px 4px 0;
}
blockquote h1, blockquote h2 {
    border: none; margin-top: 0; font-size: 13pt; color: #c0392b;
}
blockquote p { margin: 6px 0; }
blockquote em { color: #555; }

img { max-width: 100%; display: block; margin: 14px auto; }

.titlepage {
    text-align: center; padding-top: 30%;
    font-family: "Helvetica", "Arial", sans-serif;
}
.titlepage h1 { border: none; font-size: 26pt; margin-bottom: 6px; }
.titlepage .subtitle { font-size: 13pt; color: #555; margin-top: 4px; }
.titlepage .meta { font-size: 10pt; color: #888; margin-top: 40px; }
"""


def render(md_path: Path, pdf_path: Path, title: str, subtitle: str) -> None:
    text = md_path.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )
    # Figure paths in the markdown are relative ("figures/x.png"); wkhtmltopdf needs them
    # resolved against the outputs/ directory it's rendering from.
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div class="titlepage">
  <h1>{title}</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="meta"><strong>Gauri Shinde</strong><br>Second-year Computer Science, University of Greenwich<br><br>EduPro &middot; Predictive Modeling for Course Demand and Revenue Forecasting</div>
</div>
<div style="page-break-before: always;"></div>
{body_html}
</body></html>"""

    html_path = md_path.with_suffix(".render.html")
    html_path.write_text(html, encoding="utf-8")

    subprocess.run(
        [
            "wkhtmltopdf", "--enable-local-file-access",
            "--footer-center", "[page]", "--footer-font-size", "8",
            str(html_path), str(pdf_path),
        ],
        check=True, cwd=md_path.parent, capture_output=True, text=True,
    )
    html_path.unlink()
    print(f"  wrote {pdf_path}  ({pdf_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    out = config.OUTPUT_DIR
    render(
        out / "research_paper.md", out / "research_paper.pdf",
        "Predictive Modeling for Course Demand and Revenue Forecasting on EduPro",
        "Research Paper",
    )
    render(
        out / "executive_summary.md", out / "executive_summary.pdf",
        "EduPro Course Demand & Revenue Forecasting",
        "Executive Summary",
    )


if __name__ == "__main__":
    main()
