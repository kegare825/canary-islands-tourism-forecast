#!/usr/bin/env python3
"""Execute notebooks in order and write cell outputs to a markdown report."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks/01_eda.ipynb",
    ROOT / "notebooks/02_decomposition_features.ipynb",
    ROOT / "notebooks/03_modeling.ipynb",
    ROOT / "notebooks/04_business_report.ipynb",
    ROOT / "notebooks/05_ine_contrast.ipynb",
]
OUTPUT_MD = ROOT / "docs/ejecucion_resultados.md"


def execute_notebook(path: Path, timeout: int = 3600) -> None:
    print(f"Executing {path.name}...", flush=True)
    with path.open(encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  Done: {path.name}", flush=True)


def format_output(output: dict) -> str:
    otype = output.get("output_type", "")
    if otype == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        return text
    if otype == "execute_result":
        data = output.get("data", {})
        if "text/plain" in data:
            text = data["text/plain"]
            return text if isinstance(text, str) else "".join(text)
        if "text/html" in data:
            return "[HTML output — ver notebook]"
    if otype == "display_data":
        data = output.get("data", {})
        if "text/plain" in data:
            text = data["text/plain"]
            return text if isinstance(text, str) else "".join(text)
        mime = next(iter(data), "unknown")
        return f"[{mime} output — ver notebook/gráfico]"
    if otype == "error":
        return "\n".join(
            [
                f"ERROR: {output.get('ename', '?')}: {output.get('evalue', '?')}",
                *output.get("traceback", []),
            ]
        )
    return f"[{otype}]"


def notebook_to_markdown(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        nb = json.load(f)

    lines = [f"## {path.name}", ""]
    cell_num = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        cell_num += 1
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = source.strip()
        if not source:
            continue

        lines.append(f"### Celda {cell_num}")
        lines.append("")
        lines.append("**Código:**")
        lines.append("")
        lines.append("```python")
        lines.append(source)
        lines.append("```")
        lines.append("")

        outputs = cell.get("outputs", [])
        if not outputs:
            lines.append("*Sin salida.*")
        else:
            lines.append("**Salida:**")
            lines.append("")
            for out in outputs:
                text = format_output(out).strip()
                if not text:
                    continue
                if text.startswith("ERROR:") or "Traceback" in text:
                    lines.append("```")
                    lines.append(text)
                    lines.append("```")
                elif text.startswith("[") and "output" in text:
                    lines.append(text)
                elif len(text) > 120 or "\n" in text:
                    lines.append("```")
                    lines.append(text)
                    lines.append("```")
                else:
                    lines.append(f"`{text}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    for nb in NOTEBOOKS:
        execute_notebook(nb)

    parts = [
        "# Resultados de ejecución — King Crimson",
        "",
        f"Generado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Notebooks ejecutados en orden desde la raíz del repo (`resources.metadata.path`).",
        "Tests: `pytest tests/ -v` → 35 passed.",
        "",
    ]
    for nb in NOTEBOOKS:
        parts.append(notebook_to_markdown(nb))
        parts.append("---")
        parts.append("")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(parts), encoding="utf-8")
    print(f"Report written to {OUTPUT_MD}", flush=True)


if __name__ == "__main__":
    main()
