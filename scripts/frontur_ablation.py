#!/usr/bin/env python3
"""Ablation: LightGBM baseline vs LightGBM + FRONTUR (turistas proxy exógena).

Descarga/armoniza FRONTUR, genera features con lags/rolling de turistas,
ejecuta backtesting de ventana expansiva y guarda comparación de MAPE.

Salidas:
- data/processed/frontur_turistas_mensual.csv
- data/processed/frontur_ablation_results.csv
- data/processed/frontur_ablation_summary.csv
- docs/frontur_ablation.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_harmonized_series
from src.features import EXOG_COL_DEFAULT, build_features, feature_columns, warmup_columns
from src.frontur import load_harmonized_frontur, merge_frontur_with_panel
from src.model import run_expanding_window_backtest, summarize_backtest

warnings.filterwarnings("ignore")

TARGET = "revpar_eur"
EXOG = EXOG_COL_DEFAULT
HORIZON = 3
MIN_TRAIN_SIZE = 36


def mape_by_island(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["isla", "model"])["mape"].mean().unstack("model").round(4)


def main() -> None:
    print("=== FRONTUR ablation: baseline vs exógena proxy ===\n")

    panel = load_harmonized_series()
    frontur = load_harmonized_frontur()
    merged = merge_frontur_with_panel(panel, frontur)

    coverage = merged.groupby("isla")["turistas"].apply(lambda s: s.notna().mean()).round(3)
    print("Cobertura FRONTUR (fracción meses con turistas no nulos):")
    print(coverage.to_string())
    print()

    feat = build_features(merged, target_col=TARGET, exog_col=EXOG)
    warmup = warmup_columns(TARGET, exog_col=EXOG)
    df_features = feat.dropna(subset=warmup_columns(TARGET)).reset_index(drop=True)

    cols_base = feature_columns(TARGET)
    cols_exog = feature_columns(TARGET, exog_col=EXOG)

    print(f"Filas tras warm-up target: {len(df_features)}")
    print(f"Features baseline: {len(cols_base)} | con FRONTUR: {len(cols_exog)}\n")

    results = run_expanding_window_backtest(
        df_features,
        TARGET,
        cols_base,
        HORIZON,
        MIN_TRAIN_SIZE,
        lightgbm_variants={
            "lightgbm": cols_base,
            "lightgbm_exog": cols_exog,
        },
    )
    summary = summarize_backtest(results)
    by_island = mape_by_island(results)

    if "lightgbm" in by_island.columns and "lightgbm_exog" in by_island.columns:
        by_island["delta_exog_minus_base"] = by_island["lightgbm_exog"] - by_island["lightgbm"]

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "frontur_ablation_results.csv", index=False)
    summary.round(4).to_csv(out_dir / "frontur_ablation_summary.csv")
    by_island.to_csv(out_dir / "frontur_ablation_by_island.csv")

    doc = _build_report(summary, by_island, coverage, cols_base, cols_exog)
    doc_path = ROOT / "docs/frontur_ablation.md"
    doc_path.write_text(doc, encoding="utf-8")

    print("=== Resumen global ===")
    print(summary.round(4).to_string())
    print("\n=== MAPE por isla (LightGBM vs LightGBM+FRONTUR) ===")
    print(by_island.to_string())
    print(f"\nGuardado: {doc_path}")


def _build_report(
    summary: pd.DataFrame,
    by_island: pd.DataFrame,
    coverage: pd.Series,
    cols_base: list[str],
    cols_exog: list[str],
) -> str:
    base_mape = summary.loc["lightgbm", "mape"] if "lightgbm" in summary.index else float("nan")
    exog_mape = summary.loc["lightgbm_exog", "mape"] if "lightgbm_exog" in summary.index else float("nan")
    delta = exog_mape - base_mape
    winner = "lightgbm_exog" if exog_mape < base_mape else "lightgbm"
    exog_only = [c for c in cols_exog if c not in cols_base]

    lines = [
        "# Ablation FRONTUR — variable exógena proxy (turistas)",
        "",
        "Comparación de backtesting (ventana expansiva, horizonte 3 meses) entre:",
        "",
        "- **`lightgbm`**: calendario + lags/rolling de RevPAR (baseline del proyecto).",
        "- **`lightgbm_exog`**: baseline + lags/rolling de **turistas FRONTUR** (proxy exógena).",
        "",
        "SARIMA y naive estacional se mantienen univariados (sin exógena).",
        "",
        "## Fuente exógena",
        "",
        "| Cubo ISTAC | Contenido |",
        "|---|---|",
        "| `E16028B_000016` | Turistas **principales**, residencia Total, 5 islas (2010+) |",
        "| `E16028B_000019` | Turistas La Gomera, residencia Total (2017+) |",
        "",
        "**El Hierro** no tiene serie FRONTUR publicada → `turistas` = NaN (LightGBM lo acepta).",
        "",
        "## Features añadidas (exógena)",
        "",
        "```text",
        *exog_only,
        "```",
        "",
        "## Cobertura FRONTUR por isla",
        "",
        _series_to_md_table(coverage.to_frame("frac_meses_con_dato")),
        "",
        "## Resultados globales",
        "",
        _df_to_md_table(summary.round(4).reset_index()),
        "",
        "## MAPE por isla",
        "",
        _df_to_md_table(by_island.reset_index()),
        "",
        "## Conclusión",
        "",
        f"- MAPE global LightGBM baseline: **{base_mape:.1%}**",
        f"- MAPE global LightGBM + FRONTUR: **{exog_mape:.1%}**",
        f"- Delta (exog − base): **{delta:+.1%}** → gana **`{winner}`** a nivel agregado.",
        "",
        "Re-ejecutar: `make frontur-ablation` o `python scripts/frontur_ablation.py`.",
        "",
    ]
    return "\n".join(lines)


def _df_to_md_table(df: pd.DataFrame) -> str:
    headers = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([headers, sep, *rows])


def _series_to_md_table(df: pd.DataFrame) -> str:
    return _df_to_md_table(df.reset_index())


def exog_feature_lines(cols_exog: list[str], cols_base: list[str]) -> list[str]:
    return [c for c in cols_exog if c not in cols_base]


if __name__ == "__main__":
    main()
