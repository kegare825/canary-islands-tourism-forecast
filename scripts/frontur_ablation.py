#!/usr/bin/env python3
"""Ablation: LightGBM y SARIMAX con FRONTUR (turistas proxy exógena).

Features completas: mes, trimestre, lag_1/3/6/12, rolling_mean/std (3 y 12 meses)
para RevPAR y, en variantes exógenas, para turistas FRONTUR.

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
from src.features import EXOG_COL_DEFAULT, build_features, exog_feature_names, feature_columns, warmup_columns
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
    print("=== FRONTUR ablation: features completas + SARIMAX ===\n")

    panel = load_harmonized_series()
    frontur = load_harmonized_frontur()
    merged = merge_frontur_with_panel(panel, frontur)

    coverage = merged.groupby("isla")["turistas"].apply(lambda s: s.notna().mean()).round(3)
    print("Cobertura FRONTUR (fracción meses con turistas no nulos):")
    print(coverage.to_string())
    print()

    feat = build_features(merged, target_col=TARGET, exog_col=EXOG)
    df_features = feat.dropna(subset=warmup_columns(TARGET)).reset_index(drop=True)

    cols_base = feature_columns(TARGET)
    cols_exog = feature_columns(TARGET, exog_col=EXOG)
    sarimax_cols = exog_feature_names(EXOG)

    print(f"Filas tras warm-up RevPAR: {len(df_features)}")
    print(f"Features baseline: {len(cols_base)} | LightGBM+FRONTUR: {len(cols_exog)}")
    print(f"SARIMAX exog cols: {sarimax_cols}\n")

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
        sarimax_exog_cols=sarimax_cols,
    )
    summary = summarize_backtest(results)
    by_island = mape_by_island(results)

    if "lightgbm" in by_island.columns and "lightgbm_exog" in by_island.columns:
        by_island["delta_lgbm_exog"] = by_island["lightgbm_exog"] - by_island["lightgbm"]
    if "sarima" in by_island.columns and "sarimax_exog" in by_island.columns:
        by_island["delta_sarimax_exog"] = by_island["sarimax_exog"] - by_island["sarima"]

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "frontur_ablation_results.csv", index=False)
    summary.round(4).to_csv(out_dir / "frontur_ablation_summary.csv")
    by_island.to_csv(out_dir / "frontur_ablation_by_island.csv")

    doc = _build_report(summary, by_island, coverage, cols_base, cols_exog, sarimax_cols)
    doc_path = ROOT / "docs/frontur_ablation.md"
    doc_path.write_text(doc, encoding="utf-8")

    print("=== Resumen global ===")
    print(summary.round(4).to_string())
    print("\n=== MAPE por isla ===")
    print(by_island.to_string())
    print(f"\nGuardado: {doc_path}")


def _build_report(
    summary: pd.DataFrame,
    by_island: pd.DataFrame,
    coverage: pd.Series,
    cols_base: list[str],
    cols_exog: list[str],
    sarimax_cols: list[str],
) -> str:
    def _m(model: str, col: str) -> float:
        return float(summary.loc[model, col]) if model in summary.index else float("nan")

    exog_mape = _m("lightgbm_exog", "mape")
    base_mape = _m("lightgbm", "mape")
    exog_ex_covid = _m("lightgbm_exog", "mape_ex_covid")
    base_ex_covid = _m("lightgbm", "mape_ex_covid")
    delta_lgbm = exog_mape - base_mape
    winner_lgbm = "lightgbm_exog" if exog_mape < base_mape else "lightgbm"

    sarimax_mape = _m("sarimax_exog", "mape")
    sarima_mape = _m("sarima", "mape")
    delta_sarimax = sarimax_mape - sarima_mape
    winner_sarimax = "sarimax_exog" if sarimax_mape < sarima_mape else "sarima"

    improved_lgbm: list[str] = []
    improved_sarimax: list[str] = []
    if "delta_lgbm_exog" in by_island.columns:
        improved_lgbm = by_island.index[by_island["delta_lgbm_exog"] < 0].tolist()
    if "delta_sarimax_exog" in by_island.columns:
        improved_sarimax = by_island.index[by_island["delta_sarimax_exog"] < 0].tolist()

    lines = [
        "# Ablation FRONTUR — exógena proxy + features completas",
        "",
        "Backtesting (ventana expansiva, horizonte 3 meses):",
        "",
        "| Modelo | Descripción |",
        "|---|---|",
        "| `lightgbm` | mes, trimestre, lags 1/3/6/12, rolling mean/std de RevPAR |",
        "| `lightgbm_exog` | baseline + mismas transforms de **turistas FRONTUR** |",
        "| `sarima` | univariado (referencia) |",
        "| `sarimax_exog` | SARIMAX con columnas exógenas de turistas (lags + rolling) |",
        "",
        "SARIMAX solo corre en folds/islas donde **todo el train y test** tienen exógena",
        "sin NaN (El Hierro queda fuera; La Gomera solo folds recientes).",
        "",
        "## Fuente exógena (no es C00065A_000003)",
        "",
        "`C00065A_000003` = ADR/RevPAR hotelero (target). Llegadas = **FRONTUR** `E16028B`.",
        "",
        "| Cubo | Contenido |",
        "|---|---|",
        "| `E16028B_000016` | Turistas principales, 5 islas (2010+) |",
        "| `E16028B_000019` | Turistas La Gomera (2017+) |",
        "",
        "## Features baseline (RevPAR)",
        "",
        "```text",
        *cols_base,
        "```",
        "",
        "## Features exógenas añadidas (turistas)",
        "",
        "```text",
        *[c for c in cols_exog if c not in cols_base],
        "```",
        "",
        "## SARIMAX — columnas exógenas",
        "",
        "```text",
        *sarimax_cols,
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
        f"- LightGBM baseline: **{base_mape:.1%}** (ex-COVID {base_ex_covid:.1%})",
        f"- LightGBM + FRONTUR: **{exog_mape:.1%}** (ex-COVID {exog_ex_covid:.1%}, Δ {delta_lgbm:+.1%} pp → gana **{winner_lgbm}**)",
        f"- SARIMA: **{sarima_mape:.1%}** (ex-COVID {_m('sarima', 'mape_ex_covid'):.1%})",
        f"- SARIMAX + FRONTUR: **{sarimax_mape:.1%}** (ex-COVID {_m('sarimax_exog', 'mape_ex_covid'):.1%}, Δ {delta_sarimax:+.1%} pp → gana **{winner_sarimax}**)",
        "",
        f"- Islas donde FRONTUR mejora LightGBM: {', '.join(improved_lgbm) or 'ninguna'}",
        f"- Islas donde FRONTUR mejora SARIMA: {', '.join(improved_sarimax) or 'ninguna'}",
        "",
        "No se cambia el registry de producción: la mejora global es marginal y mixta por isla.",
        "",
        "Re-ejecutar: `make frontur-ablation`.",
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


if __name__ == "__main__":
    main()
