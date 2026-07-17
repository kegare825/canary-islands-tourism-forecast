#!/usr/bin/env python3
"""Recalcula métricas extendidas de backtesting (MAPE ex-COVID, cobertura IC SARIMA)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_features, feature_columns
from src.model import run_expanding_window_backtest, summarize_backtest

warnings.filterwarnings("ignore")

TARGET = "revpar_eur"
HORIZON = 3
MIN_TRAIN_SIZE = 36


def main() -> None:
    from src.data import load_harmonized_series

    df = load_harmonized_series()
    feat = build_features(df, target_col=TARGET)
    warmup = [c for c in feat.columns if "lag_" in c or "rolling_" in c]
    df_features = feat.dropna(subset=warmup).reset_index(drop=True)

    results = run_expanding_window_backtest(
        df_features, TARGET, feature_columns(TARGET), HORIZON, MIN_TRAIN_SIZE
    )
    summary = summarize_backtest(results)

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "model_eval_results.csv", index=False)
    summary.round(4).to_csv(out_dir / "model_eval_summary.csv")
    print(summary.round(4).to_string())
    print(f"\nSaved {out_dir / 'model_eval_results.csv'} and model_eval_summary.csv")


if __name__ == "__main__":
    main()
