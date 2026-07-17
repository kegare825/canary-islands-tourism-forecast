#!/usr/bin/env python3
"""Feature importance de LightGBM para las islas donde gana el backtesting."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import feature_columns
from src.model import lightgbm_feature_importance

FIG_PATH = ROOT / "reports/figures/06_lightgbm_feature_importance.png"
LIGHTGBM_ISLANDS = ["La Gomera", "La Palma"]


def main() -> None:
    cols = feature_columns("revpar_eur")
    fig, axes = plt.subplots(1, len(LIGHTGBM_ISLANDS), figsize=(12, 4.5), sharey=True)
    if len(LIGHTGBM_ISLANDS) == 1:
        axes = [axes]

    for ax, isla in zip(axes, LIGHTGBM_ISLANDS):
        slug = isla.lower().replace(" ", "_")
        path = ROOT / f"models/{slug}_lightgbm.pkl"
        with path.open("rb") as f:
            model = pickle.load(f)
        imp = lightgbm_feature_importance(model, cols).head(10)
        imp.sort_values().plot.barh(ax=ax, color="#2ecc71")
        ax.set_title(isla)
        ax.set_xlabel("Importancia")

    fig.suptitle("Top-10 features LightGBM (islas donde gana el backtesting)", y=1.02)
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
