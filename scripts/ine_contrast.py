#!/usr/bin/env python3
"""Contraste INE (pernoctaciones provinciales) vs RevPAR agregado ISTAC."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_harmonized_series
from src.ine import save_contrast_series

FIG_PATH = ROOT / "reports/figures/05_ine_contrast_canarias.png"

# RevPAR ISTAC agregado por provincia (aproximación geográfica)
PROVINCE_ISLANDS = {
    "las_palmas": ["Gran Canaria", "Lanzarote", "Fuerteventura"],
    "tenerife": ["Tenerife", "La Palma", "La Gomera", "El Hierro"],
}


def main() -> None:
    ine_path = save_contrast_series()
    ine = pd.read_csv(ine_path, parse_dates=["fecha"])
    istac = load_harmonized_series()

    revpar_prov = {}
    for prov, islands in PROVINCE_ISLANDS.items():
        revpar_prov[prov] = (
            istac[istac["isla"].isin(islands)].groupby("fecha")["revpar_eur"].sum().rename(prov)
        )

    revpar_df = pd.DataFrame(revpar_prov)
    pernoct = ine.pivot_table(index="fecha", columns="serie", values="valor", aggfunc="sum")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    pernoct[["nacional_pernoctaciones", "canarias_pernoctaciones"]].plot(ax=axes[0], linewidth=1.8)
    axes[0].set_title("Pernoctaciones hoteleras INE (nacional vs Canarias)")
    axes[0].set_ylabel("Pernoctaciones")
    axes[0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color="grey", alpha=0.15)
    axes[0].legend(loc="upper left")

    revpar_df.plot(ax=axes[1], linewidth=1.8)
    axes[1].set_title("RevPAR mensual ISTAC agregado por provincia (suma de islas)")
    axes[1].set_ylabel("€ RevPAR")
    axes[1].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color="grey", alpha=0.15)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_PATH} and {ine_path}")


if __name__ == "__main__":
    main()
