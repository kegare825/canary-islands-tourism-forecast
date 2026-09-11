#!/usr/bin/env python3
"""Genera capturas estáticas (y GIF) de la demo Streamlit para el README."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.business import demand_alert_level, revpar_gap_vs_last_year
from src.data import load_harmonized_series
from src.model import forecast_recursive_lightgbm, forecast_sarima_with_intervals

FIG_DIR = ROOT / "reports/figures"
DEMO_DIR = FIG_DIR / "demo"


def _sidebar_panel(ax, isla: str, model_type: str, horizonte: int, ultimo_mes: str, adr: float, occ_pct: int):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02", fc="#f0f2f6", ec="#d0d4dc"))
    ax.text(0.08, 0.88, "Parámetros", fontsize=14, fontweight="bold")
    lines = [
        f"Isla: {isla}",
        f"Horizonte: {horizonte} meses",
        f"Modelo: {model_type}",
        f"Último dato real: {ultimo_mes}",
        f"ADR asumido: {adr:.1f} €",
        f"Ocupación año pasado: {occ_pct} %",
    ]
    y = 0.72
    for line in lines:
        ax.text(0.08, y, line, fontsize=11)
        y -= 0.11


def _forecast_panel(ax, historial: pd.Series, forecast: pd.Series, lower: pd.Series | None, upper: pd.Series | None, isla: str):
    ax.plot(historial.index, historial.values, color="#1f77b4", linewidth=2, label="Histórico RevPAR")
    ax.plot(forecast.index, forecast.values, color="#d62728", marker="o", linewidth=2, label="Pronóstico")
    if lower is not None and upper is not None:
        ax.fill_between(forecast.index, lower.values, upper.values, color="#d62728", alpha=0.15, label="IC 95 % SARIMA")
    ax.set_title(f"Pronóstico de RevPAR — {isla}")
    ax.set_ylabel("€")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)


def _alert_panel(ax, revpar_obj: float, ocupacion: float, gap: float, yoy: float, alerta: str):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02", fc="#ffffff", ec="#d0d4dc"))
    ax.text(0.08, 0.88, "Oportunidad de RevPAR", fontsize=14, fontweight="bold")
    metrics = [
        ("Ocupación pronosticada", f"{ocupacion:.0%}"),
        ("RevPAR pronosticado", f"{revpar_obj:,.1f} €"),
        ("Brecha vs año pasado", f"{gap:,.1f} €"),
        ("Crecimiento interanual", f"{yoy:+.1%}"),
        ("Nivel de alerta", alerta),
    ]
    y = 0.72
    for label, value in metrics:
        ax.text(0.08, y, f"{label}: {value}", fontsize=11)
        y -= 0.12


def _load_forecast(isla: str, registry: dict, df: pd.DataFrame, horizonte: int = 3):
    info = registry[isla]
    historial_isla = df[df["isla"] == isla].sort_values("fecha").reset_index(drop=True)
    with open(ROOT / info["path"], "rb") as f:
        model = pickle.load(f)

    lower = upper = None
    if info["model_type"] == "sarima":
        pred, conf = forecast_sarima_with_intervals(model, horizonte)
        fechas = pd.date_range(historial_isla["fecha"].max() + pd.DateOffset(months=1), periods=horizonte, freq="MS")
        forecast = pd.Series(pred.values, index=fechas)
        lower = pd.Series(conf.iloc[:, 0].values, index=fechas)
        upper = pd.Series(conf.iloc[:, 1].values, index=fechas)
    else:
        forecast = forecast_recursive_lightgbm(model, historial_isla, "revpar_eur", horizonte)

    return historial_isla, forecast, lower, upper, info["model_type"]


def generate(isla: str = "Gran Canaria", horizonte: int = 3) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / "models/registry.json").read_text())
    df = load_harmonized_series()
    historial_isla, forecast, lower, upper, model_type = _load_forecast(isla, registry, df, horizonte)

    ultimo_mes = historial_isla["fecha"].max().date()
    adr = float(historial_isla["adr_eur"].iloc[-1])
    fecha_obj = forecast.index[-1]
    revpar_obj = float(forecast.iloc[-1])
    mismo_mes = historial_isla[historial_isla["fecha"] == fecha_obj - pd.DateOffset(years=1)]
    occ_yp = float(mismo_mes["grado_ocupacion"].iloc[0]) if not mismo_mes.empty else 0.65
    revpar_yp = float(mismo_mes["revpar_eur"].iloc[0]) if not mismo_mes.empty else revpar_obj

    ocupacion = min(revpar_obj / adr, 1.0) if adr > 0 else 0.0
    opp = revpar_gap_vs_last_year(ocupacion, occ_yp, adr)
    yoy = (revpar_obj - revpar_yp) / revpar_yp if revpar_yp else 0.0
    alerta = demand_alert_level(yoy)

    # 1) Sidebar
    fig, ax = plt.subplots(figsize=(4, 5))
    _sidebar_panel(ax, isla, model_type, horizonte, str(ultimo_mes), adr, int(round(occ_yp * 100)))
    fig.savefig(DEMO_DIR / "01_sidebar.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2) Forecast + interval
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _forecast_panel(ax, historial_isla.set_index("fecha")["revpar_eur"].tail(36), forecast, lower, upper, isla)
    fig.savefig(DEMO_DIR / "02_forecast_intervalo.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3) Alert panel
    fig, ax = plt.subplots(figsize=(6, 4))
    _alert_panel(ax, revpar_obj, ocupacion, opp.gap_eur, yoy, alerta)
    fig.savefig(DEMO_DIR / "03_alerta_revpar.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 4) Composite for README
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2], height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    _sidebar_panel(ax0, isla, model_type, horizonte, str(ultimo_mes), adr, int(round(occ_yp * 100)))
    _forecast_panel(ax1, historial_isla.set_index("fecha")["revpar_eur"].tail(36), forecast, lower, upper, isla)
    _alert_panel(ax2, revpar_obj, ocupacion, opp.gap_eur, yoy, alerta)
    fig.suptitle("Canary Islands Tourism Forecast — demo interactiva (captura representativa)", fontsize=14, y=0.98)
    fig.savefig(DEMO_DIR / "00_demo_composite.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # GIF cycling islands
    from PIL import Image

    frames = []
    for isl in ["Gran Canaria", "La Gomera", "Tenerife"]:
        h, f, lo, up, _ = _load_forecast(isl, registry, df, horizonte=3)
        fig, ax = plt.subplots(figsize=(7, 4))
        _forecast_panel(ax, h.set_index("fecha")["revpar_eur"].tail(36), f, lo, up, isl)
        tmp = DEMO_DIR / f"_gif_{isl.replace(' ', '_')}.png"
        fig.savefig(tmp, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frames.append(Image.open(tmp))
        tmp.unlink(missing_ok=True)

    gif_path = DEMO_DIR / "demo_islas.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1800, loop=0)
    print(f"Demo assets written to {DEMO_DIR}")


if __name__ == "__main__":
    generate()
