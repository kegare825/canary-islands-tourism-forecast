"""Descarga de series complementarias del INE (Encuesta de Ocupación Hotelera).

Tabla Tempus3 67190 — viajeros y pernoctaciones por provincia, mensual.
Usada en `notebooks/05_ine_contrast.ipynb` para contrastar la tendencia
provincial/nacional frente al RevPAR por isla del ISTAC.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

INE_API_BASE = "https://servicios.ine.es/wstempus/js/es"
TABLE_PERNOCTACIONES_PROVINCIA = 67190

# COD de serie dentro de la tabla 67190 (ver SERIES_TABLA/67190)
INE_SERIES_COD = {
    "nacional_pernoctaciones": "EOT25",
    "canarias_pernoctaciones": "EOT1544",
    "las_palmas_pernoctaciones": "EOT1550",
    "tenerife_pernoctaciones": "EOT1556",
}


def fetch_ine_table(table_id: int = TABLE_PERNOCTACIONES_PROVINCIA, timeout: int = 120) -> list[dict]:
    url = f"{INE_API_BASE}/DATOS_TABLA/{table_id}?tip=AM"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_ine_series_by_cod(series_cod: str, table_id: int = TABLE_PERNOCTACIONES_PROVINCIA) -> pd.DataFrame:
    """Extrae una serie mensual concreta de la tabla EOH por su código COD."""
    payload = fetch_ine_table(table_id)
    match = next((row for row in payload if row.get("COD") == series_cod), None)
    if match is None:
        raise ValueError(f"Serie {series_cod} no encontrada en tabla INE {table_id}")

    rows = []
    for point in match["Data"]:
        rows.append(
            {
                "fecha": pd.to_datetime(point["Fecha"]),
                "valor": float(point["Valor"]),
                "serie_cod": series_cod,
                "serie_nombre": match.get("Nombre", ""),
                "unidad": match.get("T3_Unidad", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)


def load_canarias_contrast_series() -> pd.DataFrame:
    """Carga pernoctaciones mensuales: nacional, Canarias CCAA y sus dos provincias."""
    frames = []
    for label, series_cod in INE_SERIES_COD.items():
        df = fetch_ine_series_by_cod(series_cod)
        df["serie"] = label
        frames.append(df[["fecha", "valor", "serie", "serie_nombre"]])
    return pd.concat(frames, ignore_index=True)


def save_contrast_series(path: Path = Path("data/processed/ine_pernoctaciones_contrast.csv")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    load_canarias_contrast_series().to_csv(path, index=False)
    return path
