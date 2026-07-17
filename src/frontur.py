"""Descarga y armonización de FRONTUR-Canarias (ISTAC E16028B).

Turistas principales por isla y mes — variable exógena proxy de demanda
turística para enriquecer modelos tabulares (LightGBM).

Cubos usados:
- E16028B_000016: 5 islas grandes (2010+) — turistas principales, residencia Total.
- E16028B_000019: La Gomera (2017+) — turistas, residencia Total (metodología 2018).

El Hierro no aparece en FRONTUR publicado; queda `turistas` = NaN (LightGBM lo tolera).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.data import ISLANDS

ISTAC_API = "https://datos.canarias.es/api/estadisticas/statistical-resources/v1.0/datasets/ISTAC"

# (cubo, versión preferida, isla fija si el cubo no trae columna territorio útil)
FRONTUR_SOURCES = [
    ("E16028B_000016", "1.64", None),
    ("E16028B_000019", "1.4", "La Gomera"),
]

PROCESSED_PATH = Path("data/processed/frontur_turistas_mensual.csv")
RAW_DIR = Path("data/raw")

ISLANDS_IN_MAIN_CUBE = [
    "Tenerife",
    "Gran Canaria",
    "Lanzarote",
    "Fuerteventura",
    "La Palma",
]


def frontur_csv_url(cube_id: str, version: str) -> str:
    return f"{ISTAC_API}/{cube_id}/{version}.csv"


def download_frontur_cube(
    cube_id: str,
    version: str,
    dest: Path | None = None,
    timeout: int = 180,
) -> Path:
    """Descarga un cubo FRONTUR crudo a data/raw/."""
    dest = dest or RAW_DIR / f"{cube_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = frontur_csv_url(cube_id, version)
    response = requests.get(url, timeout=timeout)
    if not response.ok:
        # Reintento con versión anterior documentada en el catálogo
        fallback = "1.59" if cube_id == "E16028B_000016" else version
        if fallback != version:
            url = frontur_csv_url(cube_id, fallback)
            response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def _parse_monthly_period(code: str) -> pd.Timestamp | pd.NaT:
    if not isinstance(code, str) or "-M" not in code:
        return pd.NaT
    year, month = code.split("-M")
    return pd.Timestamp(f"{year}-{int(month):02d}-01")


def _is_total_residence(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized == "total" or normalized.startswith("total")


def harmonize_frontur_cube(raw: pd.DataFrame, fixed_isla: str | None = None) -> pd.DataFrame:
    """Convierte export SDMX de FRONTUR a (fecha, isla, turistas)."""
    df = raw.copy()

    if "MEDIDAS_CODE" in df.columns:
        # Cubo general: MEDIDAS_CODE == TURISTAS
        if (df["MEDIDAS_CODE"] == "TURISTAS").any():
            df = df[df["MEDIDAS_CODE"] == "TURISTAS"]
        elif (df["MEDIDAS_CODE"] == "VIAJEROS").any():
            df = df[(df["MEDIDAS_CODE"] == "VIAJEROS") & (df["TIPO_VIAJERO#es"] == "Turista")]
        else:
            raise ValueError("Cubo FRONTUR sin medida TURISTAS ni VIAJEROS reconocible")

    if "TIPO_VIAJERO#es" in df.columns and (df["TIPO_VIAJERO#es"] == "Turistas principales").any():
        df = df[df["TIPO_VIAJERO#es"] == "Turistas principales"]

    resid_col = "LUGAR_RESIDENCIA#es"
    if resid_col in df.columns:
        df = df[df[resid_col].map(_is_total_residence)]

    df = df[df["TIME_PERIOD_CODE"].astype(str).str.contains("-M", na=False)]
    df["fecha"] = df["TIME_PERIOD_CODE"].map(_parse_monthly_period)
    df = df.dropna(subset=["fecha"])

    if fixed_isla:
        df["isla"] = fixed_isla
    else:
        df["isla"] = df["TERRITORIO#es"]
        df = df[df["isla"].isin(ISLANDS_IN_MAIN_CUBE)]

    df["turistas"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    out = df[["fecha", "isla", "turistas"]].dropna(subset=["turistas"])
    return out.sort_values(["isla", "fecha"]).reset_index(drop=True)


def load_harmonized_frontur(
    path: Path = PROCESSED_PATH,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["fecha"])

    if not download_if_missing:
        raise FileNotFoundError(f"No se encontró {path} y download_if_missing=False")

    frames = []
    for cube_id, version, fixed_isla in FRONTUR_SOURCES:
        raw_path = download_frontur_cube(cube_id, version)
        raw = pd.read_csv(raw_path)
        frames.append(harmonize_frontur_cube(raw, fixed_isla=fixed_isla))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["isla", "fecha"]).drop_duplicates(["isla", "fecha"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def merge_frontur_with_panel(panel: pd.DataFrame, frontur: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left join de turistas FRONTUR sobre el panel (fecha, isla)."""
    frontur = frontur if frontur is not None else load_harmonized_frontur()
    merged = panel.merge(frontur, on=["fecha", "isla"], how="left")
    unknown = set(merged["isla"].unique()) - set(ISLANDS)
    if unknown:
        raise ValueError(f"Islas no reconocidas tras merge FRONTUR: {unknown}")
    return merged.sort_values(["isla", "fecha"]).reset_index(drop=True)
