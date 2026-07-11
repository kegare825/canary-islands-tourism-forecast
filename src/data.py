"""Carga de las series ya armonizadas de turismo en Canarias (ISTAC + INE).

Ver data/README.md para las fuentes exactas y el esquema esperado. La
armonización desde el formato "cubo" crudo de ISTAC/INE al formato largo de
aquí vive en notebooks/01_eda.ipynb, porque depende de inspeccionar el CSV
real tras descargarlo (los nombres de columna del export no están fijados
de antemano).
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_PATH = Path("data/processed/canarias_turismo_mensual.csv")

ISLANDS = [
    "Tenerife",
    "Gran Canaria",
    "Lanzarote",
    "Fuerteventura",
    "La Palma",
    "La Gomera",
    "El Hierro",
]

REQUIRED_COLUMNS = ["fecha", "isla"]
OPTIONAL_INDICATOR_COLUMNS = ["pernoctaciones", "viajeros", "adr_eur", "revpar_eur", "grado_ocupacion"]


def load_harmonized_series(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Carga la serie ya armonizada (fecha, isla, indicadores) y valida el esquema mínimo.

    No exige que TODAS las columnas de indicador estén presentes: distintas
    fuentes (ISTAC pernoctaciones vs. ISTAC ADR/RevPAR) pueden combinarse en
    momentos distintos del proyecto.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descarga y armoniza los datos siguiendo "
            "data/README.md y notebooks/01_eda.ipynb."
        )

    df = pd.read_csv(path, parse_dates=["fecha"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas mínimas en la serie armonizada: {missing}")

    present_indicators = [c for c in OPTIONAL_INDICATOR_COLUMNS if c in df.columns]
    if not present_indicators:
        raise ValueError(
            "La serie no trae ningún indicador reconocido "
            f"({OPTIONAL_INDICATOR_COLUMNS}) — revisa la armonización."
        )

    unknown_islands = set(df["isla"].unique()) - set(ISLANDS)
    if unknown_islands:
        raise ValueError(
            f"Nombres de isla no reconocidos: {unknown_islands}. "
            "Normaliza los nombres en la armonización (ej. 'Sta. Cruz de Tenerife' -> 'Tenerife')."
        )

    return df.sort_values(["isla", "fecha"]).reset_index(drop=True)


def to_wide_by_island(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """Pivota a una columna por isla para un indicador — útil para comparar series lado a lado."""
    return df.pivot(index="fecha", columns="isla", values=indicator)
