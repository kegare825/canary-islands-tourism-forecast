"""Feature engineering para forecasting de series temporales.

Todas las funciones operan sobre una serie ya ordenada por fecha para UNA
sola isla — el pipeline se aplica por grupo (`groupby("isla")`) para no
mezclar rezagos entre islas distintas.
"""

import numpy as np
import pandas as pd

# Meses de temporada alta verificados con ocupación real derivada de ISTAC
# 2009-2026 (ver notebooks/01_eda.ipynb): los 6 meses con mayor ocupación
# media son agosto, noviembre, febrero, marzo, septiembre y enero — no
# coincide del todo con la intuición inicial de "invierno + verano clásico"
# (julio y diciembre resultan ser temporada media, no alta, en el dato real;
# noviembre sí es un pico real que la intuición inicial no capturaba).
HIGH_SEASON_MONTHS = {1, 2, 3, 8, 9, 11}

EXOG_COL_DEFAULT = "turistas"
TARGET_LAGS = (1, 3, 12)
TARGET_ROLLING = (3, 12)
EXOG_LAGS = (1, 3, 12)
EXOG_ROLLING = (3, 12)


def add_calendar_features(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["is_high_season"] = df["month"].isin(HIGH_SEASON_MONTHS).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: tuple[int, ...] = (1, 3, 12)) -> pd.DataFrame:
    """Rezagos del target — asume df ya ordenado por fecha dentro de una sola isla.

    lag_12 es el más importante en turismo: compara contra el mismo mes del
    año anterior, que es la comparación que de verdad le importa al negocio
    (no el mes anterior, que arrastra estacionalidad).
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: tuple[int, ...] = (3, 12)) -> pd.DataFrame:
    """Medias móviles — usa shift(1) antes del rolling para no incluir el propio mes (fuga de futuro)."""
    df = df.copy()
    shifted = df[target_col].shift(1)
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window).mean()
    return df


def add_yoy_growth(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Crecimiento interanual — la métrica que un gestor hotelero entiende de un vistazo."""
    df = df.copy()
    lag_12 = df[target_col].shift(12)
    df[f"{target_col}_yoy_growth"] = (df[target_col] - lag_12) / lag_12
    return df


def _base_feature_names(target_col: str) -> list[str]:
    return [
        "month", "quarter", "is_high_season",
        f"{target_col}_lag_1", f"{target_col}_lag_3", f"{target_col}_lag_12",
        f"{target_col}_rolling_mean_3", f"{target_col}_rolling_mean_12",
    ]


def exog_feature_names(exog_col: str = EXOG_COL_DEFAULT) -> list[str]:
    """Features de la variable exógena proxy (FRONTUR turistas): lags + rolling."""
    return [
        f"{exog_col}_lag_1", f"{exog_col}_lag_3", f"{exog_col}_lag_12",
        f"{exog_col}_rolling_mean_3", f"{exog_col}_rolling_mean_12",
    ]


def feature_columns(target_col: str, exog_col: str | None = None) -> list[str]:
    """Columnas de entrada para LightGBM — única fuente de verdad.

    Si `exog_col` está definido (p. ej. ``turistas``), añade rezagos y medias
    móviles de esa serie exógena proxy (FRONTUR).
    """
    cols = _base_feature_names(target_col)
    if exog_col:
        cols.extend(exog_feature_names(exog_col))
    return cols


def build_features_for_island(
    df_island: pd.DataFrame,
    target_col: str,
    exog_col: str | None = None,
) -> pd.DataFrame:
    """Pipeline completo para una isla — llamar dentro de un groupby("isla").apply(...)."""
    df_island = df_island.sort_values("fecha")
    df_island = add_calendar_features(df_island)
    df_island = add_lag_features(df_island, target_col, lags=TARGET_LAGS)
    df_island = add_rolling_features(df_island, target_col, windows=TARGET_ROLLING)
    df_island = add_yoy_growth(df_island, target_col)
    if exog_col and exog_col in df_island.columns:
        df_island = add_lag_features(df_island, exog_col, lags=EXOG_LAGS)
        df_island = add_rolling_features(df_island, exog_col, windows=EXOG_ROLLING)
    return df_island


def build_features(
    df: pd.DataFrame,
    target_col: str,
    exog_col: str | None = None,
) -> pd.DataFrame:
    """Aplica el pipeline isla por isla, para no mezclar rezagos entre series distintas.

    Se evita `groupby(...).apply(...)` a propósito: desde pandas 2.2 (y ya por
    defecto en pandas 3.0) la columna de agrupación se excluye del grupo que
    recibe la función salvo que se pase `include_groups=True`, lo que hacía
    desaparecer silenciosamente la columna `isla` del resultado final. Un
    `groupby` + `concat` explícito no depende de ese comportamiento.
    """
    return pd.concat(
        [
            build_features_for_island(grupo, target_col, exog_col=exog_col)
            for _, grupo in df.groupby("isla", sort=False)
        ],
        ignore_index=True,
    )


def warmup_columns(target_col: str, exog_col: str | None = None) -> list[str]:
    """Columnas cuyo NaN inicial delimita el warm-up (lags + rolling)."""
    cols = [c for c in _base_feature_names(target_col) if "lag_" in c or "rolling_" in c]
    if exog_col:
        cols.extend(exog_feature_names(exog_col))
    return cols
