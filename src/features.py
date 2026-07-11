"""Feature engineering para forecasting de series temporales.

Todas las funciones operan sobre una serie ya ordenada por fecha para UNA
sola isla — el pipeline se aplica por grupo (`groupby("isla")`) para no
mezclar rezagos entre islas distintas.
"""

import numpy as np
import pandas as pd

# Meses de temporada alta en Canarias por afluencia de turismo de invierno
# (evasión de frío en Europa) + puente de Semana Santa/verano — a diferencia
# de la España peninsular, Canarias tiene una estacionalidad más suave
# repartida en dos picos (invierno + verano), no solo verano.
HIGH_SEASON_MONTHS = {1, 2, 3, 7, 8, 12}


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


def build_features_for_island(df_island: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Pipeline completo para una isla — llamar dentro de un groupby("isla").apply(...)."""
    df_island = df_island.sort_values("fecha")
    df_island = add_calendar_features(df_island)
    df_island = add_lag_features(df_island, target_col)
    df_island = add_rolling_features(df_island, target_col)
    df_island = add_yoy_growth(df_island, target_col)
    return df_island


def build_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Aplica el pipeline isla por isla, para no mezclar rezagos entre series distintas."""
    return (
        df.groupby("isla", group_keys=False)
        .apply(lambda g: build_features_for_island(g, target_col))
        .reset_index(drop=True)
    )
