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
LAGS = (1, 3, 6, 12)
ROLLING_WINDOWS = (3, 12)


def add_calendar_features(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["is_high_season"] = df["month"].isin(HIGH_SEASON_MONTHS).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Rezagos del target — asume df ya ordenado por fecha dentro de una sola isla.

    lag_12 es el más importante en turismo: compara contra el mismo mes del
    año anterior, que es la comparación que de verdad le importa al negocio
    (no el mes anterior, que arrastra estacionalidad).
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_mean_features(
    df: pd.DataFrame, target_col: str, windows: tuple[int, ...] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Medias móviles — shift(1) antes del rolling para no incluir el propio mes."""
    df = df.copy()
    shifted = df[target_col].shift(1)
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window).mean()
    return df


def add_rolling_std_features(
    df: pd.DataFrame, target_col: str, windows: tuple[int, ...] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Desviación móvil — shift(1) antes del rolling (misma regla anti-fuga que la media)."""
    df = df.copy()
    shifted = df[target_col].shift(1)
    for window in windows:
        df[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window).std()
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Medias y desviaciones móviles (compatibilidad con llamadas existentes)."""
    df = add_rolling_mean_features(df, target_col, windows)
    return add_rolling_std_features(df, target_col, windows)


def add_yoy_growth(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Crecimiento interanual — la métrica que un gestor hotelero entiende de un vistazo."""
    df = df.copy()
    lag_12 = df[target_col].shift(12)
    df[f"{target_col}_yoy_growth"] = (df[target_col] - lag_12) / lag_12
    return df


def _lag_rolling_feature_names(col: str) -> list[str]:
    names = [f"{col}_lag_{lag}" for lag in LAGS]
    names.extend(f"{col}_rolling_mean_{w}" for w in ROLLING_WINDOWS)
    names.extend(f"{col}_rolling_std_{w}" for w in ROLLING_WINDOWS)
    return names


def _base_feature_names(target_col: str) -> list[str]:
    return [
        "month", "quarter", "is_high_season",
        *_lag_rolling_feature_names(target_col),
    ]


def exog_feature_names(exog_col: str = EXOG_COL_DEFAULT) -> list[str]:
    """Features de la variable exógena proxy (FRONTUR turistas): lags + rolling."""
    return _lag_rolling_feature_names(exog_col)


def feature_columns(target_col: str, exog_col: str | None = None) -> list[str]:
    """Columnas de entrada para LightGBM — única fuente de verdad.

    Si `exog_col` está definido (p. ej. ``turistas``), añade rezagos y rolling
    (mean + std) de esa serie exógena proxy (FRONTUR).
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
    df_island = add_lag_features(df_island, target_col, lags=LAGS)
    df_island = add_rolling_features(df_island, target_col, windows=ROLLING_WINDOWS)
    df_island = add_yoy_growth(df_island, target_col)
    if exog_col and exog_col in df_island.columns:
        df_island = add_lag_features(df_island, exog_col, lags=LAGS)
        df_island = add_rolling_features(df_island, exog_col, windows=ROLLING_WINDOWS)
    return df_island


def build_features(
    df: pd.DataFrame,
    target_col: str,
    exog_col: str | None = None,
) -> pd.DataFrame:
    """Aplica el pipeline isla por isla, para no mezclar rezagos entre series distintas."""
    return pd.concat(
        [
            build_features_for_island(grupo, target_col, exog_col=exog_col)
            for _, grupo in df.groupby("isla", sort=False)
        ],
        ignore_index=True,
    )


def warmup_columns(target_col: str, exog_col: str | None = None) -> list[str]:
    """Columnas cuyo NaN inicial delimita el warm-up (lags + rolling)."""
    cols = _lag_rolling_feature_names(target_col)
    if exog_col:
        cols.extend(exog_feature_names(exog_col))
    return cols
