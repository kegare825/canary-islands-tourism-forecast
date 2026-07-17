"""Tests de src/features.py — funciones puras sobre DataFrames pequeños en memoria."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_yoy_growth,
    build_features,
    build_features_for_island,
    feature_columns,
)


def _series(n_months: int, start="2020-01-01") -> pd.DataFrame:
    fechas = pd.date_range(start, periods=n_months, freq="MS")
    return pd.DataFrame({"fecha": fechas, "valor": np.arange(1, n_months + 1, dtype=float)})


def test_add_calendar_features_marks_high_season():
    df = add_calendar_features(_series(3, start="2020-01-01"))
    assert list(df["month"]) == [1, 2, 3]
    # Enero, febrero y marzo están en HIGH_SEASON_MONTHS
    assert df["is_high_season"].tolist() == [1, 1, 1]


def test_add_calendar_features_marks_low_season():
    df = add_calendar_features(_series(1, start="2020-05-01"))
    # Mayo no está en HIGH_SEASON_MONTHS
    assert df["is_high_season"].tolist() == [0]


def test_add_lag_features_shifts_correctly():
    df = add_lag_features(_series(4), target_col="valor", lags=(1,))
    assert df["valor_lag_1"].isna().iloc[0]
    assert df["valor_lag_1"].iloc[1] == 1.0
    assert df["valor_lag_1"].iloc[-1] == 3.0


def test_add_rolling_features_excludes_current_month():
    # valor = [1, 2, 3, 4]; rolling_mean_2 en la fila 3 (valor=3) debe promediar
    # los DOS meses anteriores (1, 2), no incluir el 3 -> evita fuga de futuro.
    df = add_rolling_features(_series(4), target_col="valor", windows=(2,))
    assert df["valor_rolling_mean_2"].iloc[2] == pytest.approx(1.5)


def test_add_yoy_growth_computes_interannual_change():
    df = _series(13)
    df = add_yoy_growth(df, target_col="valor")
    # mes 13 vs mes 1: (13 - 1) / 1 = 12.0
    assert df["valor_yoy_growth"].iloc[-1] == pytest.approx(12.0)
    assert df["valor_yoy_growth"].iloc[:12].isna().all()


def test_build_features_for_island_is_idempotent_on_row_count():
    df = _series(15)
    result = build_features_for_island(df, target_col="valor")
    assert len(result) == len(df)


def test_build_features_keeps_isla_column_across_groups():
    """Regresión: pandas 3.0 excluye la columna de agrupación en groupby.apply
    por defecto, lo que hacía desaparecer 'isla' del resultado (ver comentario
    en build_features). Este test falla si esa regresión reaparece."""
    df = pd.concat(
        [
            _series(14).assign(isla="Tenerife"),
            _series(14).assign(isla="Gran Canaria"),
        ],
        ignore_index=True,
    )
    result = build_features(df, target_col="valor")
    assert "isla" in result.columns
    assert set(result["isla"].unique()) == {"Tenerife", "Gran Canaria"}
    assert len(result) == len(df)


def test_build_features_does_not_mix_lags_across_islands():
    df = pd.concat(
        [
            _series(3, start="2020-01-01").assign(isla="A", valor=[10.0, 20.0, 30.0]),
            _series(3, start="2020-01-01").assign(isla="B", valor=[100.0, 200.0, 300.0]),
        ],
        ignore_index=True,
    )
    result = build_features(df, target_col="valor")
    lag_b = result.loc[result["isla"] == "B", "valor_lag_1"].tolist()
    # Si se mezclaran islas, el primer lag de B podría venir de A (30.0).
    assert lag_b[0] is None or pd.isna(lag_b[0])
    assert lag_b[1] == 100.0


def test_feature_columns_matches_generated_columns():
    df = _series(15)
    generated = build_features_for_island(df, target_col="valor")
    cols = feature_columns("valor")
    for col in cols:
        assert col in generated.columns, f"{col} no está entre las columnas generadas"
