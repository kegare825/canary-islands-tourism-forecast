"""Tests de src/model.py — funciones puras + smoke tests de fit/forecast con series pequeñas."""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    evaluate_forecast,
    expanding_window_splits,
    fit_lightgbm,
    forecast_recursive_lightgbm,
    interval_coverage,
    is_covid_period,
    seasonal_naive_forecast,
    summarize_backtest,
)


def test_seasonal_naive_forecast_repeats_last_season():
    series = pd.Series(range(1, 25))  # 24 meses, valores 1..24
    forecast = seasonal_naive_forecast(series, horizon=3, season_length=12)
    # Los últimos 12 valores son 13..24; el naive repite esos primeros 3 (13, 14, 15)
    assert list(forecast) == [13, 14, 15]


def test_seasonal_naive_forecast_handles_horizon_longer_than_season():
    series = pd.Series(range(1, 13))  # exactamente 1 temporada
    forecast = seasonal_naive_forecast(series, horizon=14, season_length=12)
    assert len(forecast) == 14


def test_expanding_window_splits_never_shrinks_train():
    splits = expanding_window_splits(n_periods=48, min_train_size=12, horizon=3)
    train_ends = [s[0] for s in splits]
    assert train_ends == sorted(train_ends)  # estrictamente creciente/no decreciente
    assert train_ends[0] == 12


def test_expanding_window_splits_respects_horizon_bound():
    splits = expanding_window_splits(n_periods=20, min_train_size=12, horizon=5)
    for train_end, test_end in splits:
        assert test_end - train_end == 5
        assert test_end <= 20


def test_expanding_window_splits_empty_when_too_short():
    assert expanding_window_splits(n_periods=10, min_train_size=12, horizon=3) == []


def test_evaluate_forecast_perfect_prediction_is_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    result = evaluate_forecast(y, y, "modelo_perfecto")
    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.mape == pytest.approx(0.0)


def test_evaluate_forecast_computes_known_mae():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    result = evaluate_forecast(y_true, y_pred, "x")
    assert result.mae == pytest.approx(2.0)


def test_interval_coverage_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert interval_coverage(y, y - 0.1, y + 0.1) == pytest.approx(1.0)


def test_is_covid_period_marks_2020_04():
    dates = pd.date_range("2020-04-01", periods=1, freq="MS")
    assert is_covid_period(dates)[0]


def test_summarize_backtest_excludes_covid_folds():
    results = pd.DataFrame(
        {
            "model": ["sarima", "sarima", "sarima"],
            "mae": [1.0, 2.0, 3.0],
            "rmse": [1.0, 2.0, 3.0],
            "mape": [0.10, 0.50, 0.20],
            "covid_fold": [False, True, False],
            "interval_coverage": [0.9, 0.8, 1.0],
        }
    )
    summary = summarize_backtest(results)
    assert summary.loc["sarima", "mape"] == pytest.approx((0.10 + 0.50 + 0.20) / 3)
    assert summary.loc["sarima", "mape_ex_covid"] == pytest.approx(0.15, rel=1e-3)
    assert summary.loc["sarima", "interval_coverage"] == pytest.approx(0.9, rel=1e-3)


def _synthetic_history(n_months: int = 30) -> pd.DataFrame:
    """Serie sintética con estacionalidad simple, suficiente para un smoke test de LightGBM."""
    fechas = pd.date_range("2021-01-01", periods=n_months, freq="MS")
    valores = [50 + 10 * np.sin(2 * np.pi * i / 12) + i * 0.5 for i in range(n_months)]
    return pd.DataFrame({"fecha": fechas, "revpar_eur": valores})


def test_fit_lightgbm_and_forecast_recursive_smoke():
    from src.features import build_features_for_island, feature_columns

    history = _synthetic_history(30)
    features = build_features_for_island(history, "revpar_eur").dropna(
        subset=[c for c in feature_columns("revpar_eur")]
    )
    model = fit_lightgbm(features[feature_columns("revpar_eur")], features["revpar_eur"])

    forecast = forecast_recursive_lightgbm(model, history, "revpar_eur", horizon=3)
    assert len(forecast) == 3
    assert not forecast.isna().any()
    # El pronóstico debe estar en un orden de magnitud razonable frente al histórico,
    # no en valores absurdos (asegura que las features recursivas no se rompieron).
    assert forecast.between(0, history["revpar_eur"].max() * 3).all()
