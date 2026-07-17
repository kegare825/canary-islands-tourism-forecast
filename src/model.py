"""Modelos de forecasting + validación cruzada temporal.

La regla de oro de series temporales: nunca usar un split aleatorio. La
validación tiene que respetar el orden cronológico (ventana expansiva),
o el modelo "aprende del futuro" y las métricas mienten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-06-01")


@dataclass
class ForecastEvalResult:
    model_name: str
    mae: float
    rmse: float
    mape: float
    interval_coverage: float | None = None


def is_covid_period(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Marca observaciones dentro de la ruptura COVID usada en el proyecto."""
    dates = pd.DatetimeIndex(dates)
    return np.asarray((dates >= COVID_START) & (dates <= COVID_END))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fracción de observaciones dentro del intervalo de confianza pronosticado."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())


def seasonal_naive_forecast(series: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    """Baseline obligatorio antes de cualquier modelo "de verdad": repite el valor
    de hace `season_length` periodos. Si SARIMA/LightGBM no le ganan a esto,
    no están aportando nada.
    """
    last_season = series.iloc[-season_length:].values
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]


def expanding_window_splits(n_periods: int, min_train_size: int, horizon: int):
    """Genera índices (train_end, test_end) para validación con ventana expansiva.

    Cada fold entrena con todo el histórico disponible hasta ese punto y
    evalúa sobre los `horizon` periodos siguientes — simula cómo se usaría
    el modelo de verdad (pronosticar hacia adelante, nunca hacia atrás).
    """
    splits = []
    train_end = min_train_size
    while train_end + horizon <= n_periods:
        splits.append((train_end, train_end + horizon))
        train_end += horizon
    return splits


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> ForecastEvalResult:
    # RMSE calculado a mano (np.sqrt) en vez de squared=False: ese parámetro de
    # mean_squared_error fue retirado en versiones recientes de scikit-learn.
    return ForecastEvalResult(
        model_name=model_name,
        mae=mean_absolute_error(y_true, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mape=mean_absolute_percentage_error(y_true, y_pred),
    )


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas globales, excluyendo COVID y cobertura de intervalos SARIMA."""
    summary_rows = []
    for model, group in results.groupby("model"):
        row = {
            "model": model,
            "mae": group["mae"].mean(),
            "rmse": group["rmse"].mean(),
            "mape": group["mape"].mean(),
        }
        if "covid_fold" in group.columns:
            ex_covid = group[~group["covid_fold"]]
            row["mape_ex_covid"] = ex_covid["mape"].mean() if len(ex_covid) else float("nan")
        if model == "sarima" and "interval_coverage" in group.columns:
            row["interval_coverage"] = group["interval_coverage"].mean()
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).set_index("model")


def run_expanding_window_backtest(
    df_features: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int = 3,
    min_train_size: int = 36,
    lightgbm_variants: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Backtesting completo naive / SARIMA / LightGBM con ventana expansiva.

    `lightgbm_variants` permite comparar ablations, p. ej.
    ``{"lightgbm": cols_base, "lightgbm_exog": cols_base + exog}``.
    Si es None, se usa un único modelo ``lightgbm`` con `feature_cols`.
    """
    if lightgbm_variants is None:
        lightgbm_variants = {"lightgbm": feature_cols}

    rows = []

    for isla, grupo in df_features.groupby("isla", sort=False):
        grupo = grupo.sort_values("fecha").reset_index(drop=True)
        serie = grupo.set_index("fecha")[target_col]
        splits = expanding_window_splits(len(serie), min_train_size=min_train_size, horizon=horizon)

        for train_end, test_end in splits:
            train_s, test_s = serie.iloc[:train_end], serie.iloc[train_end:test_end]
            test_dates = test_s.index
            covid_fold = bool(is_covid_period(test_dates).any())
            base = {
                "isla": isla,
                "train_end": train_end,
                "test_end": test_end,
                "test_start": test_dates[0],
                "covid_fold": covid_fold,
            }

            naive_pred = seasonal_naive_forecast(train_s, horizon=horizon)
            r = evaluate_forecast(test_s.values, naive_pred, "naive_estacional")
            rows.append({**base, "model": r.model_name, "mae": r.mae, "rmse": r.rmse, "mape": r.mape, "interval_coverage": None})

            sarima_fit = fit_sarima(train_s)
            sarima_fc = sarima_fit.get_forecast(steps=horizon)
            sarima_pred = sarima_fc.predicted_mean
            conf = sarima_fc.conf_int()
            coverage = interval_coverage(test_s.values, conf.iloc[:, 0].values, conf.iloc[:, 1].values)
            r = evaluate_forecast(test_s.values, sarima_pred.values, "sarima")
            rows.append(
                {**base, "model": r.model_name, "mae": r.mae, "rmse": r.rmse, "mape": r.mape, "interval_coverage": coverage}
            )

            for model_name, cols in lightgbm_variants.items():
                X_train = grupo.loc[: train_end - 1, cols]
                y_train = grupo.loc[: train_end - 1, target_col]
                X_test = grupo.loc[train_end : test_end - 1, cols]
                lgbm_pred = fit_lightgbm(X_train, y_train).predict(X_test)
                r = evaluate_forecast(test_s.values, lgbm_pred, model_name)
                rows.append({**base, "model": r.model_name, "mae": r.mae, "rmse": r.rmse, "mape": r.mape, "interval_coverage": None})

    return pd.DataFrame(rows)


def fit_sarima(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """SARIMA con estacionalidad anual (periodo 12 = meses). Import perezoso de
    statsmodels aquí para no forzar la dependencia si solo se usa LightGBM.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    return model.fit(disp=False)


def forecast_sarima_with_intervals(fitted, horizon: int) -> tuple[pd.Series, pd.DataFrame]:
    """Pronóstico SARIMA con intervalo de confianza del 95 %."""
    fc = fitted.get_forecast(steps=horizon)
    return fc.predicted_mean, fc.conf_int()


def forecast_recursive_lightgbm(
    model, history: pd.DataFrame, target_col: str, horizon: int
) -> pd.Series:
    """Pronostica `horizon` meses hacia delante con un modelo tabular (LightGBM).

    A diferencia de SARIMA (que se extrapola solo con `.forecast()`), un
    modelo tabular solo sabe predecir UNA fila con sus features ya
    calculadas. Para varios meses hacia delante hay que generar cada mes de
    forma recursiva: predecir el mes+1, tratarlo como si fuera dato real para
    recalcular sus rezagos/medias móviles, predecir el mes+2 con eso, etc.
    El error de cada paso se propaga a los siguientes — esperable y
    documentado, no es un bug: es la limitación conocida de encadenar
    predicciones tabulares en vez de un modelo de series temporales nativo.

    `history` debe tener como mínimo las columnas `fecha` y `target_col`,
    ordenado cronológicamente, con al menos 12 meses de historia (para poder
    calcular `lag_12` del primer mes pronosticado).
    """
    from src.features import build_features_for_island, feature_columns

    cols = feature_columns(target_col)
    extended = history[["fecha", target_col]].copy()
    predictions = []

    for _ in range(horizon):
        next_date = extended["fecha"].max() + pd.DateOffset(months=1)
        candidate = pd.concat(
            [extended, pd.DataFrame({"fecha": [next_date], target_col: [float("nan")]})],
            ignore_index=True,
        )
        features_row = build_features_for_island(candidate, target_col).iloc[[-1]][cols]
        pred = float(model.predict(features_row)[0])
        predictions.append(pred)
        extended = pd.concat(
            [extended, pd.DataFrame({"fecha": [next_date], target_col: [pred]})],
            ignore_index=True,
        )

    forecast_dates = extended["fecha"].iloc[-horizon:]
    return pd.Series(predictions, index=forecast_dates, name=f"{target_col}_forecast")


def fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    from lightgbm import LGBMRegressor

    # min_child_samples bajo a propósito: con ~36-60 meses de entrenamiento en
    # los primeros folds de la ventana expansiva, el valor por defecto (20)
    # deja el árbol sin hojas válidas y LightGBM solo emite warnings sin
    # aprender nada. verbose=-1 silencia esos warnings esperables en folds
    # pequeños (no son errores, son folds con poco dato todavía).
    model = LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_samples=5, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def lightgbm_feature_importance(model, feature_cols: list[str]) -> pd.Series:
    """Importancia de features de un LGBMRegressor entrenado."""
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
