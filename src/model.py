"""Modelos de forecasting + validación cruzada temporal.

La regla de oro de series temporales: nunca usar un split aleatorio. La
validación tiene que respetar el orden cronológico (ventana expansiva),
o el modelo "aprende del futuro" y las métricas mienten.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


@dataclass
class ForecastEvalResult:
    model_name: str
    mae: float
    rmse: float
    mape: float


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


def fit_sarima(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """SARIMA con estacionalidad anual (periodo 12 = meses). Import perezoso de
    statsmodels aquí para no forzar la dependencia si solo se usa LightGBM.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    return model.fit(disp=False)


def fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    return model
