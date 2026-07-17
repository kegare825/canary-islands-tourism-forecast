# Resultados de ejecución — King Crimson

Generado: 2026-07-17 18:28 UTC

Notebooks ejecutados en orden desde la raíz del repo (`resources.metadata.path`).
Tests: `pytest tests/ -v` → 35 passed.

## 01_eda.ipynb

### Celda 1

**Código:**

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

raw = pd.read_csv("data/raw/C00065A_000003.csv")
print(raw.shape)
raw.head()
```

**Salida:**

`(596736, 14)`
```
TIME_PERIOD#es TIME_PERIOD_CODE TERRITORIO#es TERRITORIO_CODE  \
0        01/2009         2009-M01      Canarias            ES70   
1        02/2009         2009-M02      Canarias            ES70   
2        03/2009         2009-M03      Canarias            ES70   
3        04/2009         2009-M04      Canarias            ES70   
4        05/2009         2009-M05      Canarias            ES70   

                MEDIDAS#es MEDIDAS_CODE ALOJAMIENTO_TURISTICO_CATEGORIA#es  \
0  Ingresos por habitación       REVPAR                              Total   
1  Ingresos por habitación       REVPAR                              Total   
2  Ingresos por habitación       REVPAR                              Total   
3  Ingresos por habitación       REVPAR                              Total   
4  Ingresos por habitación       REVPAR                              Total   

  ALOJAMIENTO_TURISTICO_CATEGORIA_CODE  OBS_VALUE  NOTAS_OBSERVACION#es  \
0                                   _T      48.05                   NaN   
1                                   _T      43.83                   NaN   
2                                   _T      46.44                   NaN   
3                                   _T      43.95                   NaN   
4                                   _T      32.73                   NaN   

  CONFIDENCIALIDAD_OBSERVACION#es CONFIDENCIALIDAD_OBSERVACION_CODE  \
0                             NaN                               NaN   
1                             NaN                               NaN   
2                             NaN                               NaN   
3                             NaN                               NaN   
4                             NaN                               NaN   

  ESTADO_OBSERVACION#es ESTADO_OBSERVACION_CODE  
0                   NaN                     NaN  
1                   NaN                     NaN  
2                   NaN                     NaN  
3                   NaN                     NaN  
4                   NaN                     NaN
```

### Celda 2

**Código:**

```python
from src.data import ISLANDS

MEASURES = {"Tarifa media diaria": "adr_eur", "Ingresos por habitación": "revpar_eur"}

is_monthly = raw["TIME_PERIOD_CODE"].str.contains("-M", na=False)
is_island = raw["TERRITORIO#es"].isin(ISLANDS)
is_total_category = raw["ALOJAMIENTO_TURISTICO_CATEGORIA#es"] == "Total"
is_target_measure = raw["MEDIDAS#es"].isin(MEASURES)

filtered = raw[is_monthly & is_island & is_total_category & is_target_measure].copy()
print(f"{len(raw)} filas crudas -> {len(filtered)} filas tras filtrar")

filtered["fecha"] = pd.to_datetime(filtered["TIME_PERIOD_CODE"], format="%Y-M%m")
filtered["indicador"] = filtered["MEDIDAS#es"].map(MEASURES)
filtered["isla"] = filtered["TERRITORIO#es"]

wide = filtered.pivot_table(index=["fecha", "isla"], columns="indicador", values="OBS_VALUE").reset_index()
wide.columns.name = None
wide["grado_ocupacion"] = wide["revpar_eur"] / wide["adr_eur"]

print(wide.shape)
wide.sort_values(["isla", "fecha"]).head()
```

**Salida:**

```
596736 filas crudas -> 2870 filas tras filtrar
(1414, 5)
```
```
fecha       isla  adr_eur  revpar_eur  grado_ocupacion
0  2009-01-01  El Hierro    56.55        8.38         0.148187
7  2009-02-01  El Hierro    58.57       10.26         0.175175
14 2009-03-01  El Hierro    60.87       11.75         0.193034
21 2009-04-01  El Hierro    66.49       21.12         0.317642
28 2009-05-01  El Hierro    67.40       14.48         0.214837
```

### Celda 3

**Código:**

```python
print("Meses por isla:")
print(wide.groupby("isla")["fecha"].count())
print()
print("Rango de fechas:", wide["fecha"].min().date(), "->", wide["fecha"].max().date())
print()
print("grado_ocupacion fuera de [0, 1] (posibles anomalías de la fuente):")
anomalias = wide[(wide["grado_ocupacion"] < 0) | (wide["grado_ocupacion"] > 1)]
print(len(anomalias), "filas")
anomalias.sort_values("grado_ocupacion", ascending=False).head()
```

**Salida:**

```
Meses por isla:
isla
El Hierro        202
Fuerteventura    202
Gran Canaria     202
La Gomera        202
La Palma         202
Lanzarote        202
Tenerife         202
Name: fecha, dtype: int64

Rango de fechas: 2009-01-01 -> 2026-01-01

grado_ocupacion fuera de [0, 1] (posibles anomalías de la fuente):
0 filas
```
```
Empty DataFrame
Columns: [fecha, isla, adr_eur, revpar_eur, grado_ocupacion]
Index: []
```

### Celda 4

**Código:**

```python
from pathlib import Path

output_path = Path("data/processed/canarias_turismo_mensual.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

final = wide.sort_values(["isla", "fecha"])[["fecha", "isla", "adr_eur", "revpar_eur", "grado_ocupacion"]]
final.to_csv(output_path, index=False)
print(f"Guardado {len(final)} filas en {output_path}")

# Verificar que src/data.py puede cargarlo sin errores
from src.data import load_harmonized_series

df = load_harmonized_series()
df.head()
```

**Salida:**

`Guardado 1414 filas en data/processed/canarias_turismo_mensual.csv`
```
fecha       isla  adr_eur  revpar_eur  grado_ocupacion
0 2009-01-01  El Hierro    56.55        8.38         0.148187
1 2009-02-01  El Hierro    58.57       10.26         0.175175
2 2009-03-01  El Hierro    60.87       11.75         0.193034
3 2009-04-01  El Hierro    66.49       21.12         0.317642
4 2009-05-01  El Hierro    67.40       14.48         0.214837
```

### Celda 5

**Código:**

```python
from src.data import to_wide_by_island

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

wide_revpar = to_wide_by_island(df, "revpar_eur")
wide_revpar.plot(ax=axes[0])
axes[0].set_title("RevPAR mensual por isla (€)")
axes[0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color="grey", alpha=0.2, label="COVID")
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))

wide_occ = to_wide_by_island(df, "grado_ocupacion")
wide_occ.plot(ax=axes[1], legend=False)
axes[1].set_title("Ocupación derivada (RevPAR / ADR) por isla")
axes[1].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color="grey", alpha=0.2)

plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/01_revpar_ocupacion_por_isla.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Salida:**

`<Figure size 1200x900 with 2 Axes>`

### Celda 6

**Código:**

```python
monthly_avg = df.groupby(df["fecha"].dt.month)["grado_ocupacion"].mean()
print("Ocupación media por mes (todas las islas, todos los años):")
print(monthly_avg.round(3))

peak_months = monthly_avg.sort_values(ascending=False).head(4).index.tolist()
print(f"\nMeses de mayor ocupación real: {sorted(peak_months)}")
print("HIGH_SEASON_MONTHS asumidos en src/features.py: {1, 2, 3, 7, 8, 12}")

seasonality_by_island = df.groupby("isla")["grado_ocupacion"].agg(["mean", "std"])
seasonality_by_island["coef_variacion"] = seasonality_by_island["std"] / seasonality_by_island["mean"]
seasonality_by_island.sort_values("coef_variacion", ascending=False)
```

**Salida:**

```
Ocupación media por mes (todas las islas, todos los años):
fecha
1     0.674
2     0.692
3     0.689
4     0.618
5     0.574
6     0.603
7     0.641
8     0.721
9     0.682
10    0.667
11    0.718
12    0.640
Name: grado_ocupacion, dtype: float64

Meses de mayor ocupación real: [2, 3, 8, 11]
HIGH_SEASON_MONTHS asumidos en src/features.py: {1, 2, 3, 7, 8, 12}
```
```
mean       std  coef_variacion
isla                                             
La Palma       0.542825  0.185529        0.341784
La Gomera      0.591839  0.196969        0.332808
El Hierro      0.458098  0.148707        0.324619
Lanzarote      0.757504  0.164397        0.217025
Tenerife       0.752681  0.144906        0.192520
Gran Canaria   0.770541  0.142977        0.185554
Fuerteventura  0.752964  0.136596        0.181411
```

---

## 02_decomposition_features.ipynb

### Celda 1

**Código:**

```python
from pathlib import Path

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data import load_harmonized_series, to_wide_by_island
from src.features import build_features

df = load_harmonized_series()
df_features = build_features(df, target_col="revpar_eur")
df_features.head()
```

**Salida:**

```
fecha       isla  adr_eur  revpar_eur  grado_ocupacion  month  quarter  \
0 2009-01-01  El Hierro    56.55        8.38         0.148187      1        1   
1 2009-02-01  El Hierro    58.57       10.26         0.175175      2        1   
2 2009-03-01  El Hierro    60.87       11.75         0.193034      3        1   
3 2009-04-01  El Hierro    66.49       21.12         0.317642      4        2   
4 2009-05-01  El Hierro    67.40       14.48         0.214837      5        2   

   is_high_season  revpar_eur_lag_1  revpar_eur_lag_3  revpar_eur_lag_12  \
0               1               NaN               NaN                NaN   
1               1              8.38               NaN                NaN   
2               1             10.26               NaN                NaN   
3               0             11.75              8.38                NaN   
4               0             21.12             10.26                NaN   

   revpar_eur_rolling_mean_3  revpar_eur_rolling_mean_12  \
0                        NaN                         NaN   
1                        NaN                         NaN   
2                        NaN                         NaN   
3                  10.130000                         NaN   
4                  14.376667                         NaN   

   revpar_eur_yoy_growth  
0                    NaN  
1                    NaN  
2                    NaN  
3                    NaN  
4                    NaN
```

### Celda 2

**Código:**

```python
wide = to_wide_by_island(df, "revpar_eur")

fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
for ax, isla in zip(axes.flat, wide.columns):
    result = seasonal_decompose(wide[isla].dropna(), model="additive", period=12)
    ax.plot(result.trend, label="tendencia")
    ax.plot(result.seasonal + result.trend.mean(), label="estacional (+ media tendencia)", alpha=0.6)
    ax.set_title(isla, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

axes.flat[-1].axis("off")
plt.suptitle("Descomposición aditiva de RevPAR por isla (periodo=12 meses)")
plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/02_decomposicion_estacional.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Salida:**

`<Figure size 1300x1400 with 8 Axes>`

### Celda 3

**Código:**

```python
print(f"Filas antes de descartar warm-up de lags/rolling: {len(df_features)}")

usable = df_features.dropna(subset=[c for c in df_features.columns if "lag_" in c or "rolling_" in c])
print(f"Filas utilizables tras descartar warm-up (necesita lag_12 + rolling_12): {len(usable)}")
print(f"Se pierden los primeros 12 meses por isla (7 islas x 12 = 84 filas) por diseño: nunca hay fuga de futuro.")

output_path = Path("data/processed/canarias_features.csv")
df_features.to_csv(output_path, index=False)
print(f"\nGuardado (incluye NaNs de warm-up, tal cual las genera build_features) en {output_path}")
df_features[df_features["isla"] == "Tenerife"].tail()
```

**Salida:**

```
Filas antes de descartar warm-up de lags/rolling: 1414
Filas utilizables tras descartar warm-up (necesita lag_12 + rolling_12): 1330
Se pierden los primeros 12 meses por isla (7 islas x 12 = 84 filas) por diseño: nunca hay fuga de futuro.

Guardado (incluye NaNs de warm-up, tal cual las genera build_features) en data/processed/canarias_features.csv
```
```
fecha      isla  adr_eur  revpar_eur  grado_ocupacion  month  \
1409 2025-09-01  Tenerife   128.68      107.60         0.836183      9   
1410 2025-10-01  Tenerife   141.37      119.67         0.846502     10   
1411 2025-11-01  Tenerife   151.11      132.33         0.875720     11   
1412 2025-12-01  Tenerife   171.63      138.32         0.805920     12   
1413 2026-01-01  Tenerife   169.45      142.77         0.842549      1   

      quarter  is_high_season  revpar_eur_lag_1  revpar_eur_lag_3  \
1409        3               1            123.32             95.51   
1410        4               0            107.60            112.96   
1411        4               1            119.67            123.32   
1412        4               0            132.33            107.60   
1413        1               1            138.32            119.67   

      revpar_eur_lag_12  revpar_eur_rolling_mean_3  \
1409             105.59                 110.596667   
1410             114.72                 114.626667   
1411             129.45                 116.863333   
1412             132.19                 119.866667   
1413             132.20                 130.106667   

      revpar_eur_rolling_mean_12  revpar_eur_yoy_growth  
1409                  117.940000               0.019036  
1410                  118.107500               0.043149  
1411                  118.520000               0.022248  
1412                  118.760000               0.046373  
1413                  119.270833               0.079955
```

---

## 03_modeling.ipynb

### Celda 1

**Código:**

```python
import warnings

import pandas as pd

from src.data import load_harmonized_series
from src.features import build_features, feature_columns
from src.model import evaluate_forecast, expanding_window_splits, fit_lightgbm, fit_sarima, seasonal_naive_forecast

# SARIMA/LightGBM emiten warnings esperables en folds pequeños (ventana
# expansiva empieza en solo 36 meses de historia) — silenciados a propósito
# para no inundar la salida del notebook; no afectan al resultado.
warnings.filterwarnings("ignore")

TARGET = "revpar_eur"
HORIZON = 3
MIN_TRAIN_SIZE = 36
FEATURE_COLS = feature_columns(TARGET)

df = load_harmonized_series()
feat_full = build_features(df, target_col=TARGET)
warmup_cols = [c for c in feat_full.columns if "lag_" in c or "rolling_" in c]
df_features = feat_full.dropna(subset=warmup_cols).reset_index(drop=True)
print(f"{len(feat_full)} filas totales -> {len(df_features)} tras descartar warm-up de lags/rolling")
```

**Salida:**

`1414 filas totales -> 1330 tras descartar warm-up de lags/rolling`

### Celda 2

**Código:**

```python
rows = []

for isla, grupo in df_features.groupby("isla", sort=False):
    grupo = grupo.sort_values("fecha").reset_index(drop=True)
    serie = grupo.set_index("fecha")[TARGET]
    splits = expanding_window_splits(len(serie), min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON)

    for train_end, test_end in splits:
        train_s, test_s = serie.iloc[:train_end], serie.iloc[train_end:test_end]

        naive_pred = seasonal_naive_forecast(train_s, horizon=HORIZON)
        r = evaluate_forecast(test_s.values, naive_pred, "naive_estacional")
        rows.append((isla, train_end, test_end, r.model_name, r.mae, r.rmse, r.mape))

        sarima_fit = fit_sarima(train_s)
        sarima_pred = sarima_fit.forecast(steps=HORIZON)
        r = evaluate_forecast(test_s.values, sarima_pred.values, "sarima")
        rows.append((isla, train_end, test_end, r.model_name, r.mae, r.rmse, r.mape))

        X_train, y_train = grupo.loc[: train_end - 1, FEATURE_COLS], grupo.loc[: train_end - 1, TARGET]
        X_test = grupo.loc[train_end : test_end - 1, FEATURE_COLS]
        lgbm_pred = fit_lightgbm(X_train, y_train).predict(X_test)
        r = evaluate_forecast(test_s.values, lgbm_pred, "lightgbm")
        rows.append((isla, train_end, test_end, r.model_name, r.mae, r.rmse, r.mape))

    print(f"{isla}: {len(splits)} folds evaluados (x3 modelos)")

results = pd.DataFrame(rows, columns=["isla", "train_end", "test_end", "model", "mae", "rmse", "mape"])
print(f"\nTotal filas fold x modelo: {len(results)}")
```

**Salida:**

```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
`El Hierro: 51 folds evaluados (x3 modelos)`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
`Fuerteventura: 51 folds evaluados (x3 modelos)`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
`Gran Canaria: 51 folds evaluados (x3 modelos)`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
`La Gomera: 51 folds evaluados (x3 modelos)`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
`La Palma: 51 folds evaluados (x3 modelos)`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
`Lanzarote: 51 folds evaluados (x3 modelos)`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
Tenerife: 51 folds evaluados (x3 modelos)

Total filas fold x modelo: 1071
```

### Celda 3

**Código:**

```python
print("=== MAPE medio global por modelo (menor es mejor) ===")
print(results.groupby("model")[["mae", "rmse", "mape"]].mean().round(3))

print("\n=== MAPE medio por isla y modelo ===")
mape_por_isla = results.groupby(["isla", "model"])["mape"].mean().unstack().round(3)
print(mape_por_isla)

winners = (
    results.groupby(["isla", "model"])["mape"].mean().reset_index()
    .sort_values(["isla", "mape"]).groupby("isla").first()[["model", "mape"]]
)
print("\n=== Modelo ganador por isla (menor MAPE medio) ===")
print(winners)

from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
results.to_csv("data/processed/model_eval_results.csv", index=False)
winners.to_csv("data/processed/model_winners_by_island.csv")
print("\nGuardados data/processed/model_eval_results.csv y model_winners_by_island.csv")
```

**Salida:**

```
=== MAPE medio global por modelo (menor es mejor) ===
                     mae    rmse   mape
model                                  
lightgbm           8.329   9.350  0.180
naive_estacional  12.626  13.650  0.298
sarima             8.084   8.993  0.183

=== MAPE medio por isla y modelo ===
model          lightgbm  naive_estacional  sarima
isla                                             
El Hierro         0.230             0.251   0.208
Fuerteventura     0.136             0.235   0.135
Gran Canaria      0.155             0.275   0.133
La Gomera         0.172             0.332   0.254
La Palma          0.266             0.326   0.296
Lanzarote         0.171             0.431   0.137
Tenerife          0.132             0.238   0.120

=== Modelo ganador por isla (menor MAPE medio) ===
                  model      mape
isla                             
El Hierro        sarima  0.208008
Fuerteventura    sarima  0.135151
Gran Canaria     sarima  0.132536
La Gomera      lightgbm  0.171923
La Palma       lightgbm  0.266335
Lanzarote        sarima  0.136970
Tenerife         sarima  0.119837

Guardados data/processed/model_eval_results.csv y model_winners_by_island.csv
```

### Celda 4

**Código:**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
mape_por_isla.plot.bar(ax=ax)
ax.set_ylabel("MAPE medio (357 folds de ventana expansiva)")
ax.set_title("Naive estacional vs SARIMA vs LightGBM, por isla")
ax.legend(title="Modelo")
plt.tight_layout()
plt.savefig("reports/figures/03_comparacion_modelos_por_isla.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Salida:**

`<Figure size 900x500 with 1 Axes>`

### Celda 5

**Código:**

```python
import pickle

Path("models").mkdir(parents=True, exist_ok=True)
model_registry = {}

for isla, grupo in df_features.groupby("isla", sort=False):
    grupo = grupo.sort_values("fecha").reset_index(drop=True)
    winning_model = winners.loc[isla, "model"]

    if winning_model == "sarima":
        serie = grupo.set_index("fecha")[TARGET]
        fitted = fit_sarima(serie)
    else:
        fitted = fit_lightgbm(grupo[FEATURE_COLS], grupo[TARGET])

    slug = isla.lower().replace(" ", "_")
    model_path = Path(f"models/{slug}_{winning_model}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(fitted, f)

    model_registry[isla] = {"model_type": winning_model, "path": str(model_path)}
    print(f"{isla}: {winning_model} entrenado con {len(grupo)} meses -> {model_path}")

import json

with open("models/registry.json", "w") as f:
    json.dump(model_registry, f, indent=2, ensure_ascii=False)
print("\nGuardado models/registry.json")
```

**Salida:**

```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
`El Hierro: sarima entrenado con 190 meses -> models/el_hierro_sarima.pkl`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
`Fuerteventura: sarima entrenado con 190 meses -> models/fuerteventura_sarima.pkl`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
`Gran Canaria: sarima entrenado con 190 meses -> models/gran_canaria_sarima.pkl`
`La Gomera: lightgbm entrenado con 190 meses -> models/la_gomera_lightgbm.pkl`
`La Palma: lightgbm entrenado con 190 meses -> models/la_palma_lightgbm.pkl`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
`Lanzarote: sarima entrenado con 190 meses -> models/lanzarote_sarima.pkl`
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
```
```
Tenerife: sarima entrenado con 190 meses -> models/tenerife_sarima.pkl

Guardado models/registry.json
```

### Celda 6

**Código:**

```python
from src.model import forecast_recursive_lightgbm

for isla, info in model_registry.items():
    grupo = df_features[df_features["isla"] == isla].sort_values("fecha")
    with open(info["path"], "rb") as f:
        fitted = pickle.load(f)

    if info["model_type"] == "sarima":
        forecast = fitted.forecast(steps=HORIZON)
    else:
        forecast = forecast_recursive_lightgbm(fitted, grupo, TARGET, HORIZON)

    last_real = grupo[TARGET].iloc[-1]
    print(f"{isla} ({info['model_type']}) — último real: {last_real:.1f} € | pronóstico +{HORIZON}m: {list(forecast.round(1))}")
    assert not forecast.isna().any(), f"{isla}: el pronóstico tiene NaN"
    assert (forecast > 0).all(), f"{isla}: el pronóstico tiene valores no positivos"

print("\nTodos los modelos cargan y pronostican sin errores.")
```

**Salida:**

```
El Hierro (sarima) — último real: 58.5 € | pronóstico +3m: [55.4, 56.2, 52.6]
Fuerteventura (sarima) — último real: 97.7 € | pronóstico +3m: [104.4, 111.7, 96.5]
Gran Canaria (sarima) — último real: 149.6 € | pronóstico +3m: [160.2, 153.4, 125.0]
La Gomera (lightgbm) — último real: 128.1 € | pronóstico +3m: [125.0, 115.9, 70.8]
La Palma (lightgbm) — último real: 67.9 € | pronóstico +3m: [65.8, 63.8, 56.4]
Lanzarote (sarima) — último real: 126.2 € | pronóstico +3m: [136.0, 137.1, 123.1]
Tenerife (sarima) — último real: 142.8 € | pronóstico +3m: [149.4, 145.7, 129.0]

Todos los modelos cargan y pronostican sin errores.
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```

---

## 04_business_report.ipynb

### Celda 1

**Código:**

```python
import json
import pickle
from pathlib import Path

import pandas as pd

from src.business import demand_alert_level, revpar_gap_vs_last_year
from src.data import load_harmonized_series
from src.model import forecast_recursive_lightgbm

HORIZON = 3

with open("models/registry.json") as f:
    registry = json.load(f)

df = load_harmonized_series()
rows = []

for isla, info in registry.items():
    grupo = df[df["isla"] == isla].sort_values("fecha").reset_index(drop=True)
    with open(info["path"], "rb") as f:
        fitted = pickle.load(f)

    if info["model_type"] == "sarima":
        forecast_revpar = fitted.forecast(steps=HORIZON)
        forecast_dates = pd.date_range(
            grupo["fecha"].max() + pd.DateOffset(months=1), periods=HORIZON, freq="MS"
        )
        forecast_revpar = pd.Series(forecast_revpar.values, index=forecast_dates)
    else:
        forecast_revpar = forecast_recursive_lightgbm(fitted, grupo, "revpar_eur", HORIZON)

    last_adr = grupo["adr_eur"].iloc[-1]  # se asume ADR constante al del último mes real disponible

    for fecha_pronostico, revpar_pred in forecast_revpar.items():
        mismo_mes_año_pasado = grupo[grupo["fecha"] == fecha_pronostico - pd.DateOffset(years=1)]
        if mismo_mes_año_pasado.empty:
            continue

        last_year_occupancy = mismo_mes_año_pasado["grado_ocupacion"].iloc[0]
        last_year_revpar = mismo_mes_año_pasado["revpar_eur"].iloc[0]
        forecast_occupancy = min(revpar_pred / last_adr, 1.0)

        opportunity = revpar_gap_vs_last_year(forecast_occupancy, last_year_occupancy, last_adr)
        yoy_growth = (revpar_pred - last_year_revpar) / last_year_revpar
        alerta = demand_alert_level(yoy_growth)

        rows.append(
            {
                "isla": isla,
                "modelo": info["model_type"],
                "fecha_pronostico": fecha_pronostico.date(),
                "revpar_pronosticado_eur": round(revpar_pred, 1),
                "revpar_año_pasado_eur": round(last_year_revpar, 1),
                "yoy_growth_pct": round(yoy_growth * 100, 1),
                "nivel_alerta": alerta,
                "gap_revpar_eur": round(opportunity.gap_eur, 1),
                "adr_aumento_necesario_pct": round(opportunity.adr_increase_needed_pct * 100, 1),
            }
        )

informe = pd.DataFrame(rows)
informe
```

**Salida:**

```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
  return get_prediction_index(
/home/kegare/Documentos/repositorios/Inari/king-crimson/.venv/lib/python3.14/site-packages/statsmodels/tsa/base/tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
```
```
isla    modelo fecha_pronostico  revpar_pronosticado_eur  \
0       El Hierro    sarima       2026-02-01                     55.4   
1       El Hierro    sarima       2026-03-01                     56.2   
2       El Hierro    sarima       2026-04-01                     52.6   
3   Fuerteventura    sarima       2026-02-01                    104.4   
4   Fuerteventura    sarima       2026-03-01                    111.7   
5   Fuerteventura    sarima       2026-04-01                     96.5   
6    Gran Canaria    sarima       2026-02-01                    160.2   
7    Gran Canaria    sarima       2026-03-01                    153.4   
8    Gran Canaria    sarima       2026-04-01                    125.0   
9       La Gomera  lightgbm       2026-02-01                    125.0   
10      La Gomera  lightgbm       2026-03-01                    115.9   
11      La Gomera  lightgbm       2026-04-01                     70.8   
12       La Palma  lightgbm       2026-02-01                     65.8   
13       La Palma  lightgbm       2026-03-01                     63.8   
14       La Palma  lightgbm       2026-04-01                     56.4   
15      Lanzarote    sarima       2026-02-01                    136.0   
16      Lanzarote    sarima       2026-03-01                    137.1   
17      Lanzarote    sarima       2026-04-01                    123.1   
18       Tenerife    sarima       2026-02-01                    149.4   
19       Tenerife    sarima       2026-03-01                    145.7   
20       Tenerife    sarima       2026-04-01                    129.0   

    revpar_año_pasado_eur  yoy_growth_pct  nivel_alerta  gap_revpar_eur  \
0                    51.5             7.6   crecimiento            -2.1   
1                    55.8             0.8       estable             0.2   
2                    55.4            -5.1  alerta_media             5.4   
3                   101.7             2.7       estable             2.1   
4                   106.9             4.5       estable            -3.2   
5                    91.5             5.5   crecimiento            -3.9   
6                   150.8             6.2   crecimiento            -7.7   
7                   142.5             7.7   crecimiento            -7.2   
8                   113.0            10.6   crecimiento             1.4   
9                   122.7             1.9       estable             5.7   
10                  112.6             2.9       estable             7.9   
11                   80.0           -11.5  alerta_media            19.2   
12                   62.0             6.1   crecimiento            -0.6   
13                   61.0             4.5       estable             1.6   
14                   46.2            22.2   crecimiento            -3.6   
15                  134.5             1.1       estable            -4.6   
16                  131.1             4.6       estable            -5.6   
17                  120.7             2.0       estable            -3.4   
18                  135.7            10.2   crecimiento             2.6   
19                  130.2            11.9   crecimiento             0.3   
20                  114.6            12.5   crecimiento             1.3   

    adr_aumento_necesario_pct  
0                        -3.9  
1                         0.4  
2                        10.3  
3                         2.0  
4                        -2.8  
5                        -4.1  
6                        -4.8  
7                        -4.7  
8                         1.1  
9                         4.6  
10                        6.8  
11                       27.1  
12                       -1.0  
13                        2.5  
14                       -6.4  
15                       -3.4  
16                       -4.1  
17                       -2.8  
18                        1.7  
19                        0.2  
20                        1.0
```

### Celda 2

**Código:**

```python
Path("reports").mkdir(exist_ok=True)
informe.to_csv("reports/informe_negocio.csv", index=False)

print("=== Resumen por isla (nivel de alerta más frecuente en el horizonte de 3 meses) ===")
resumen = informe.groupby("isla").agg(
    nivel_alerta_predominante=("nivel_alerta", lambda s: s.mode()[0]),
    yoy_growth_medio_pct=("yoy_growth_pct", "mean"),
    gap_revpar_medio_eur=("gap_revpar_eur", "mean"),
).round(1).sort_values("yoy_growth_medio_pct")
print(resumen)
```

**Salida:**

```
=== Resumen por isla (nivel de alerta más frecuente en el horizonte de 3 meses) ===
              nivel_alerta_predominante  yoy_growth_medio_pct  \
isla                                                            
La Gomera                       estable                  -2.2   
El Hierro                  alerta_media                   1.1   
Lanzarote                       estable                   2.6   
Fuerteventura                   estable                   4.2   
Gran Canaria                crecimiento                   8.2   
La Palma                    crecimiento                  10.9   
Tenerife                    crecimiento                  11.5   

               gap_revpar_medio_eur  
isla                                 
La Gomera                      10.9  
El Hierro                       1.2  
Lanzarote                      -4.5  
Fuerteventura                  -1.7  
Gran Canaria                   -4.5  
La Palma                       -0.9  
Tenerife                        1.4
```

### Celda 3

**Código:**

```python
import matplotlib.pyplot as plt

color_por_alerta = {
    "alerta_alta": "#c0392b",
    "alerta_media": "#e67e22",
    "estable": "#7f8c8d",
    "crecimiento": "#27ae60",
}

fig, ax = plt.subplots(figsize=(9, 5))
colores = resumen["nivel_alerta_predominante"].map(color_por_alerta)
ax.barh(resumen.index, resumen["yoy_growth_medio_pct"], color=colores)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Crecimiento interanual medio de RevPAR pronosticado (%, próximos 3 meses)")
ax.set_title("Alerta de demanda por isla — modelo ganador por isla")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_por_alerta.values()]
ax.legend(handles, color_por_alerta.keys(), title="Nivel de alerta", loc="lower right")

plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/04_alerta_demanda_por_isla.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Salida:**

`<Figure size 900x500 with 1 Axes>`

---

## Validación de resultados (¿son sensatos?)

Ejecución completa verificada el 2026-07-17. Resumen de comprobaciones:

| Comprobación | Resultado | ¿Sensato? |
|---|---|---|
| Tests unitarios (`pytest tests/ -v`) | 35/35 passed | ✅ |
| Cubo ISTAC crudo | 596.736 filas → 1.414 filas armonizadas (7 islas × 202 meses) | ✅ |
| Rango temporal | 2009-01 → 2026-01, sin huecos por isla | ✅ |
| RevPAR mensual | 7,5 € – 150,8 € (Gran Canaria/Tenerife líderes ~72–77 € de media) | ✅ |
| Ocupación derivada | 13 % – 97 %, 0 filas fuera de [0, 1] | ✅ |
| Ruptura COVID | Caída visible 2020-2021 en gráficos EDA | ✅ |
| Temporada alta real | Picos en ago/nov/feb/mar — distinto de la intuición inicial (jul/dic no son pico) | ✅ coherente con datos |
| Backtesting | 357 folds (51/isla × 7), horizonte 3 meses | ✅ |
| MAPE global | Naive 29,8 % > SARIMA 18,3 % ≈ LightGBM 18,0 % | ✅ modelos aportan vs baseline |
| Ganadores por isla | SARIMA 5/7, LightGBM 2/7 (La Gomera, La Palma — islas más volátiles) | ✅ |
| Modelos serializados | 7 `.pkl` + `registry.json` | ✅ |
| Informe de negocio | 21 filas (7 islas × 3 meses), alertas y gaps en € razonables | ✅ |
| App Streamlit | Smoke test OK (Gran Canaria SARIMA: ocupación ~76 %, RevPAR ~129 €) | ✅ |

**Nota sobre el README:** `jupyter nbconvert --execute` falla si el kernel arranca en `notebooks/` (rutas relativas a `data/`). Para re-ejecutar desde cero usar `python scripts/execute_and_report.py`, que fija el cwd en la raíz del repo.

---

## Qué falta para un 10/10 de cara a un reclutador

El proyecto ya está muy por encima de la media de portfolios (dato real oficial, pipeline completo, tests, app, traducción a negocio). Para cerrarlo como referencia de portfolio:

### Impacto visual e inmediato (alto ROI)

1. **Desplegar la app** en Streamlit Community Cloud o Hugging Face Spaces y poner el enlace en el README — un reclutador no va a clonar 67 MB de CSV ni esperar 15 min de backtesting.
2. **Capturas/GIF de la demo** en el README (selector de isla, pronóstico con intervalo, alerta de demanda).
3. **Commit/push de las figuras** en `reports/figures/` (ahora existen localmente pero pueden estar gitignored) — el README ya las referencia.

### Narrativa y credibilidad

4. **Página de portfolio o post corto** (LinkedIn/dev.to): problema de negocio → fuente de datos → decisión de modelado → resultado en €. El README es bueno; falta la versión “storytelling” de 2 minutos de lectura.
5. **MAPE excluyendo COVID** como métrica secundaria — demuestra rigor metodológico sin esconder la ruptura (ya documentada en ROADMAP).
6. **Intervalos de confianza en el backtest** — hoy hay MAE/RMSE/MAPE; añadir cobertura del intervalo SARIMA reforzaría la parte de “forecasting serio”.

### Ingeniería de producto

7. **Script `make run` o `Makefile`** con targets `test`, `notebooks`, `app` — reduce fricción de onboarding (el fix de cwd ya está en `scripts/execute_and_report.py`; integrarlo en el README).
8. **CI con badge** en el README (`pytest` ya corre en GitHub Actions; añadir el shield verde).
9. **Pin de versiones** (`requirements.txt` con `==` o `requirements.lock`) — reproducibilidad explícita para entrevistas técnicas.

### Diferenciación técnica (nice-to-have)

10. **Contraste con INE** (EOH provincial) — estaba en el plan original; añadir una celda/notebook corto mostrando si Canarias se desacopla de la media nacional.
11. **Feature importance de LightGBM** por isla — explica por qué gana en La Gomera/La Palma.
12. **Docker opcional** solo para la app (no para el pipeline entero) — útil si el despliegue en cloud da problemas con rutas locales.

### Lo que NO haría falta tocar

- Más modelos (Prophet, LSTM…) — el trío naive/SARIMA/LightGBM ya cuenta una historia clara.
- Más tests triviales — 35 tests bien elegidos son suficientes.
- Spark/Docker para todo el pipeline — el README acierta en mantenerlo simple.

**Veredicto:** hoy es un **8–8,5/10** para reclutador técnico de DS (contenido sólido, original, ejecutable). El salto a **10/10** es casi todo presentación y accesibilidad: deploy + capturas + enlace en portfolio + una métrica/figura extra de rigor (MAPE sin COVID o intervalos).

## Validación de resultados (¿son sensatos?)

Ejecución completa verificada el 2026-07-17. Resumen de comprobaciones:

| Comprobación | Resultado | ¿Sensato? |
|---|---|---|
| Tests unitarios (`pytest tests/ -v`) | 35/35 passed | ✅ |
| Cubo ISTAC crudo | 596.736 filas → 1.414 filas armonizadas (7 islas × 202 meses) | ✅ |
| Rango temporal | 2009-01 → 2026-01, sin huecos por isla | ✅ |
| RevPAR mensual | 7,5 € – 150,8 € (Gran Canaria/Tenerife líderes ~72–77 € de media) | ✅ |
| Ocupación derivada | 13 % – 97 %, 0 filas fuera de [0, 1] | ✅ |
| Ruptura COVID | Caída visible 2020-2021 en gráficos EDA | ✅ |
| Temporada alta real | Picos en ago/nov/feb/mar — distinto de la intuición inicial (jul/dic no son pico) | ✅ coherente con datos |
| Backtesting | 357 folds (51/isla × 7), horizonte 3 meses | ✅ |
| MAPE global | Naive 29,8 % > SARIMA 18,3 % ≈ LightGBM 18,0 % | ✅ modelos aportan vs baseline |
| Ganadores por isla | SARIMA 5/7, LightGBM 2/7 (La Gomera, La Palma — islas más volátiles) | ✅ |
| Modelos serializados | 7 `.pkl` + `registry.json` | ✅ |
| Informe de negocio | 21 filas (7 islas × 3 meses), alertas y gaps en € razonables | ✅ |
| App Streamlit | Smoke test OK (Gran Canaria SARIMA: ocupación ~76 %, RevPAR ~129 €) | ✅ |

**Nota sobre el README:** `jupyter nbconvert --execute` falla si el kernel arranca en `notebooks/` (rutas relativas a `data/`). Para re-ejecutar desde cero usar `python scripts/execute_and_report.py`, que fija el cwd en la raíz del repo.

---

## Qué falta para un 10/10 de cara a un reclutador

El proyecto ya está muy por encima de la media de portfolios (dato real oficial, pipeline completo, tests, app, traducción a negocio). Para cerrarlo como referencia de portfolio:

### Impacto visual e inmediato (alto ROI)

1. **Desplegar la app** en Streamlit Community Cloud o Hugging Face Spaces y poner el enlace en el README — un reclutador no va a clonar 67 MB de CSV ni esperar 15 min de backtesting.
2. **Capturas/GIF de la demo** en el README (selector de isla, pronóstico con intervalo, alerta de demanda).
3. **Commit/push de las figuras** en `reports/figures/` (ahora existen localmente pero pueden estar gitignored) — el README ya las referencia.

### Narrativa y credibilidad

4. **Página de portfolio o post corto** (LinkedIn/dev.to): problema de negocio → fuente de datos → decisión de modelado → resultado en €. El README es bueno; falta la versión “storytelling” de 2 minutos de lectura.
5. **MAPE excluyendo COVID** como métrica secundaria — demuestra rigor metodológico sin esconder la ruptura (ya documentada en ROADMAP).
6. **Intervalos de confianza en el backtest** — hoy hay MAE/RMSE/MAPE; añadir cobertura del intervalo SARIMA reforzaría la parte de “forecasting serio”.

### Ingeniería de producto

7. **Script `make run` o `Makefile`** con targets `test`, `notebooks`, `app` — reduce fricción de onboarding (el fix de cwd ya está en `scripts/execute_and_report.py`; integrarlo en el README).
8. **CI con badge** en el README (`pytest` ya corre en GitHub Actions; añadir el shield verde).
9. **Pin de versiones** (`requirements.txt` con `==` o `requirements.lock`) — reproducibilidad explícita para entrevistas técnicas.

### Diferenciación técnica (nice-to-have)

10. **Contraste con INE** (EOH provincial) — estaba en el plan original; añadir una celda/notebook corto mostrando si Canarias se desacopla de la media nacional.
11. **Feature importance de LightGBM** por isla — explica por qué gana en La Gomera/La Palma.
12. **Docker opcional** solo para la app (no para el pipeline entero) — útil si el despliegue en cloud da problemas con rutas locales.

### Lo que NO haría falta tocar

- Más modelos (Prophet, LSTM…) — el trío naive/SARIMA/LightGBM ya cuenta una historia clara.
- Más tests triviales — 35 tests bien elegidos son suficientes.
- Spark/Docker para todo el pipeline — el README acierta en mantenerlo simple.

**Veredicto:** hoy es un **8–8,5/10** para reclutador técnico de DS (contenido sólido, original, ejecutable). El salto a **10/10** es casi todo presentación y accesibilidad: deploy + capturas + enlace en portfolio + una métrica/figura extra de rigor (MAPE sin COVID o intervalos).
