# Ablation FRONTUR — exógena proxy + features completas

Backtesting (ventana expansiva, horizonte 3 meses):

| Modelo | Descripción |
|---|---|
| `lightgbm` | mes, trimestre, lags 1/3/6/12, rolling mean/std de RevPAR |
| `lightgbm_exog` | baseline + mismas transforms de **turistas FRONTUR** |
| `sarima` | univariado (referencia) |
| `sarimax_exog` | SARIMAX con columnas exógenas de turistas (lags + rolling) |

SARIMAX solo corre en folds/islas donde **todo el train y test** tienen exógena
sin NaN (El Hierro queda fuera; La Gomera solo folds recientes).

## Fuente exógena (no es C00065A_000003)

`C00065A_000003` = ADR/RevPAR hotelero (target). Llegadas = **FRONTUR** `E16028B`.

| Cubo | Contenido |
|---|---|
| `E16028B_000016` | Turistas principales, 5 islas (2010+) |
| `E16028B_000019` | Turistas La Gomera (2017+) |

## Features baseline (RevPAR)

```text
month
quarter
is_high_season
revpar_eur_lag_1
revpar_eur_lag_3
revpar_eur_lag_6
revpar_eur_lag_12
revpar_eur_rolling_mean_3
revpar_eur_rolling_mean_12
revpar_eur_rolling_std_3
revpar_eur_rolling_std_12
```

## Features exógenas añadidas (turistas)

```text
turistas_lag_1
turistas_lag_3
turistas_lag_6
turistas_lag_12
turistas_rolling_mean_3
turistas_rolling_mean_12
turistas_rolling_std_3
turistas_rolling_std_12
```

## SARIMAX — columnas exógenas

```text
turistas_lag_1
turistas_lag_3
turistas_lag_6
turistas_lag_12
turistas_rolling_mean_3
turistas_rolling_mean_12
turistas_rolling_std_3
turistas_rolling_std_12
```

## Cobertura FRONTUR por isla

| isla | frac_meses_con_dato |
| --- | --- |
| El Hierro | 0.0 |
| Fuerteventura | 0.941 |
| Gran Canaria | 0.941 |
| La Gomera | 0.485 |
| La Palma | 0.941 |
| Lanzarote | 0.941 |
| Tenerife | 0.941 |

## Resultados globales

| model | mae | rmse | mape | mape_ex_covid | interval_coverage |
| --- | --- | --- | --- | --- | --- |
| lightgbm | 8.3552 | 9.3792 | 0.1811 | 0.1352 | nan |
| lightgbm_exog | 8.6384 | 9.6635 | 0.1811 | 0.1394 | nan |
| naive_estacional | 12.6263 | 13.6505 | 0.2983 | 0.17 | nan |
| sarima | 8.0844 | 8.9933 | 0.1832 | 0.1321 | 0.8506 |
| sarimax_exog | 9.4496 | 10.7676 | 0.1916 | 0.1381 | 0.8482 |

## MAPE por isla

| isla | lightgbm | lightgbm_exog | naive_estacional | sarima | sarimax_exog | delta_lgbm_exog | delta_sarimax_exog |
| --- | --- | --- | --- | --- | --- | --- | --- |
| El Hierro | 0.2308 | 0.2308 | 0.2513 | 0.208 | nan | 0.0 | nan |
| Fuerteventura | 0.1339 | 0.1441 | 0.2349 | 0.1352 | 0.1454 | 0.010200000000000015 | 0.010200000000000015 |
| Gran Canaria | 0.1507 | 0.1461 | 0.2745 | 0.1325 | 0.1593 | -0.004599999999999993 | 0.02679999999999999 |
| La Gomera | 0.1746 | 0.1892 | 0.332 | 0.2544 | nan | 0.014600000000000002 | nan |
| La Palma | 0.2698 | 0.2518 | 0.3257 | 0.2957 | 0.3147 | -0.01799999999999996 | 0.01899999999999996 |
| Lanzarote | 0.1777 | 0.1667 | 0.431 | 0.137 | 0.2117 | -0.01100000000000001 | 0.07469999999999999 |
| Tenerife | 0.1299 | 0.1387 | 0.2383 | 0.1198 | 0.127 | 0.008800000000000002 | 0.007199999999999998 |

## Conclusión

- LightGBM baseline: **18.1%** (ex-COVID 13.5%)
- LightGBM + FRONTUR: **18.1%** (ex-COVID 13.9%, Δ -0.0% pp → gana **lightgbm_exog**)
- SARIMA: **18.3%** (ex-COVID 13.2%)
- SARIMAX + FRONTUR: **19.2%** (ex-COVID 13.8%, Δ +0.8% pp → gana **sarima**)

- Islas donde FRONTUR mejora LightGBM: Gran Canaria, La Palma, Lanzarote
- Islas donde FRONTUR mejora SARIMA: ninguna

No se cambia el registry de producción: la mejora global es marginal y mixta por isla.

Re-ejecutar: `make frontur-ablation`.
