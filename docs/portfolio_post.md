# Canary Islands Tourism Forecast — post de portfolio (2 minutos de lectura)

**TL;DR:** Pronostico demanda turística mensual por isla en Canarias con datos abiertos oficiales (ISTAC), comparo naive estacional vs SARIMA vs LightGBM con backtesting honesto (357 folds), y traduzco el resultado a euros de oportunidad RevPAR para equipos de revenue.

---

## El problema de negocio

En hotelería, **saber con antelación que viene un mes flojo** da margen para actuar: ajustar ADR, campañas, staffing. El indicador que une pricing y demanda es **RevPAR** (ingresos por habitación disponible), que depende de ocupación × tarifa.

La pregunta no es solo “¿cuánto RevPAR habrá?”, sino **“¿voy peor que el mismo mes del año pasado y cuántos euros pierdo si no reacciono?”**

---

## Por qué estos datos (y no Kaggle)

Usé el cubo ISTAC `C00065A_000003`: ADR y RevPAR mensuales por isla desde 2009. Es dato **oficial, geográfico y poco común en portfolios** — hay que armarizar el export SDMX/cubo (596k filas crudas → 1.414 filas útiles).

Complemento con **INE EOH** (tabla 67190) para contrastar pernoctaciones provinciales/nacionales vs la señal RevPAR del ISTAC.

---

## Decisiones de modelado

1. **Baseline obligatorio:** naive estacional (repetir hace 12 meses). Si un modelo no le gana, no aporta.
2. **SARIMA(1,1,1)(1,1,1,12):** referencia clásica para estacionalidad mensual.
3. **LightGBM** con calendario, rezagos y medias móviles: captura no linealidades; gana en islas pequeñas y volátiles.

**Validación:** ventana expansiva, horizonte 3 meses, **357 folds** — nunca split aleatorio (fuga de futuro).

Dejamos COVID **dentro** del backtest (honesto para producción), pero reportamos también **MAPE ex-COVID** e **intervalos de confianza SARIMA** (cobertura ~95 %).

---

## Resultados (RevPAR, horizonte 3 meses)

| Modelo | MAPE global | MAPE ex-COVID | Notas |
|---|---|---|---|
| Naive estacional | 29,8 % | 17,0 % | baseline |
| SARIMA | **18,3 %** | **13,2 %** | gana en 5/7 islas; IC 95 % ~85 % cobertura |
| LightGBM | 18,0 % | 13,3 % | gana en La Gomera y La Palma |

Hallazgo de EDA que cambió el código: la temporada alta real (ago, nov, feb, mar…) **no coincide** con la intuición inicial — corregí `HIGH_SEASON_MONTHS` con el dato.

---

## De métrica a euros

`src/business.py` convierte pronóstico → **brecha de RevPAR vs año anterior** + nivel de alerta (`alerta_media`, `crecimiento`…). Ejemplo: La Gomera abril −11,5 % YoY → gap ~19 € RevPAR → subir ADR ~27 % para compensar (si la ocupación cae).

Demo Streamlit: isla + horizonte → tabla con **intervalo de confianza** (SARIMA) + botón de oportunidad en €.
[Demo en vivo](https://canary-islands-tourism-forecast.streamlit.app/)

---

## Stack y reproducibilidad

Python · pandas · statsmodels · LightGBM · Streamlit · pytest (35 tests).

```bash
make install test          # venv + tests
make metrics               # backtesting (~15 min)
make reports               # figuras README + INE + feature importance
make app                   # demo local
make docker-build          # contenedor solo para la app
```

---

## Qué demuestra en una entrevista

- Trabajo con **datos abiertos reales** (no dataset limpio).
- **Backtesting temporal** riguroso y baseline sensato.
- Puente **ML → negocio** (RevPAR, alertas, €).
- Producto mínimo (**Streamlit**) + tests + Docker opcional.

Repo: [canary-islands-tourism-forecast](https://github.com/kegare825/canary-islands-tourism-forecast) ·
Demo: [canary-islands-tourism-forecast.streamlit.app](https://canary-islands-tourism-forecast.streamlit.app/)
