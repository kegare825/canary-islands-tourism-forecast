# Roadmap — King Crimson

A diferencia de `bites-the-dust`, aquí el día 0-1 fue más largo porque hubo que
descubrir y armonizar el dato real (no es un CSV limpio de Kaggle). El resto
del pipeline corre en un venv normal, sin Docker ni Spark.

- [x] **Descubrimiento de datos**: descargado `C00065A_000003` del ISTAC
      (67 MB, 596.736 filas crudas). Columnas reales del export documentadas en
      `notebooks/01_eda.ipynb`: `TERRITORIO#es` mezcla islas/municipios/total
      regional en la misma columna, `MEDIDAS#es` trae 8 medidas distintas (solo
      2 usadas: ADR y RevPAR), `TIME_PERIOD_CODE` mezcla periodos anuales y
      mensuales. No hay medida directa de ocupación en este cubo — se deriva
      como `revpar_eur / adr_eur`, consistente con la fórmula que ya asume
      `src/business.py`. *(No se llegó a necesitar el `IdSERIE` del INE — el
      cubo ISTAC solo ya cubre ADR+RevPAR con suficiente profundidad histórica.)*
- [x] **EDA** (`notebooks/01_eda.ipynb`): 205 meses por isla (2009-2026), sin
      huecos, ruptura COVID claramente visible en 2020-2021. Los meses de
      temporada alta real (agosto, noviembre, febrero, marzo, septiembre, enero)
      **no coinciden del todo** con la intuición inicial de `HIGH_SEASON_MONTHS`
      — julio y diciembre resultan ser temporada media, no alta; noviembre sí es
      un pico real que no se había anticipado. Corregido en `src/features.py`.
      Las islas menores (La Palma, La Gomera, El Hierro) tienen menor ocupación
      media Y mayor variabilidad relativa que las islas grandes.
- [x] **Descomposición y features** (`02_decomposition_features.ipynb`):
      descomposición aditiva (trend/seasonal) por isla — la ruptura COVID se ve
      con nitidez en la tendencia de las 7 islas. Bug real encontrado y arreglado
      aquí: con pandas 3.0, `groupby(...).apply(...)` deja de propagar la columna
      de agrupación por defecto, lo que hacía desaparecer `isla` del resultado de
      `build_features` — arreglado con `groupby` + `concat` explícito (ver
      comentario en `src/features.py`).
- [x] **Modelado** (`03_modeling.ipynb`): 357 folds de ventana expansiva (51
      folds/isla × 7 islas), horizonte 3 meses. Naive estacional MAPE 29.8%,
      SARIMA 18.3%, LightGBM 18.0% — ambos le ganan claramente al naive. SARIMA
      gana en 5/7 islas, LightGBM en las 2 islas más pequeñas/volátiles. Modelo
      final por isla entrenado con el 100% del histórico y serializado en
      `models/` (`registry.json` mapea isla → tipo de modelo → ruta del `.pkl`).
- [x] **Informe de negocio** (`04_business_report.ipynb`): `src/business.py`
      aplicado sobre el pronóstico real de cada modelo ganador, comparado contra
      el mismo mes real del año pasado. Exportado a `reports/informe_negocio.csv`
      y `reports/figures/04_alerta_demanda_por_isla.png`.
- [x] **App de demo**: `app/streamlit_app.py` conectada a `models/registry.json`
      (ya no hay placeholder). Bug real encontrado y arreglado: `streamlit run
      app/streamlit_app.py` no añade la raíz del repo a `sys.path` (solo el
      directorio del script), así que `from src...` fallaba con
      `ModuleNotFoundError` — nunca se había ejecutado la app hasta ahora, por
      lo que este bug llevaba desde la primera versión sin detectarse.
      Verificado con `streamlit.testing.v1.AppTest` + navegador real (SARIMA y
      LightGBM, cambio de isla, botón de cálculo).
- [x] **Capturas de demo** en `reports/figures/demo/` (composite, sidebar,
      pronóstico con IC 95 %, alerta RevPAR, GIF por isla) — referenciadas en README.
- [x] **MAPE ex-COVID + cobertura IC SARIMA** en backtesting (`src/model.py`,
      `scripts/compute_backtest_summary.py`): MAPE global 18,3 % → ex-COVID 13,2 %;
      cobertura intervalo 95 % ~85 %.
- [x] **Contraste INE** (`notebooks/05_ine_contrast.ipynb`, `src/ine.py`): tabla
      EOH 67190, figura `05_ine_contrast_canarias.png`.
- [x] **Feature importance LightGBM** para La Gomera/La Palma
      (`reports/figures/06_lightgbm_feature_importance.png`).
- [x] **Makefile + Docker** (`Makefile`, `docker/Dockerfile`) para onboarding y
      despliegue opcional de la app.
- [x] **Post de portfolio** (`docs/portfolio_post.md`).
- [x] **Despliegue Streamlit** en Community Cloud + enlace en README:
      [canary-islands-tourism-forecast.streamlit.app](https://canary-islands-tourism-forecast.streamlit.app/)
      (repo público: `kegare825/canary-islands-tourism-forecast`).
- [x] **Ablation FRONTUR** (`src/frontur.py`, `scripts/frontur_ablation.py`):
      LightGBM + turistas proxy (lags/rolling) vs baseline — ver `docs/frontur_ablation.md`.

## Riesgos que se confirmaron (y cómo se trataron)

- **Ruptura COVID dentro del backtest**: se dejó a propósito dentro de los 357
  folds en vez de excluirla — es honesto (así se comportaría el modelo en
  producción si viene otra ruptura), pero infla el MAPE global reportado más
  arriba. **Resuelto:** se reporta también MAPE ex-COVID (mar–jun 2020–2021
  excluidos por fold) en README y `data/processed/model_eval_summary.csv`.
- **LightGBM recursivo**: al pronosticar >1 mes, LightGBM no se extrapola solo
  como SARIMA — hay que recalcular sus features (rezagos/medias) con sus propias
  predicciones anteriores (`src/model.forecast_recursive_lightgbm`). El error se
  acumula paso a paso; es una limitación conocida y documentada, no un bug.
