# Roadmap — King Crimson

A diferencia de `bites-the-dust`, aquí el día 0-1 es más largo porque hay que
descubrir y armonizar el dato real (no es un CSV limpio de Kaggle). El resto
del pipeline corre igual en Colab, sin Docker ni Spark.

- [ ] **Día 0-1 — Descubrimiento de datos**: descargar `C00065A_000003` (y
      opcionalmente `_000040`/`_000060`) del ISTAC en CSV/JSON, inspeccionar
      las columnas reales del export, y escribir el parser de armonización en
      `notebooks/01_eda.ipynb` hacia el esquema de `data/README.md`.
      Opcional: localizar el `IdSERIE` del INE para la serie de contraste nacional.
- [ ] **Día 1-2 — EDA**: series por isla, estacionalidad visual, comparación
      de picos (¿temporada alta real en Canarias son enero-marzo y julio-agosto,
      como asume `HIGH_SEASON_MONTHS` en `src/features.py`, o hay que ajustarlo
      con el dato real?).
- [ ] **Día 2-3 — Descomposición y features** (`02_decomposition_features.ipynb`):
      descomposición estacional (trend/seasonal/residual) por isla, aplicar
      `src/features.py` (rezagos, medias móviles, YoY).
- [ ] **Día 3-5 — Modelado** (`03_modeling.ipynb`): baseline naive estacional
      obligatorio primero, luego SARIMA y LightGBM. Validar con
      `expanding_window_splits` — **nunca** un split aleatorio en series
      temporales, es fuga de información directa.
- [ ] **Día 5 — Informe de negocio** (`04_business_report.ipynb`): aplicar
      `src/business.py` para traducir el pronóstico a oportunidad de RevPAR
      por isla, exportar gráficos a `reports/figures/`.
- [ ] **Día 5-6 — App de demo**: conectar `app/streamlit_app.py` al modelo real.
- [ ] **Día 6-7 — Cuando llegue el portátil**: capturas de pantalla, desplegar
      en Streamlit Community Cloud/HF Spaces, enlazar desde el README.

## Riesgo conocido a vigilar

La serie mensual más larga confirmada (`C00065A_000003`, desde 2009) da ~16
años de histórico — suficiente para SARIMA, pero conviene revisar en el EDA
si hay rupturas estructurales claras (2020-2021, covid) que haya que tratar
aparte en el backtesting, no promediarlas sin más con el resto de la serie.
