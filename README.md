# King Crimson ⏳

> King Crimson, el Stand de Diavolo, borra un tramo de tiempo entre una acción y su
> resultado — así "ya sabe" cómo termina algo antes de que pase. Este proyecto hace lo
> mismo con la demanda turística: se adelanta al resultado (la ocupación/RevPAR del mes
> que viene) antes de que llegue.

*(Nombre cambiado desde la versión inicial "Weather Report" — ya en uso en otro
repositorio del autor para predicción meteorológica real, para evitar confusión.)*

Proyecto de Data Science de forecasting de series temporales: pernoctaciones/ocupación
turística por isla, usando **datos abiertos oficiales** (ISTAC + INE), no un dataset de
Kaggle. Es la pieza de portfolio más original de las tres — nadie más la va a tener con
exactamente esta fuente ni esta geografía.

## Estado del proyecto

✅ **Ejecutado de extremo a extremo con datos reales.** Cubo ISTAC `C00065A_000003`
descargado y armonizado (205 meses, 2009-2026, 7 islas), backtesting completo
(357 folds de ventana expansiva), modelos entrenados y serializados, e informe de
negocio y app de Streamlit conectados a esos modelos reales — sin placeholders.
Corre entero en un venv normal, sin Docker ni Spark. Detalle de cómo se ejecutó
y qué se encontró en `docs/ROADMAP.md`.

## Objetivo

1. Pronosticar pernoctaciones/ocupación mensual por isla con un horizonte de 3-6 meses.
2. Comparar un baseline de series temporales (naive estacional) contra SARIMA y un modelo
   de gradient boosting con features de calendario y rezagos.
3. Traducir el pronóstico a **oportunidad de RevPAR** — si se sabe con antelación que
   viene un mes flojo de demanda, hay margen de reacción en pricing (mismo concepto de
   ADR × Ocupación que ya se usa en ADS/NDMI).
4. Demo interactiva: elegir isla y horizonte, ver el pronóstico con intervalo de confianza.

## Fuentes de datos (reales, verificadas — ver `data/README.md` para el detalle completo)

- **ISTAC** (Instituto Canario de Estadística) — cubos estadísticos con descarga directa
  en CSV/JSON: pernoctaciones/viajeros por isla, y tarifa media diaria (ADR)/RevPAR/ingresos
  por isla, con series mensuales desde 2009.
- **INE** (Encuesta de Ocupación Hotelera) — serie nacional por provincia (Las Palmas /
  Santa Cruz de Tenerife), vía API Tempus3, para contrastar Canarias contra la tendencia
  nacional.

## Estructura

```
king-crimson/
├── data/
│   ├── raw/                 Descargas originales de ISTAC/INE — gitignored
│   ├── processed/           Series ya armonizadas a formato largo (fecha, isla, indicador)
│   └── README.md            Fuentes exactas, URLs de descarga, esquema esperado
├── notebooks/
│   ├── 01_eda.ipynb                     Carga + armonización + primera exploración
│   ├── 02_decomposition_features.ipynb  Descomposición estacional + features de calendario/rezagos
│   ├── 03_modeling.ipynb                Naive estacional vs SARIMA vs LightGBM, backtesting temporal
│   └── 04_business_report.ipynb         Traducción a € de oportunidad de RevPAR
├── src/
│   ├── data.py               Carga y armonización de las series
│   ├── features.py           Features de calendario, rezagos, medias móviles
│   ├── model.py                Modelos + validación cruzada temporal (sin fuga de futuro)
│   └── business.py             Pronóstico → oportunidad de RevPAR
├── app/
│   └── streamlit_app.py       Demo: isla + horizonte → pronóstico con intervalo
├── models/                    Modelos entrenados (uno por isla) + registry.json — gitignored
├── reports/figures/            Gráficos para el README/portfolio
├── tests/                      35 tests (business/data/features/model) — no necesitan datos reales
└── docs/
    └── ROADMAP.md              Plan de preparación + resultados reales de la ejecución
```

## Stack

Python · pandas · statsmodels (SARIMA) · LightGBM · scikit-learn · matplotlib/seaborn ·
Streamlit · pytest. Sin infraestructura pesada.

## Resultados reales (backtesting, 357 folds de ventana expansiva, horizonte 3 meses)

| Modelo | MAE (€) | RMSE (€) | MAPE |
|---|---|---|---|
| Naive estacional (baseline obligatorio) | 12.6 | 13.7 | 29.8% |
| SARIMA(1,1,1)(1,1,1,12) | 8.1 | 9.0 | **18.3%** |
| LightGBM (calendario + rezagos + medias móviles) | 8.3 | 9.4 | 18.0% |

Ambos modelos le ganan claramente al naive (regla de oro del proyecto: si no le
ganan, no aportan nada). SARIMA gana en 5 de 7 islas grandes/medianas; LightGBM
gana en las 2 islas más pequeñas y volátiles (La Gomera, La Palma) — ver
`reports/figures/03_comparacion_modelos_por_isla.png`. El MAPE incluye a
propósito la ruptura de COVID (2020-2021) dentro del propio backtest, lo que
infla el error de todos los modelos por igual; en periodos normales es menor.

Hallazgo de EDA que sí cambió el código: los meses de temporada alta reales
(agosto, noviembre, febrero, marzo, septiembre, enero) no coinciden del todo con
la intuición inicial de `HIGH_SEASON_MONTHS` (asumía julio y diciembre en vez de
septiembre y noviembre) — corregido en `src/features.py` con el dato real.

![RevPAR y ocupación derivada por isla](reports/figures/01_revpar_ocupacion_por_isla.png)

## Cómo correrlo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Descargar el cubo ISTAC (ver data/README.md) a data/raw/, o usar el ya
#    descargado. Luego correr los notebooks EN ORDEN (cada uno depende del
#    CSV/modelo que genera el anterior):
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_decomposition_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb   # ~15 min: 357 folds de backtesting + entrenamiento final
jupyter nbconvert --to notebook --execute --inplace notebooks/04_business_report.ipynb

# 2. Tests (no dependen de los notebooks anteriores, corren siempre)
pytest tests/ -v

# 3. App — necesita que 03_modeling.ipynb ya haya generado models/registry.json
streamlit run app/streamlit_app.py
```

## Roadmap

Ver `docs/ROADMAP.md`.
