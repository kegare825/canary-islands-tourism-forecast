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

🚧 **En preparación.** Estructura y código base listos; pendiente descargar los datos
reales y ejecutar el pipeline en cuanto haya cómputo disponible (ver `docs/ROADMAP.md`).
Corre entero en Google Colab / Kaggle Notebooks — sin Docker ni Spark.

## Objetivo

1. Pronosticar pernoctaciones/ocupación mensual por isla con un horizonte de 3-6 meses.
2. Comparar un baseline de series temporales (naive estacional) contra SARIMA y un modelo
   de gradient boosting con features de calendario y rezagos.
3. Traducir el pronóstico a **oportunidad de RevPAR** — si se sabe con antelación que
   viene un mes flojo de demanda, hay margen de reacción en pricing (mismo concepto de
   ADR × Ocupación que ya se usa en ADS/ADMI).
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
├── models/                    Modelos entrenados — gitignored
├── reports/figures/            Gráficos para el README/portfolio
├── tests/                      Tests de business.py — no necesitan datos ni modelo
└── docs/
    └── ROADMAP.md              Plan de preparación
```

## Stack

Python · pandas · statsmodels (SARIMA) · LightGBM · scikit-learn · matplotlib/seaborn ·
Streamlit · pytest. Sin infraestructura pesada.

## Cómo correrlo (cuando haya datos descargados y armonizados)

```bash
pip install -r requirements.txt
# 1. Descargar y armonizar los datos (ver data/README.md) en data/processed/
# 2. Ejecutar notebooks/ en orden, o:
python -m src.model
streamlit run app/streamlit_app.py
```

## Roadmap

Ver `docs/ROADMAP.md`.
