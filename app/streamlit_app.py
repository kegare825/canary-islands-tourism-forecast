"""Demo interactiva de King Crimson.

Elige isla + horizonte, carga el modelo real entrenado en
`notebooks/03_modeling.ipynb` (SARIMA o LightGBM según cuál ganó el
backtesting para esa isla — ver `models/registry.json`), pronostica RevPAR
y muestra la oportunidad de negocio si la ocupación pronosticada cae
respecto al mismo mes del año pasado.
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` pone el directorio del script (app/) en
# sys.path, no la raíz del repo — sin esto, `from src...` falla con
# ModuleNotFoundError salvo que se ejecute con `cwd` == raíz y PYTHONPATH
# puesto a mano. Se fuerza aquí para que funcione sin configuración extra,
# desde cualquier directorio de partida.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.business import demand_alert_level, revpar_gap_vs_last_year
from src.data import ISLANDS, load_harmonized_series
from src.model import forecast_recursive_lightgbm

st.set_page_config(page_title="King Crimson", page_icon="⏳")

st.title("⏳ King Crimson")
st.caption("Pronóstico de demanda turística por isla + oportunidad de RevPAR")

MODEL_DIR = Path("models")
REGISTRY_PATH = MODEL_DIR / "registry.json"


@st.cache_resource
def cargar_registro() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def cargar_historial() -> pd.DataFrame:
    return load_harmonized_series()


if not REGISTRY_PATH.exists():
    st.warning(
        "No hay modelos entrenados todavía en `models/`. Ejecuta "
        "`notebooks/01_eda.ipynb` a `03_modeling.ipynb` en orden y vuelve a "
        "cargar esta página."
    )
    st.stop()

registry = cargar_registro()
df = cargar_historial()

with st.sidebar:
    st.header("Parámetros")
    isla = st.selectbox("Isla", [i for i in ISLANDS if i in registry] or ISLANDS)
    horizonte = st.slider("Horizonte (meses)", min_value=1, max_value=6, value=3)

if isla not in registry:
    st.warning(f"No hay modelo entrenado para **{isla}** en `models/registry.json`.")
    st.stop()

info = registry[isla]
historial_isla = df[df["isla"] == isla].sort_values("fecha").reset_index(drop=True)
modelo = cargar_modelo(info["path"])

if info["model_type"] == "sarima":
    pred = modelo.forecast(steps=horizonte)
    fechas_pronostico = pd.date_range(
        historial_isla["fecha"].max() + pd.DateOffset(months=1), periods=horizonte, freq="MS"
    )
    forecast_revpar = pd.Series(pred.values, index=fechas_pronostico)
else:
    forecast_revpar = forecast_recursive_lightgbm(modelo, historial_isla, "revpar_eur", horizonte)

ultimo_mes_real = historial_isla["fecha"].max().date()
ultimo_adr_real = float(historial_isla["adr_eur"].iloc[-1])
fecha_objetivo = forecast_revpar.index[-1]
revpar_objetivo = float(forecast_revpar.iloc[-1])

mismo_mes_ano_pasado = historial_isla[historial_isla["fecha"] == fecha_objetivo - pd.DateOffset(years=1)]
hay_dato_ano_pasado = not mismo_mes_ano_pasado.empty
ocupacion_ano_pasado_real = float(mismo_mes_ano_pasado["grado_ocupacion"].iloc[0]) if hay_dato_ano_pasado else 0.65
revpar_ano_pasado_real = float(mismo_mes_ano_pasado["revpar_eur"].iloc[0]) if hay_dato_ano_pasado else None

with st.sidebar:
    st.caption(f"Modelo: **{info['model_type']}** · último dato real: {ultimo_mes_real}")
    # key atado a `isla` a propósito: si no, Streamlit conserva el valor anterior
    # del widget entre reruns y el autocompletado por isla deja de funcionar.
    adr_actual = st.number_input(
        "ADR asumido para el pronóstico (€/noche)",
        min_value=0.0,
        value=round(ultimo_adr_real, 1),
        step=5.0,
        key=f"adr_{isla}",
        help="Autocompletado con el último ADR real. El modelo pronostica RevPAR; "
        "la ocupación implícita se deriva de RevPAR / ADR, así que este valor "
        "cambia la ocupación pronosticada que se muestra.",
    )
    ocupacion_ano_pasado = st.slider(
        "Ocupación mismo mes año pasado (%)",
        0, 100,
        int(round(ocupacion_ano_pasado_real * 100)),
        key=f"occ_{isla}",
        help="Autocompletado con el dato histórico real de ISTAC; "
        "muévelo para simular otro escenario de partida.",
    ) / 100

st.subheader(f"Pronóstico de RevPAR — {isla}")
tabla_pronostico = pd.DataFrame(
    {
        "fecha": forecast_revpar.index.date,
        "revpar_pronosticado_eur": forecast_revpar.values.round(1),
        "ocupacion_implícita_%": (forecast_revpar.values / adr_actual * 100).round(1) if adr_actual > 0 else None,
    }
)
st.dataframe(tabla_pronostico, hide_index=True, width="stretch")
st.line_chart(historial_isla.set_index("fecha")["revpar_eur"].tail(36))

if not hay_dato_ano_pasado:
    st.info(
        "No hay dato histórico real de hace un año para el mes objetivo "
        f"({fecha_objetivo.date()}) — se usa un 65% de ocupación como referencia por defecto."
    )

if st.button("Calcular oportunidad de RevPAR", type="primary"):
    ocupacion_pronosticada = min(revpar_objetivo / adr_actual, 1.0) if adr_actual > 0 else 0.0

    opportunity = revpar_gap_vs_last_year(
        forecast_occupancy=ocupacion_pronosticada,
        last_year_occupancy=ocupacion_ano_pasado,
        adr_eur=adr_actual,
    )
    yoy_growth = (
        (revpar_objetivo - revpar_ano_pasado_real) / revpar_ano_pasado_real
        if revpar_ano_pasado_real
        else 0.0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Ocupación pronosticada ({fecha_objetivo.date()})", f"{ocupacion_pronosticada:.0%}")
    col2.metric("RevPAR pronosticado", f"{opportunity.forecast_revpar_eur:,.1f} €")
    col3.metric("Brecha de RevPAR vs año pasado", f"{opportunity.gap_eur:,.1f} €")

    st.write(f"Crecimiento interanual de RevPAR pronosticado: **{yoy_growth:+.1%}**")
    st.write(f"Nivel de alerta de demanda: **{demand_alert_level(yoy_growth)}**")
    if opportunity.gap_eur > 0:
        st.write(
            f"Para no perder esos {opportunity.gap_eur:,.1f} € de RevPAR, el ADR tendría "
            f"que subir un **{opportunity.adr_increase_needed_pct:.1%}** respecto al actual."
        )
    else:
        st.success("El pronóstico ya supera al mismo mes del año pasado — no hay brecha que compensar.")
