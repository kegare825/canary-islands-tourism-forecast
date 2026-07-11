"""Demo interactiva de King Crimson.

Elige isla + horizonte, muestra el pronóstico de demanda y la oportunidad de
RevPAR si la ocupación pronosticada cae respecto al año pasado.
"""

from pathlib import Path

import streamlit as st

from src.business import demand_alert_level, revpar_gap_vs_last_year
from src.data import ISLANDS

st.set_page_config(page_title="King Crimson", page_icon="⏳")

st.title("⏳ King Crimson")
st.caption("Pronóstico de demanda turística por isla + oportunidad de RevPAR")

MODEL_DIR = Path("models")

with st.sidebar:
    st.header("Parámetros")
    isla = st.selectbox("Isla", ISLANDS)
    horizonte = st.slider("Horizonte (meses)", min_value=1, max_value=6, value=3)
    adr_actual = st.number_input("ADR actual (€/noche)", min_value=0.0, value=90.0, step=5.0)
    ocupacion_ano_pasado = st.slider("Ocupación mismo mes año pasado (%)", 0, 100, 65) / 100

if not any(MODEL_DIR.glob("*.pkl")):
    st.warning(
        "No hay modelos entrenados todavía en `models/`. Entrena con "
        "`notebooks/03_modeling.ipynb` y vuelve a cargar esta página."
    )
    st.stop()

# TODO: cargar el modelo real para `isla` y generar el pronóstico de
# ocupación a `horizonte` meses. Se deja explícito el placeholder para que
# quede claro qué falta conectar en cuanto haya modelo entrenado.
st.info("Conectar aquí el pronóstico real del modelo una vez entrenado (ver TODO en el código).")

if st.button("Calcular pronóstico"):
    ocupacion_pronosticada = ocupacion_ano_pasado  # placeholder hasta conectar el modelo real
    yoy_growth = (ocupacion_pronosticada - ocupacion_ano_pasado) / ocupacion_ano_pasado if ocupacion_ano_pasado else 0.0

    opportunity = revpar_gap_vs_last_year(
        forecast_occupancy=ocupacion_pronosticada,
        last_year_occupancy=ocupacion_ano_pasado,
        adr_eur=adr_actual,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Ocupación pronosticada", f"{ocupacion_pronosticada:.0%}")
    col2.metric("RevPAR pronosticado", f"{opportunity.forecast_revpar_eur:,.1f} €")
    col3.metric("Brecha de RevPAR", f"{opportunity.gap_eur:,.1f} €")

    st.write(f"Nivel de alerta de demanda: **{demand_alert_level(yoy_growth)}**")
    if opportunity.gap_eur > 0:
        st.write(
            f"Para no perder esos {opportunity.gap_eur:,.1f} € de RevPAR, el ADR tendría "
            f"que subir un **{opportunity.adr_increase_needed_pct:.1%}** respecto al actual."
        )
