"""Traduce el pronóstico de demanda a oportunidad de negocio en RevPAR.

Mismo concepto que ya usa ADS/ADMI: RevPAR = ADR × Ocupación. Aquí se usa
para responder "si sabemos con 3 meses de antelación que la ocupación va a
caer respecto al mismo mes del año pasado, ¿cuánto RevPAR se puede recuperar
moviendo el ADR?" — funciones puras, testeables sin datos ni modelo.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RevPAROpportunity:
    forecast_revpar_eur: float
    target_revpar_eur: float
    gap_eur: float
    adr_increase_needed_pct: float


def revpar(adr_eur: float, occupancy_rate: float) -> float:
    """RevPAR = ADR × Ocupación. Misma fórmula que en app/services/valuation.py de ADS."""
    if adr_eur < 0:
        raise ValueError("adr_eur debe ser no negativo")
    if not 0 <= occupancy_rate <= 1:
        raise ValueError("occupancy_rate debe estar entre 0 y 1")
    return adr_eur * occupancy_rate


def revpar_gap_vs_last_year(forecast_occupancy: float, last_year_occupancy: float, adr_eur: float) -> RevPAROpportunity:
    """Si el pronóstico de ocupación es menor que el mismo mes del año pasado,
    calcula cuánto RevPAR se pierde y qué subida de ADR compensaría la caída
    de ocupación (para no perder RevPAR total, aun con menos habitaciones vendidas).
    """
    forecast_rp = revpar(adr_eur, forecast_occupancy)
    target_rp = revpar(adr_eur, last_year_occupancy)
    gap = target_rp - forecast_rp

    adr_needed = target_rp / forecast_occupancy if forecast_occupancy > 0 else float("inf")
    adr_increase_pct = (adr_needed - adr_eur) / adr_eur if adr_eur > 0 else float("inf")

    return RevPAROpportunity(
        forecast_revpar_eur=forecast_rp,
        target_revpar_eur=target_rp,
        gap_eur=gap,
        adr_increase_needed_pct=adr_increase_pct,
    )


def demand_alert_level(yoy_growth: float) -> str:
    """Clasifica la caída/subida interanual de demanda en niveles de alerta,
    para saber cuándo vale la pena reaccionar en pricing/marketing.
    """
    if yoy_growth <= -0.15:
        return "alerta_alta"
    if yoy_growth <= -0.05:
        return "alerta_media"
    if yoy_growth < 0.05:
        return "estable"
    return "crecimiento"
