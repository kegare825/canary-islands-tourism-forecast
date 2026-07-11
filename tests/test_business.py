"""Tests de src/business.py — no dependen de datos ni de un modelo entrenado."""

import pytest

from src.business import demand_alert_level, revpar, revpar_gap_vs_last_year


def test_revpar_basic():
    assert revpar(adr_eur=100, occupancy_rate=0.7) == pytest.approx(70.0)


def test_revpar_rejects_invalid_occupancy():
    with pytest.raises(ValueError):
        revpar(adr_eur=100, occupancy_rate=1.5)


def test_revpar_gap_when_occupancy_drops():
    result = revpar_gap_vs_last_year(forecast_occupancy=0.5, last_year_occupancy=0.7, adr_eur=100)
    assert result.forecast_revpar_eur == pytest.approx(50.0)
    assert result.target_revpar_eur == pytest.approx(70.0)
    assert result.gap_eur == pytest.approx(20.0)
    # Para no perder los 20€ de RevPAR con menos ocupación, el ADR tendría que subir a 140€ (+40%)
    assert result.adr_increase_needed_pct == pytest.approx(0.4)


def test_revpar_gap_is_zero_when_occupancy_matches():
    result = revpar_gap_vs_last_year(forecast_occupancy=0.6, last_year_occupancy=0.6, adr_eur=100)
    assert result.gap_eur == pytest.approx(0.0)
    assert result.adr_increase_needed_pct == pytest.approx(0.0)


@pytest.mark.parametrize(
    "yoy,expected",
    [(-0.30, "alerta_alta"), (-0.15, "alerta_alta"), (-0.10, "alerta_media"), (-0.05, "alerta_media"), (0.0, "estable"), (0.10, "crecimiento")],
)
def test_demand_alert_level_boundaries(yoy, expected):
    assert demand_alert_level(yoy) == expected
