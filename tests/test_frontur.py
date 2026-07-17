"""Tests de armonización FRONTUR (sin descarga de red)."""

import pandas as pd

from src.frontur import harmonize_frontur_cube, merge_frontur_with_panel


def _main_cube_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TIME_PERIOD_CODE": ["2020-M01", "2020-M01", "2020-A01"],
            "TIPO_VIAJERO#es": ["Turistas principales", "Turistas principales", "Turistas principales"],
            "LUGAR_RESIDENCIA#es": ["Total", "Total", "Total"],
            "TERRITORIO#es": ["Tenerife", "Canarias", "Tenerife"],
            "MEDIDAS_CODE": ["TURISTAS", "TURISTAS", "TURISTAS"],
            "OBS_VALUE": [1000.0, 5000.0, 12000.0],
        }
    )


def _gomera_cube_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TIME_PERIOD_CODE": ["2020-M03"],
            "TIPO_VIAJERO#es": ["Turista"],
            "LUGAR_RESIDENCIA#es": ["Total"],
            "TERRITORIO#es": ["La Gomera"],
            "MEDIDAS_CODE": ["VIAJEROS"],
            "OBS_VALUE": [500.0],
        }
    )


def test_harmonize_main_cube_filters_monthly_and_islands():
    out = harmonize_frontur_cube(_main_cube_sample())
    assert len(out) == 1
    assert out.iloc[0]["isla"] == "Tenerife"
    assert out.iloc[0]["turistas"] == 1000.0
    assert out.iloc[0]["fecha"] == pd.Timestamp("2020-01-01")


def test_harmonize_gomera_cube_fixed_isla():
    out = harmonize_frontur_cube(_gomera_cube_sample(), fixed_isla="La Gomera")
    assert len(out) == 1
    assert out.iloc[0]["isla"] == "La Gomera"
    assert out.iloc[0]["turistas"] == 500.0


def test_merge_frontur_with_panel():
    panel = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "isla": ["Tenerife", "El Hierro"],
            "revpar_eur": [80.0, 40.0],
        }
    )
    frontur = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2020-01-01"]),
            "isla": ["Tenerife"],
            "turistas": [1000.0],
        }
    )
    merged = merge_frontur_with_panel(panel, frontur)
    assert merged.loc[merged["isla"] == "Tenerife", "turistas"].iloc[0] == 1000.0
    assert pd.isna(merged.loc[merged["isla"] == "El Hierro", "turistas"].iloc[0])
