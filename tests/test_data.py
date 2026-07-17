"""Tests de src/data.py — validación de esquema y carga, sin depender del CSV real."""

import pandas as pd
import pytest

from src.data import ISLANDS, load_harmonized_series, to_wide_by_island


def _write_valid_csv(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "fecha": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "isla": ["Tenerife", "Tenerife", "Gran Canaria", "Gran Canaria"],
            "revpar_eur": [80.0, 82.0, 90.0, 91.0],
        }
    ).to_csv(path, index=False)
    return path


def test_load_harmonized_series_raises_if_file_missing(tmp_path):
    missing = tmp_path / "no_existe.csv"
    with pytest.raises(FileNotFoundError):
        load_harmonized_series(path=missing)


def test_load_harmonized_series_raises_on_missing_required_columns(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame({"fecha": ["2020-01-01"], "revpar_eur": [80.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Faltan columnas"):
        load_harmonized_series(path=path)


def test_load_harmonized_series_raises_without_any_indicator(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame({"fecha": ["2020-01-01"], "isla": ["Tenerife"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ningún indicador"):
        load_harmonized_series(path=path)


def test_load_harmonized_series_raises_on_unknown_island(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {"fecha": ["2020-01-01"], "isla": ["Mallorca"], "revpar_eur": [80.0]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no reconocidos"):
        load_harmonized_series(path=path)


def test_load_harmonized_series_sorts_by_isla_and_fecha(tmp_path):
    path = _write_valid_csv(tmp_path)
    df = load_harmonized_series(path=path)
    assert list(df["isla"]) == sorted(df["isla"])  # islas en orden alfabético
    assert df.groupby("isla")["fecha"].apply(lambda s: s.is_monotonic_increasing).all()


def test_load_harmonized_series_accepts_partial_indicators(tmp_path):
    """No exige que TODAS las columnas indicador estén presentes (docstring de data.py)."""
    path = _write_valid_csv(tmp_path)
    df = load_harmonized_series(path=path)
    assert "revpar_eur" in df.columns
    assert "adr_eur" not in df.columns


def test_to_wide_by_island_pivots_correctly(tmp_path):
    path = _write_valid_csv(tmp_path)
    df = load_harmonized_series(path=path)
    wide = to_wide_by_island(df, "revpar_eur")
    assert set(wide.columns) == {"Tenerife", "Gran Canaria"}
    assert wide.loc[pd.Timestamp("2020-01-01"), "Tenerife"] == 80.0


def test_islands_constant_has_seven_canary_islands():
    assert len(ISLANDS) == 7
    assert "Tenerife" in ISLANDS and "El Hierro" in ISLANDS
