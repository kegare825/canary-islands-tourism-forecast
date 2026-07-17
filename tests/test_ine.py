"""Tests de src/ine.py — parseo de respuesta INE mockeada."""

import pandas as pd

from src.ine import fetch_ine_series_by_cod


class TestFetchIneSeriesByCod:
    def test_parses_table_payload(self, monkeypatch):
        payload = [
            {
                "COD": "EOT1544",
                "Nombre": "Canarias. Pernoctaciones. Total.",
                "T3_Unidad": "Pernoctaciones",
                "Data": [
                    {"Fecha": "2024-01-01T00:00:00.000+01:00", "Valor": 1000.0},
                    {"Fecha": "2024-02-01T00:00:00.000+01:00", "Valor": 1100.0},
                ],
            }
        ]

        def fake_table(table_id):
            return payload

        monkeypatch.setattr("src.ine.fetch_ine_table", fake_table)
        df = fetch_ine_series_by_cod("EOT1544")
        assert len(df) == 2
        assert df["valor"].tolist() == [1000.0, 1100.0]
        assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
