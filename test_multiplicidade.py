"""Teste de crosstab_engine.mentions_per_respondent (índice de multiplicidade)."""

import pandas as pd
import pytest

from crosstab_engine import mentions_per_respondent


class _Bloco:
    """Só o que a função lê de um BannerBlock: rótulo e %COLUNA."""

    def __init__(self, label, pct):
        self.banner_label = label
        self.pct = pct


def test_soma_do_pct_coluna_dividida_por_100():
    # 10 pessoas no Total: 6 marcaram A, 5 marcaram B, 4 marcaram C -> 15 menções / 10 = 1,5
    total = _Bloco("Total", pd.DataFrame({"Total": [60.0, 50.0, 40.0]}, index=["A", "B", "C"]))
    ano = _Bloco("ANO", pd.DataFrame({"2024": [50.0, 50.0, 0.0], "2025": [70.0, 50.0, 80.0]}, index=["A", "B", "C"]))
    s = mentions_per_respondent([total, ano])
    assert s[("Total", "Total")] == pytest.approx(1.5)
    assert s[("ANO", "2024")] == pytest.approx(1.0)
    assert s[("ANO", "2025")] == pytest.approx(2.0)
