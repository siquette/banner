"""
Testes de indices.compute_index_cross (visão "Duas variáveis" da aba
Índices) e de indices.media_values (a leitura de índice que a aba
Estatística passou a usar).

A referência é o próprio `compute_index_trend`, que o app já usa há
meses: o resultado de uma cidade no cruzamento tem que ser exatamente o
de `compute_index_trend` rodado só nas entrevistas dessa cidade.
"""

import numpy as np
import pandas as pd
import pytest

from crosstab_engine import get_weights
from indices import compute_index_cross, compute_index_trend, media_values, neutral_imputation
from metadata import classify_columns

RNG = np.random.default_rng(7)
N = 600


def _banco() -> tuple[pd.DataFrame, dict]:
    iacom = RNG.integers(1, 6, N).astype(float)
    iacom[RNG.random(N) < 0.3] = np.nan          # 30% sem contato com a comunicação
    outro = RNG.normal(3.5, 0.8, N).clip(1, 5)
    outro[RNG.random(N) < 0.2] = np.nan
    cidades = RNG.choice(["Cidade A", "Cidade B", "Cidade C"], N).astype(object)
    cidades[RNG.random(N) < 0.05] = None          # sem cidade: fica fora do cruzamento
    d = pd.DataFrame({
        "ANO": RNG.choice(["2024", "2025", "2026"], N),
        "CIDADES": cidades,
        "IACOM_media": iacom,
        "IFA_media": outro,
        "PESO": RNG.uniform(0.2, 3.0, N),
    })
    # cidade C não foi pesquisada em 2026: a combinação não pode aparecer
    d = d[~((d["CIDADES"] == "Cidade C") & (d["ANO"] == "2026"))].reset_index(drop=True)
    return d, classify_columns(d, list(d.columns), list(d.columns))


@pytest.mark.parametrize("media_col", ["IACOM_media", "IFA_media"])
def test_cada_linha_igual_ao_trend_so_daquela_linha(media_col):
    d, meta = _banco()
    w = get_weights(d, meta)
    cruz = compute_index_cross(d, meta, media_col, "CIDADES", "ANO", w)
    for cidade in ["Cidade A", "Cidade B", "Cidade C"]:
        so = d["CIDADES"] == cidade
        ref = compute_index_trend(d[so], meta, media_col, "ANO", w[so])
        ref = ref[ref.index.isin(d.loc[so, "ANO"].unique())]
        got = cruz.xs(cidade, level="CIDADES").reindex(ref.index)
        pd.testing.assert_series_equal(got["media"], ref["media"], check_names=False)
        pd.testing.assert_series_equal(got["n"], ref["n"], check_names=False, check_dtype=False)
        pd.testing.assert_series_equal(got["cobertura_pct"], ref["cobertura_pct"], check_names=False)


def test_combinacao_que_nao_existe_nao_aparece():
    d, meta = _banco()
    cruz = compute_index_cross(d, meta, "IFA_media", "CIDADES", "ANO", get_weights(d, meta))
    assert ("Cidade C", "2026") not in cruz.index
    assert ("Cidade A", "2026") in cruz.index


def test_regra_do_iacom():
    d, meta = _banco()
    assert neutral_imputation("IACOM_media") == 3.0 and neutral_imputation("IFA_media") is None
    v = media_values(d, "IACOM_media")
    assert v.notna().all() and (v[d["IACOM_media"].isna()] == 3.0).all()
    # com a regra, o IACOM cobre todo mundo; o IFA continua só com quem respondeu
    cruz_iacom = compute_index_cross(d, meta, "IACOM_media", "CIDADES", "ANO", get_weights(d, meta))
    cruz_ifa = compute_index_cross(d, meta, "IFA_media", "CIDADES", "ANO", get_weights(d, meta))
    assert (cruz_iacom["cobertura_pct"] == 100).all()
    assert (cruz_ifa["cobertura_pct"] < 100).any()


def test_linha_que_nao_e_resposta_unica_recusa():
    d, meta = _banco()
    with pytest.raises(ValueError):
        compute_index_cross(d, meta, "IFA_media", "IFA_media", "ANO", get_weights(d, meta))
