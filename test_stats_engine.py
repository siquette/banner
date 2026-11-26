"""
Testes de stats_engine.py. Rodar da raiz do repositório: `pixi run pytest`
(ou `python -m pytest tests`).

A estratégia é comparar com quem já é referência, em vez de repetir a
fórmula no teste:
1. Peso igual pra todo mundo -> o resultado tem que ser o do numpy/scipy
   sem peso (média, variância ddof=1, erro padrão, IC t, quantil
   "inverted_cdf").
2. Peso inteiro k -> média, quantis, moda e distribuição têm que ser os
   da base com cada linha repetida k vezes (é o que "peso" quer dizer).
3. Casos de borda: vazio, um valor só, peso zero, valor faltando.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import stats_engine as se

RNG = np.random.default_rng(42)
X = RNG.normal(50, 12, size=500).round(1)
ESCALA = RNG.integers(1, 6, size=400).astype(float)


# ── 1. peso igual = estatística comum ────────────────────────────────

@pytest.mark.parametrize("x", [X, ESCALA])
def test_peso_igual_bate_com_numpy_scipy(x):
    w = np.full(len(x), 2.5)  # qualquer constante, não só 1
    d = se.descrever(x, w)
    assert d["n"] == len(x)
    assert d["n_efetivo"] == pytest.approx(len(x))
    assert d["media"] == pytest.approx(np.mean(x))
    assert d["variancia"] == pytest.approx(np.var(x, ddof=1))
    assert d["desvio_padrao"] == pytest.approx(np.std(x, ddof=1))
    assert d["erro_padrao"] == pytest.approx(stats.sem(x))
    de, ate = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
    assert d["ic_de"] == pytest.approx(de)
    assert d["ic_ate"] == pytest.approx(ate)
    for p, chave in [(0.25, "q1"), (0.5, "mediana"), (0.75, "q3")]:
        assert d[chave] == pytest.approx(np.quantile(x, p, method="inverted_cdf"))
    assert d["minimo"] == x.min() and d["maximo"] == x.max()


def test_peso_igual_moda():
    x = np.array([1, 2, 2, 3, 3, 3, 4], dtype=float)
    assert se.descrever(x)["moda"] == 3


# ── 2. peso inteiro = linha repetida ─────────────────────────────────

@pytest.mark.parametrize("x", [X[:200], ESCALA[:200]])
def test_peso_inteiro_igual_a_repetir_linhas(x):
    k = RNG.integers(1, 6, size=len(x))
    repetido = np.repeat(x, k)
    d = se.descrever(x, k.astype(float))
    assert d["media"] == pytest.approx(np.mean(repetido))
    for p, chave in [(0.25, "q1"), (0.5, "mediana"), (0.75, "q3")]:
        assert d[chave] == pytest.approx(np.quantile(repetido, p, method="inverted_cdf"))
    valores, contagem = np.unique(repetido, return_counts=True)
    assert d["moda"] == valores[np.argmax(contagem)]
    dist = se.distribuicao(x, k)
    esperado = pd.Series(contagem / contagem.sum() * 100, index=valores)
    pd.testing.assert_series_equal(dist, esperado, check_names=False, check_index_type=False)


def test_n_efetivo_kish():
    w = np.array([1.0, 1.0, 4.0])
    assert se.n_efetivo(w) == pytest.approx(36 / 18)


def test_erro_padrao_cresce_com_peso_desigual():
    """Mesmo dado, peso desigual -> erro padrão maior que o sem peso (a perda de informação do peso)."""
    w = np.where(np.arange(len(X)) % 10 == 0, 20.0, 1.0)
    assert se.erro_padrao(X, w) > se.erro_padrao(X, np.ones_like(X))


# ── 3. bordas ────────────────────────────────────────────────────────

def test_vazio_e_um_valor():
    d0 = se.descrever([np.nan, np.nan], [1, 1])
    assert d0["n"] == 0 and np.isnan(d0["media"]) and np.isnan(d0["mediana"])
    d1 = se.descrever([7.0], [3.0])
    assert d1["media"] == 7 and d1["mediana"] == 7
    assert np.isnan(d1["variancia"]) and np.isnan(d1["erro_padrao"]) and np.isnan(d1["ic_de"])


def test_peso_zero_e_valor_faltando_saem():
    d = se.descrever([1.0, 2.0, np.nan, 100.0], [1.0, 1.0, 5.0, 0.0])
    assert d["n"] == 2 and d["media"] == 1.5 and d["maximo"] == 2


def test_peso_faltando_vira_1():
    d = se.descrever([1.0, 3.0], [np.nan, 1.0])
    assert d["media"] == 2


def test_tipo_numerica():
    assert se.tipo_numerica(ESCALA) == se.TIPO_ESCALA
    assert se.tipo_numerica(np.arange(0, 11)) == se.TIPO_ESCALA       # 0 a 10 = 11 valores
    assert se.tipo_numerica(np.arange(0, 12)) == se.TIPO_CONTINUA
    assert se.tipo_numerica(X) == se.TIPO_CONTINUA


def test_descrever_por_grupo():
    valores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pesos = pd.Series([1.0] * 5)
    grupos = pd.Series(["b", "a", "b", None, "a"])
    t = se.descrever_por_grupo(valores, pesos, grupos, ordem_grupos=["a", "b"])
    assert list(t.index) == ["Total", "a", "b"]
    assert t.loc["Total", "n"] == 5            # sem grupo entra só no Total
    assert t.loc["a", "media"] == pytest.approx(3.5)
    assert t.loc["b", "media"] == pytest.approx(2.0)


def test_histograma_soma_100_e_inteiros():
    bordas, pct = se.histograma(np.repeat([18, 30, 45, 90], 10).astype(float))
    assert pct.sum() == pytest.approx(100)
    assert np.allclose(np.mod(bordas, 1), 0.5)  # faixas inteiras, centradas no número


def test_caixa():
    x = np.concatenate([np.arange(1, 101, dtype=float), [1000.0]])
    c = se.caixa(x)
    assert c["mediana"] == np.quantile(x, 0.5, method="inverted_cdf")
    assert c["bigode_sup"] == 100 and list(c["fora"]) == [1000.0]


def test_amostra_indices_fixa():
    a, b = se.amostra_indices(10_000, 50), se.amostra_indices(10_000, 50)
    assert np.array_equal(a, b) and len(np.unique(a)) == 50
    assert np.array_equal(se.amostra_indices(30, 50), np.arange(30))
