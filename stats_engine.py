"""
stats_engine.py — Estatística descritiva PONDERADA de variáveis numéricas.

PAPEL NO PROJETO
-----------------
A conta da aba "📐 Estatística" (parte Descritivas). Recebe valores e
pesos, devolve números e as estruturas que os gráficos desenham
(distribuição, faixas de histograma, caixa do box plot). Nada aqui
conhece o Streamlit nem o Plotly, pelo mesmo motivo de
`crosstab_engine.py`: dá pra testar cada conta isolada (`tests/`).

POR QUE TUDO PONDERADO
-----------------------
O app trabalha com o PESO ligado por padrão, e o PESO varia muito (de
~0,02 a ~41 no banco de 2026). Média, mediana e % sem peso descrevem a
AMOSTRA; com peso, estimam a população que a amostra representa. Com o
peso desligado na barra lateral, quem chama passa peso 1 pra todo mundo
e todas as fórmulas abaixo viram as de sempre (os testes conferem isso
contra numpy/scipy) -- a mesma ideia de `crosstab_engine.get_weights`:
um caminho só, muda só o valor do peso.

AS FÓRMULAS (e por que estas)
------------------------------
- n: entrevistas com valor (contagem bruta, nunca ponderada).
- n efetivo (Kish): (Σw)² / Σw². Quanto a amostra "vale" depois do peso:
  peso muito desigual faz 1.000 entrevistas valerem como menos. No banco
  de 2026 inteiro, o n efetivo é ~49% do n.
- média: Σwx / Σw.
- variância: n/(n−1) · Σw(x−m)² / Σw. É a do pacote `survey` do R
  (`svyvar`); com peso igual vira a variância amostral (ddof=1).
- erro padrão da média: √(n/(n−1) · Σw²(x−m)²) / Σw -- linearização
  (Taylor), a do `svymean` do R pra amostra sem conglomerado. Ela já
  carrega a perda causada pelo peso desigual; com peso igual vira s/√n.
- IC 95%: média ± t(0,975; n_efetivo − 1) · erro padrão. A t (e não o
  1,96 fixo) importa em grupo pequeno -- ex.: uma cidade com 40
  entrevistas depois do filtro.
- quantis (mediana, quartis): o menor valor x em que a soma dos pesos até
  x chega à fração p do peso total ("inverted CDF"). Com peso igual é o
  `numpy.quantile(method="inverted_cdf")`. Devolve sempre um valor que
  existe no dado -- numa escala de 1 a 5, a mediana é 3 ou 4, nunca 3,5.
- moda: o valor com maior soma de pesos (empate: o menor valor).

AMOSTRA POR COTA
-----------------
As pesquisas são por cota, não por sorteio. Erro padrão e IC aqui são
a convenção de mercado (tratam a amostra como se fosse aleatória): servem
pra comparar a incerteza entre grupos, não como probabilidade de erro de
verdade. O app escreve isso na tela.

ESCALA x CONTÍNUA
------------------
Das 159 colunas `_media` do banco de 2026, 102 têm 5 valores ou menos
(item de escala 1-5). Nelas, box plot esconde a diferença (dos 15
índices, 10 têm a caixa exatamente de 3 a 4, com médias de 3,27 a 4,21)
e histograma é só um gráfico de barras. `tipo_numerica` separa as duas
pra o app recomendar o gráfico certo; o corte (até 11 valores distintos)
pega escala de 1 a 5 e de 0 a 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIPO_ESCALA = "escala"
TIPO_CONTINUA = "contínua"
LIMITE_ESCALA = 11  # até 11 valores distintos = escala (1-5, 0-10)

# (chave, rótulo na tela) na ordem em que as colunas aparecem na tabela.
# "ic" vira duas colunas (de / até).
ESTATISTICAS: list[tuple[str, str]] = [
    ("n", "n"),
    ("n_efetivo", "n efetivo"),
    ("media", "média"),
    ("erro_padrao", "erro padrão"),
    ("ic", "IC 95%"),
    ("mediana", "mediana"),
    ("moda", "moda"),
    ("desvio_padrao", "desvio padrão"),
    ("variancia", "variância"),
    ("minimo", "mínimo"),
    ("q1", "1º quartil"),
    ("q3", "3º quartil"),
    ("maximo", "máximo"),
]
ESTATISTICAS_PADRAO = ["n", "n_efetivo", "media", "ic", "mediana", "desvio_padrao"]


# ── preparação ────────────────────────────────────────────────────────

def limpar(valores, pesos=None) -> tuple[np.ndarray, np.ndarray]:
    """
    (x, w) só com as entrevistas que têm valor numérico e peso positivo.
    Peso ausente vira 1 (mesma regra de `get_weights`); peso zero ou
    negativo sai -- não representa ninguém e quebraria o n efetivo.
    """
    x = pd.to_numeric(pd.Series(valores), errors="coerce").to_numpy(dtype=float)
    if pesos is None:
        w = np.ones_like(x)
    else:
        w = pd.to_numeric(pd.Series(pesos), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if len(w) != len(x):
            raise ValueError("valores e pesos com tamanhos diferentes")
    ok = ~np.isnan(x) & (w > 0)
    return x[ok], w[ok]


def tipo_numerica(valores) -> str:
    """'escala' (até LIMITE_ESCALA valores distintos) ou 'contínua'."""
    x = pd.to_numeric(pd.Series(valores), errors="coerce").dropna()
    return TIPO_ESCALA if x.nunique() <= LIMITE_ESCALA else TIPO_CONTINUA


# ── estatísticas ──────────────────────────────────────────────────────

def n_efetivo(w: np.ndarray) -> float:
    """Kish: (Σw)² / Σw². Igual a n quando todos os pesos são iguais."""
    if len(w) == 0:
        return 0.0
    return float(w.sum() ** 2 / (w ** 2).sum())


def media(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w)) if len(x) else np.nan


def variancia(x: np.ndarray, w: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return np.nan
    m = media(x, w)
    return float(n / (n - 1) * np.sum(w * (x - m) ** 2) / np.sum(w))


def erro_padrao(x: np.ndarray, w: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return np.nan
    m = media(x, w)
    return float(np.sqrt(n / (n - 1) * np.sum((w * (x - m)) ** 2)) / np.sum(w))


def intervalo_confianca(x: np.ndarray, w: np.ndarray, nivel: float = 0.95) -> tuple[float, float]:
    from scipy import stats  # import aqui: só o IC precisa do scipy

    ep = erro_padrao(x, w)
    if np.isnan(ep):
        return (np.nan, np.nan)
    gl = max(n_efetivo(w) - 1.0, 1.0)
    t = float(stats.t.ppf(0.5 + nivel / 2, gl))
    m = media(x, w)
    return (m - t * ep, m + t * ep)


def quantil(x: np.ndarray, w: np.ndarray, p: float) -> float:
    """Menor x em que o peso acumulado chega a p do total (ver docstring do módulo)."""
    if len(x) == 0:
        return np.nan
    ordem = np.argsort(x, kind="stable")
    xs, ws = x[ordem], w[ordem]
    acum = np.cumsum(ws) / ws.sum()
    # tolerância: sem ela, 0,49999999 (soma de float) perderia de 0,5
    i = int(np.searchsorted(acum, p - 1e-12, side="left"))
    return float(xs[min(i, len(xs) - 1)])


def moda(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    soma = pd.Series(w).groupby(x).sum()
    return float(soma.index[np.argmax(soma.to_numpy())])  # argmax pega o primeiro = menor valor no empate


def descrever(valores, pesos=None) -> dict[str, float]:
    """Todas as estatísticas de ESTATISTICAS de uma vez (o app escolhe quais mostrar)."""
    x, w = limpar(valores, pesos)
    ic = intervalo_confianca(x, w)
    vazio = len(x) == 0
    var = variancia(x, w)
    return {
        "n": float(len(x)),
        "n_efetivo": n_efetivo(w),
        "media": media(x, w),
        "erro_padrao": erro_padrao(x, w),
        "ic_de": ic[0],
        "ic_ate": ic[1],
        "mediana": quantil(x, w, 0.5),
        "moda": moda(x, w),
        "desvio_padrao": float(np.sqrt(var)) if not np.isnan(var) else np.nan,
        "variancia": var,
        "minimo": np.nan if vazio else float(x.min()),
        "q1": quantil(x, w, 0.25),
        "q3": quantil(x, w, 0.75),
        "maximo": np.nan if vazio else float(x.max()),
    }


def descrever_por_grupo(valores: pd.Series, pesos: pd.Series, grupos: pd.Series | None = None,
                        ordem_grupos: list | None = None, rotulo_total: str = "Total") -> pd.DataFrame:
    """
    Uma linha por grupo (mais "Total" em cima), uma coluna por estatística.
    Entrevista sem grupo (vazio na variável de quebra) entra só no Total.
    `ordem_grupos`: ordem das linhas (o app passa `sort_categories`); sem
    ela, a ordem é a dos valores.
    """
    linhas = {rotulo_total: descrever(valores, pesos)}
    if grupos is not None:
        g = grupos.astype("object")
        presentes = [v for v in pd.unique(g.dropna())]
        if ordem_grupos is not None:
            presentes = [v for v in ordem_grupos if v in set(presentes)]
        for v in presentes:
            mascara = (g == v).to_numpy()
            linhas[str(v)] = descrever(valores[mascara], pesos[mascara])
    return pd.DataFrame.from_dict(linhas, orient="index")


# ── estruturas pros gráficos ──────────────────────────────────────────

def distribuicao(valores, pesos=None) -> pd.Series:
    """% (ponderado) de cada valor, em ordem de valor. Soma 100."""
    x, w = limpar(valores, pesos)
    if len(x) == 0:
        return pd.Series(dtype=float)
    soma = pd.Series(w).groupby(x).sum().sort_index()
    return soma / soma.sum() * 100


def histograma(valores, pesos=None, max_faixas: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """
    (bordas, % ponderado por faixa). As bordas vêm da regra 'auto' do
    numpy aplicada aos valores (o formato da distribuição não depende do
    peso, só a altura), limitadas a `max_faixas` pra não virar pente.
    Valor inteiro (idade, nº de moradores) ganha faixa de largura inteira,
    senão a faixa cai entre dois inteiros e o histograma serrilha.
    """
    x, w = limpar(valores, pesos)
    if len(x) == 0:
        return np.array([]), np.array([])
    bordas = np.histogram_bin_edges(x, bins="auto")
    if len(bordas) - 1 > max_faixas:
        bordas = np.histogram_bin_edges(x, bins=max_faixas)
    if np.all(np.equal(np.mod(x, 1), 0)):
        largura = max(1.0, np.ceil((x.max() - x.min() + 1) / max_faixas))
        bordas = np.arange(x.min() - 0.5, x.max() + 0.5 + largura, largura)
    soma, _ = np.histogram(x, bins=bordas, weights=w)
    return bordas, soma / w.sum() * 100


def caixa(valores, pesos=None, max_pontos_fora: int = 300) -> dict:
    """
    Números do box plot, com quantis ponderados: q1, mediana, q3, média,
    bigodes (o valor mais extremo dentro de 1,5 x a altura da caixa) e os
    valores fora dos bigodes (distintos, até `max_pontos_fora`). O Plotly
    não pondera box plot sozinho -- por isso a conta é feita aqui e o app
    só desenha.
    """
    x, w = limpar(valores, pesos)
    if len(x) == 0:
        return {}
    q1, med, q3 = quantil(x, w, 0.25), quantil(x, w, 0.5), quantil(x, w, 0.75)
    altura = q3 - q1
    dentro = x[(x >= q1 - 1.5 * altura) & (x <= q3 + 1.5 * altura)]
    inf, sup = float(dentro.min()), float(dentro.max())
    fora = np.unique(x[(x < inf) | (x > sup)])
    if len(fora) > max_pontos_fora:  # mantém os extremos, que são os que interessam
        metade = max_pontos_fora // 2
        fora = np.concatenate([fora[:metade], fora[-metade:]])
    return {"q1": q1, "mediana": med, "q3": q3, "media": media(x, w),
            "bigode_inf": inf, "bigode_sup": sup, "fora": fora}


def amostra_indices(n: int, tamanho: int = 5000, semente: int = 0) -> np.ndarray:
    """Posições de uma amostra aleatória fixa (mesma semente = mesmos pontos a cada rodada)."""
    if n <= tamanho:
        return np.arange(n)
    return np.sort(np.random.default_rng(semente).choice(n, size=tamanho, replace=False))
