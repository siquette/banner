"""
indices.py — Cálculo de tendência, cobertura e importância dos índices de
tracking (IACOM, IMC, IM...) por variável de segmento (tipicamente ANO).

PAPEL NO PROJETO
-----------------
Alimenta a aba "Índices" de app.py (visão geral, individual e quadrante).
Não reaproveita `crosstab_engine.build_banner` de propósito: índice não
tem categoria pra cruzar, tem um número (a média do companion `_media`,
ver `metadata.py`). É groupby + média ponderada, mais simples que o motor
de %-cruzamento -- misturar os dois deixaria os dois mais confusos de
manter, e nenhum dos dois ganharia nada com isso.

TRÊS CÁLCULOS, TRÊS PROPÓSITOS
---------------------------------
- `compute_index_trend`: um índice, segmentado (ex.: por ANO) -- usado na
  visão "Índice individual" e como base do quadrante "Tendência x Nível".
- `compute_quadrant_data`: os índices de uma vez (15, com IAG/ISDE via
  `_MEDIA_ALIASES`), resumidos em nível +
  tendência + importância -- usado nas duas visões de quadrante.
- `_weighted_corr`: a peça de "importância" -- correlação ponderada entre
  dois índices, pessoa por pessoa (não é causa, é "importância derivada",
  aproximação padrão de mercado quando não se pergunta importância
  diretamente).
"""

from __future__ import annotations

import pandas as pd

from crosstab_engine import get_column_series
from metadata import VariableMeta, VarType


# ══════════════════════════════════════════════════════════════════════
#  MAPEAMENTO ÍNDICE -> COMPANION NUMÉRICO
# ══════════════════════════════════════════════════════════════════════

_MEDIA_ALIASES: dict[str, str] = {
    "IAG": "P46_media",
    "ISDE": "P23_media",
}
"""
Exceções confirmadas ao pareamento automático por nome (`scale_base`
== nome do indicador). Os dois casos aqui são perguntas ÚNICAS (P46,
P23) cujo indicador tem um "apelido" (IAG, ISDE) que não bate com o
código da pergunta de origem -- diferente dos outros 12 índices, que
são compostos de várias perguntas e nascem com `_c`/`_media` já
consistentes entre si.

Confirmado célula a célula contra o banco (não só por classificação de
coluna): `P46_media` bate 100% com o mapeamento ÓTIMO=5.../PÉSSIMO=1
da categórica `IAG`; `P23_media` bate 100% com os buckets da
categórica `ISDE`. As colunas "óbvias" (`IAG_media` -- nem existe;
`ISDE_media` -- existe mas 100% vazia) NÃO são a fonte real.

IAG nem chega a ser classificado como INDICATOR (o rótulo completo não
termina em "_c" -- é a pergunta P46 crua, sem o prefixo "INDICADOR DE
..."), por isso entra aqui incondicionalmente, não só como override de
um pareamento que já existia.
"""


def indicator_media_map(meta: dict[str, VariableMeta]) -> dict[str, str]:
    """
    Nome do indicador (ex.: "IACOM") -> nome da sua coluna `_media`
    companion (ex.: "IACOM_media"), via `scale_base` -- mapeamento exato,
    não substring (uma tentativa anterior por substring confundia "IM"
    com "IMC_media", já que "IM" é prefixo de "IMC").

    `_MEDIA_ALIASES` entra por cima do pareamento automático -- cobre o
    caso de indicador cujo companion numérico tem nome de pergunta, não
    nome de índice (ver docstring de `_MEDIA_ALIASES`).
    """
    media_by_base = {m.scale_base: m.name for m in meta.values() if m.var_type == VarType.SCALE_MEDIA}
    mapping = {
        m.name: media_by_base[m.name]
        for m in meta.values()
        if m.var_type == VarType.INDICATOR and m.name in media_by_base
    }
    for indicator, media_col in _MEDIA_ALIASES.items():
        if media_col in meta:  # defensivo: nunca aponta pra coluna que não existe no banco carregado
            mapping[indicator] = media_col
    return mapping


# ══════════════════════════════════════════════════════════════════════
#  TENDÊNCIA DE UM ÍNDICE
# ══════════════════════════════════════════════════════════════════════

_NEUTRAL_IMPUTATION: dict[str, float] = {
    "IACOM_media": 3.0,
}
"""
Índice em que ausência de resposta é neutro por definição, não "sem
dado". Confirmado com o Ro: no IACOM, quem não teve contato com a
comunicação da empresa entra na média como nota 3 (ponto médio da
escala 1-5) em vez de ser excluído -- o universo do índice é todo
mundo, não só quem foi exposto.

Só o IACOM até agora. IAA/IAC/IAD/IAPS têm o mesmíssimo padrão de N/A
na categórica (ver auditoria anterior), mas cada um precisa ser
confirmado antes de entrar aqui -- aplicar essa regra errado muda
número reportado pro cliente, e "parece o mesmo padrão" não é
confirmação.
"""


def _media_values(data: pd.DataFrame, media_col: str) -> pd.Series:
    """
    Lê o companion `_media` já convertido pra número, com a imputação
    de `_NEUTRAL_IMPUTATION` aplicada quando for o caso. Ponto único de
    leitura -- usado tanto por `compute_index_trend` (nível/tendência)
    quanto por `compute_quadrant_data` (importância/correlação), pra
    "o valor de um respondente nesse índice" nunca divergir entre as
    duas contas. A alternativa -- imputar só na tendência e deixar a
    correlação sem imputar -- deixaria o mesmo índice com dois
    "universos" diferentes dependendo de qual gráfico se olha, o que é
    mais confuso do que qualquer ganho de pureza estatística na
    correlação.
    """
    value = pd.to_numeric(get_column_series(data, media_col), errors="coerce")
    if media_col in _NEUTRAL_IMPUTATION:
        value = value.fillna(_NEUTRAL_IMPUTATION[media_col])
    return value


def compute_index_trend(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    media_col: str,
    segment_key: str,
    weights: pd.Series,
) -> pd.DataFrame:
    """
    Média ponderada e cobertura de UM índice por categoria de uma
    variável de segmento (ex.: ANO). `segment_key` precisa ser SR --
    segmentar um índice por variável de múltipla resposta não tem uma
    definição óbvia (a pessoa contaria em mais de um segmento ao mesmo
    tempo), então não é suportado aqui de propósito (levanta `ValueError`
    se tentado).

    Devolve um DataFrame indexado pela categoria do segmento, com colunas
    `media` (ponderada), `cobertura_pct` (não ponderada) e `n` (não
    ponderado, pra avaliar confiabilidade -- mesmo princípio de Base
    Amostra no banner).
    """
    seg_meta = meta[segment_key]
    if seg_meta.var_type != VarType.SR:
        raise ValueError(f"'{segment_key}' não é resposta única -- segmentação de índice só suporta SR.")

    value = _media_values(data, media_col)
    segment = get_column_series(data, seg_meta.name)

    df = pd.DataFrame({"value": value, "segment": segment, "weight": weights})
    df = df[df["segment"].notna()]

    total_n = df.groupby("segment").size()
    covered = df[df["value"].notna()]
    covered_n = covered.groupby("segment").size().reindex(total_n.index).fillna(0).astype(int)
    coverage_pct = (covered_n / total_n * 100).fillna(0.0)

    weighted_sum = (covered["value"] * covered["weight"]).groupby(covered["segment"]).sum()
    weight_sum = covered.groupby("segment")["weight"].sum()
    media = (weighted_sum / weight_sum).reindex(total_n.index)

    return pd.DataFrame({"media": media, "cobertura_pct": coverage_pct, "n": covered_n})


# ══════════════════════════════════════════════════════════════════════
#  QUADRANTE (NÍVEL, TENDÊNCIA, IMPORTÂNCIA)
# ══════════════════════════════════════════════════════════════════════

def _sort_key(value):
    """Tenta ordenar onda como número (2022 < 2023...); cai pra texto se não for numérico."""
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _weighted_corr(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    """
    Correlação de Pearson ponderada -- usada pra "importância derivada"
    no quadrante: o quanto um índice específico anda junto com o índice
    de referência, pessoa por pessoa. NaN se não der pra calcular (poucos
    pontos em comum, ou variância zero em algum dos dois -- ex.: todo
    mundo respondeu a mesma coisa, sem variação nenhuma pra correlacionar).

    Correlação não é causa -- é aproximação padrão da indústria pra
    importância quando não se pergunta importância diretamente. Uma
    versão mais rigorosa controlaria os outros índices ao mesmo tempo
    (regressão múltipla), não par a par; fica como possível próximo
    passo, não construído agora.
    """
    mask = x.notna() & y.notna() & w.notna()
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 2 or w.sum() == 0:
        return float("nan")
    wx = (w * x).sum() / w.sum()
    wy = (w * y).sum() / w.sum()
    cov = (w * (x - wx) * (y - wy)).sum() / w.sum()
    varx = (w * (x - wx) ** 2).sum() / w.sum()
    vary = (w * (y - wy) ** 2).sum() / w.sum()
    if varx <= 0 or vary <= 0:
        return float("nan")
    return cov / (varx * vary) ** 0.5


def compute_quadrant_data(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    media_map: dict[str, str],
    weights: pd.Series,
    wave_key: str = "ANO",
    reference_indicator: str = "IM",
) -> pd.DataFrame:
    """
    Uma linha por índice, com tudo que os dois tipos de quadrante
    precisam: nível (média na onda mais recente), tendência (nível atual
    menos onda anterior), importância (correlação ponderada com o índice
    de referência, pessoa por pessoa) e cobertura.

    "NÍVEL" usa especificamente a onda mais recente de `wave_key`, não a
    média cega de todas as ondas juntas -- misturar anos daria um número
    que não corresponde a "como estamos agora". Se `wave_key` não existir
    ou só tiver uma onda na base filtrada, nível cai pra média da base
    inteira e tendência fica NaN (não dá pra comparar uma onda com ela
    mesma).

    "IMPORTÂNCIA" usa a base filtrada inteira, não só a onda mais recente
    -- correlação precisa de volume pra ser estável, e misturar ondas é
    prática padrão de mercado pra esse cálculo (a relação entre os
    índices tende a ser mais estável no tempo do que o nível de cada um
    sozinho).

    Índices sem dado suficiente (nível ou importância NaN) continuam na
    tabela devolvida -- é responsabilidade de quem desenha o gráfico
    (`app.py`) decidir se filtra antes de plotar.
    """
    reference_col = media_map.get(reference_indicator)
    reference_values = _media_values(data, reference_col) if reference_col else None

    waves_sorted: list = []
    if wave_key in meta and meta[wave_key].var_type == VarType.SR:
        wave_series = get_column_series(data, meta[wave_key].name)
        waves_sorted = sorted(wave_series.dropna().unique().tolist(), key=_sort_key)

    rows = []
    for ind, media_col in media_map.items():
        values = _media_values(data, media_col)

        level, trend, n, coverage_pct = float("nan"), float("nan"), 0, 0.0
        if len(waves_sorted) >= 1:
            trend_df = compute_index_trend(data, meta, media_col, wave_key, weights).reindex(waves_sorted)
            level = trend_df["media"].iloc[-1]
            n = int(trend_df["n"].iloc[-1])
            coverage_pct = float(trend_df["cobertura_pct"].iloc[-1])
            if len(waves_sorted) >= 2 and pd.notna(trend_df["media"].iloc[-2]):
                trend = level - trend_df["media"].iloc[-2]
        else:
            mask_cov = values.notna()
            n = int(mask_cov.sum())
            coverage_pct = n / len(data) * 100 if len(data) else 0.0
            if n > 0:
                level = (values[mask_cov] * weights[mask_cov]).sum() / weights[mask_cov].sum()

        importance = float("nan")
        if reference_values is not None and media_col != reference_col:
            importance = _weighted_corr(values, reference_values, weights)

        rows.append({
            "indice": ind, "nivel": level, "tendencia": trend,
            "importancia": importance, "cobertura_pct": coverage_pct, "n": n,
        })

    return pd.DataFrame(rows).set_index("indice")
