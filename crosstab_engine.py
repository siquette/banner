"""
Motor de cruzamento ponderado, com suporte nativo a múltipla resposta.

Decisão central de arquitetura: em vez de pivotar o df inteiro (caro, com
1164 colunas seria bobagem materializar tudo de uma vez), normalizamos cada
variável -- SR, indicador ou bloco MR -- para o MESMO formato "longo" só na
hora em que ela é selecionada no filtro:

    resp_id | category | weight

Uma SR vira uma linha por respondente. Um bloco MR vira várias linhas por
respondente (uma por opção marcada) -- esse é o "pivotar que você faz no
Power BI", só que automático e sob demanda. Depois disso, cruzar qualquer
combinação SR x SR, SR x MR, MR x MR é o mesmo merge + groupby, porque os
dois lados já estão no mesmo formato. É essa normalização que elimina o
retrabalho manual por pergunta.

Duas convenções estatísticas fixadas aqui, e o porquê:

1. Percentual usa peso (PESO), mas "Base Amostra" reporta N não ponderado.
   É o padrão de mercado: o peso corrige a leitura do %, mas quem decide se
   uma célula é confiável estatisticamente é o tamanho real da amostra, não
   o tamanho ponderado (que pode parecer maior ou menor que a realidade).

2. A base de uma variável MR é "respondentes com pelo menos 1 opção marcada
   no bloco" -- não dá para distinguir com certeza, só pelos dados, quem foi
   filtrado por lógica de pulo de quem foi perguntado e marcou zero opções.
   Isso é uma limitação real, documentada aqui, não escondida. Se o bloco tiver
   uma opção explícita tipo "Nenhuma", ela funciona como o marcador de
   "perguntado e respondeu nada", e a base fica correta automaticamente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metadata import VariableMeta, VarType, get_label

_NA_TEXT_PATTERN = r"^N/A\b"


def get_weights(data: pd.DataFrame, meta: dict[str, VariableMeta]) -> pd.Series:
    """Devolve a série de pesos alinhada ao índice do df. Peso 1.0 se não houver coluna PESO."""
    weight_names = [m.name for m in meta.values() if m.var_type == VarType.WEIGHT]
    if not weight_names:
        return pd.Series(1.0, index=data.index)
    col = data[weight_names[0]]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce").fillna(1.0)


def _get_series(data: pd.DataFrame, name: str) -> pd.Series:
    col = data[name]
    if isinstance(col, pd.DataFrame):  # nomes curtos duplicados no arquivo original
        col = col.iloc[:, 0]
    return col


def get_column_series(data: pd.DataFrame, name: str) -> pd.Series:
    """Wrapper público de _get_series -- pra app.py não precisar importar nome privado do módulo."""
    return _get_series(data, name)


def to_long(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    key: str,
    weights: pd.Series,
    na_handling: str = "keep",
) -> pd.DataFrame:
    """
    Normaliza uma variável (SR, indicador ou bloco MR identificado por `key`)
    para o formato longo resp_id | category | weight.

    na_handling: 'keep' mantém a categoria "N/A - ..." de indicadores como
    categoria própria; 'exclude' remove essas linhas (a % passa a ser só de
    quem de fato respondeu). Não se aplica a SR nem a MR -- lá "não preenchido"
    já significa "não elegível", tratado em `eligible_respondents`.
    """
    is_mr = any(m.var_type == VarType.MR_OPTION and m.mr_group == key for m in meta.values())

    if is_mr:
        option_metas = [m for m in meta.values() if m.var_type == VarType.MR_OPTION and m.mr_group == key]
        frames = []
        for m in option_metas:
            col = _get_series(data, m.name)
            mask = col.notna()
            idx = data.index[mask]
            frames.append(pd.DataFrame({
                "resp_id": idx,
                "category": m.mr_option_label,
                "weight": weights.loc[idx].values,
            }))
        if not frames:
            return pd.DataFrame(columns=["resp_id", "category", "weight"])
        return pd.concat(frames, ignore_index=True)

    m = meta[key]
    col = _get_series(data, m.name)
    mask = col.notna()
    if m.var_type == VarType.INDICATOR and na_handling == "exclude":
        mask &= ~col.astype(str).str.match(_NA_TEXT_PATTERN, case=False, na=False)
    idx = data.index[mask]
    return pd.DataFrame({
        "resp_id": idx,
        "category": col[mask].astype(str).values,
        "weight": weights.loc[idx].values,
    })


def eligible_respondents(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    key: str,
    na_handling: str = "keep",
) -> pd.Index:
    """
    Índice de respondentes elegíveis para a variável `key` -- ou seja, o
    universo que deveria compor a base do banner para essa variável, antes
    de cruzar com qualquer outra coisa.
    """
    is_mr = any(m.var_type == VarType.MR_OPTION and m.mr_group == key for m in meta.values())
    if is_mr:
        option_names = [m.name for m in meta.values() if m.var_type == VarType.MR_OPTION and m.mr_group == key]
        any_selected = pd.Series(False, index=data.index)
        for name in option_names:
            any_selected |= _get_series(data, name).notna()
        return data.index[any_selected]

    m = meta[key]
    col = _get_series(data, m.name)
    mask = col.notna()
    if m.var_type == VarType.INDICATOR and na_handling == "exclude":
        mask &= ~col.astype(str).str.match(_NA_TEXT_PATTERN, case=False, na=False)
    return data.index[mask]


@dataclass
class BannerBlock:
    banner_key: str
    banner_label: str
    pct: pd.DataFrame          # linhas = categorias do stub, colunas = categorias do banner, valores = % coluna
    base_n: pd.Series          # N não ponderado por categoria do banner
    small_n_flag: pd.Series    # True onde base_n < limiar
    coverage_warning: str | None = None  # ver _check_coverage


def _check_coverage(stub_long_filtered: pd.DataFrame, threshold: float = 0.9) -> str | None:
    """
    Detecta um problema diferente de N pequeno: a população elegível pro
    cruzamento inteiro concentrada numa única categoria do stub. Isso
    aconteceu de verdade no banco de produção -- um bloco de múltipla
    resposta (motivo de não conectar à rede) só tinha respondente do ano de
    2024, porque a pergunta simplesmente não foi feita nas outras ondas do
    estudo consolidado. O resultado (100% em 2024, 0% nos outros anos, em
    TODAS as opções do bloco) é aritmeticamente correto, mas não descreve
    diferença de comportamento entre anos -- descreve em que ano a pergunta
    existiu. N pequeno não pega isso porque a base total pode ser grande
    (577 pessoas, no caso real); o problema é a base inteira estar num só
    balde do stub, não o tamanho dela.
    """
    if stub_long_filtered.empty:
        return None
    dist = stub_long_filtered.drop_duplicates("resp_id")["category"].value_counts(normalize=True)
    if dist.empty:
        return None
    top_cat, top_share = dist.index[0], dist.iloc[0]
    if top_share >= threshold:
        return (
            f"{top_share:.0%} da base elegível pra esse cruzamento está em "
            f"'{top_cat}' -- essa variável de banner provavelmente só existe "
            f"nessa categoria do stub (ex.: só foi perguntada numa onda "
            f"específica), não é diferença de comportamento real."
        )
    return None


def _build_single_block(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    stub_key: str,
    banner_key: str,
    weights: pd.Series,
    na_handling: str,
    small_n_threshold: int,
) -> BannerBlock:
    stub_long = to_long(data, meta, stub_key, weights, na_handling)
    banner_long = to_long(data, meta, banner_key, weights, na_handling)

    stub_elig = eligible_respondents(data, meta, stub_key, na_handling)
    banner_elig = eligible_respondents(data, meta, banner_key, na_handling)
    both_elig = stub_elig.intersection(banner_elig)

    stub_long = stub_long[stub_long.resp_id.isin(both_elig)]
    banner_long = banner_long[banner_long.resp_id.isin(both_elig)]

    # Base por categoria do banner: respondentes distintos elegíveis para o
    # stub, dentro de cada categoria do banner. dropna por resp_id+categoria
    # evita contar a mesma opção 2x (não deveria acontecer, mas é barato garantir).
    base_pairs = banner_long.drop_duplicates(["resp_id", "category"])
    base_weighted = base_pairs.groupby("category")["weight"].sum()
    base_n = base_pairs.groupby("category")["resp_id"].nunique()

    joined = stub_long.merge(banner_long, on="resp_id", suffixes=("_stub", "_banner"))
    cell_weighted = joined.groupby(["category_stub", "category_banner"])["weight_stub"].sum()
    cell_table = cell_weighted.unstack("category_banner").reindex(columns=base_weighted.index)

    pct = cell_table.divide(base_weighted, axis=1) * 100
    pct = pct.fillna(0.0)

    return BannerBlock(
        banner_key=banner_key,
        banner_label=get_label(meta, banner_key),
        pct=pct,
        base_n=base_n.reindex(pct.columns).fillna(0).astype(int),
        small_n_flag=(base_n.reindex(pct.columns).fillna(0) < small_n_threshold),
        coverage_warning=_check_coverage(stub_long),
    )


def build_banner(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    stub_key: str,
    banner_keys: list[str],
    na_handling: str = "keep",
    small_n_threshold: int = 30,
) -> list[BannerBlock]:
    """
    Ponto de entrada principal. Um bloco por variável de banner selecionada --
    é assim que uma tabela banner de verdade é composta: cada corte (Região,
    Gênero, Cliente x Não Cliente...) é cruzado independentemente contra o
    mesmo stub, não combinado entre si.

    A primeira coluna sempre é "Total" -- toda a base elegível para o stub,
    sem nenhum corte de banner -- porque é a referência que todo banner real
    tem antes das colunas de corte.
    """
    weights = get_weights(data, meta)

    total_meta_key = "__TOTAL__"
    blocks = []

    stub_elig_all = eligible_respondents(data, meta, stub_key, na_handling)
    stub_long_all = to_long(data, meta, stub_key, weights, na_handling)
    stub_long_all = stub_long_all[stub_long_all.resp_id.isin(stub_elig_all)]
    total_base_pairs = stub_long_all.drop_duplicates(["resp_id"])
    total_weighted = total_base_pairs["weight"].sum()
    total_cell = stub_long_all.groupby("category")["weight"].sum()
    total_pct = (total_cell / total_weighted * 100).to_frame("Total")
    total_block = BannerBlock(
        banner_key=total_meta_key,
        banner_label="Total",
        pct=total_pct,
        base_n=pd.Series({"Total": total_base_pairs["resp_id"].nunique()}),
        small_n_flag=pd.Series({"Total": total_base_pairs["resp_id"].nunique() < small_n_threshold}),
    )
    blocks.append(total_block)

    for bk in banner_keys:
        block = _build_single_block(data, meta, stub_key, bk, weights, na_handling, small_n_threshold)
        if block.pct.empty:
            # Acontece quando a variável de banner não tem nenhum respondente
            # elegível nesse recorte (ex.: pergunta de um branch de rota que
            # ninguém caiu nesse estudo/onda). Não é erro -- é sinal de que
            # essa variável não se aplica a esse recorte de dados.
            continue
        blocks.append(block)

    return blocks


def format_banner_table(blocks: list[BannerBlock]) -> pd.DataFrame:
    """
    Junta os blocos num único DataFrame de exibição: colunas em MultiIndex
    (variável de banner, categoria), linha extra "Base Amostra" no topo,
    igual à convenção que aparece nas suas planilhas SPC.
    """
    pct_frames = []
    base_row = {}
    for b in blocks:
        cols = pd.MultiIndex.from_product([[b.banner_label], b.pct.columns])
        frame = b.pct.copy()
        frame.columns = cols
        pct_frames.append(frame)
        for cat, n in b.base_n.items():
            base_row[(b.banner_label, cat)] = n

    table = pd.concat(pct_frames, axis=1).fillna(0.0).round(1)
    table.index.name = None
    base_series = pd.Series(base_row)
    base_series.index = pd.MultiIndex.from_tuples(base_series.index)
    table.loc["Base Amostra"] = base_series.reindex(table.columns)
    return table


def small_n_mask(blocks: list[BannerBlock]) -> pd.DataFrame:
    """Máscara booleana (mesmo shape de format_banner_table, sem a linha Base Amostra) para estilizar células de N pequeno."""
    masks = []
    for b in blocks:
        cols = pd.MultiIndex.from_product([[b.banner_label], b.pct.columns])
        m = pd.DataFrame(
            np.tile(b.small_n_flag.reindex(b.pct.columns).values, (len(b.pct.index), 1)),
            index=b.pct.index, columns=cols,
        )
        masks.append(m)
    return pd.concat(masks, axis=1)
