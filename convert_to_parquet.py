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


def _mr_selected_mask(col: pd.Series) -> pd.Series:
    """
    Uma coluna-opção de bloco MR aparece em dois formatos possíveis: o
    original (texto da opção repetido / vazio quando não marcado) ou o
    otimizado por convert_to_parquet.py (booleano puro, True = marcado --
    ver _optimize_memory lá). Os dois precisam funcionar aqui porque
    list_variables.py e testes locais às vezes leem xlsx cru direto, sem
    passar pela otimização. Um booleano puro nunca é "nulo" -- .notna()
    nele sempre devolveria True e marcaria todo mundo como selecionado,
    por isso o branch por dtype.
    """
    if pd.api.types.is_bool_dtype(col):
        return col.fillna(False).astype(bool)
    return col.notna()


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
            mask = _mr_selected_mask(col)
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
            any_selected |= _mr_selected_mask(_get_series(data, name))
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
    cell_n: pd.DataFrame       # N não ponderado por célula (mesma forma de pct) -- vira a linha "NA"
    cell_weighted: pd.DataFrame  # N ponderado por célula, pré-divisão -- usado pra calcular %LINHA
    base_n: pd.Series          # N não ponderado por categoria do banner
    small_n_flag: pd.Series    # True onde base_n < limiar
    coverage_warning: str | None = None  # ver _check_coverage


def _check_coverage(stub_long_filtered: pd.DataFrame, stub_label: str, threshold: float = 0.9) -> str | None:
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

    Texto em linguagem simples de propósito -- quem usa o app não
    necessariamente sabe o que é "lógica de pulo" ou "stub".
    """
    if stub_long_filtered.empty:
        return None
    dist = stub_long_filtered.drop_duplicates("resp_id")["category"].value_counts(normalize=True)
    if dist.empty:
        return None
    top_cat, top_share = dist.index[0], dist.iloc[0]
    if top_share >= threshold:
        return (
            f"Quase todo mundo que respondeu essa pergunta é do grupo "
            f"'{top_cat}' em {stub_label} ({top_share:.0%}) -- provavelmente "
            f"a pergunta só foi feita pra esse grupo. Não dá pra comparar "
            f"{stub_label} de forma justa aqui, porque os outros grupos "
            f"quase não têm gente respondendo."
        )
    return None


def _check_base_coverage(banner_n: int, total_n: int, threshold: float = 0.9) -> str | None:
    """
    Segundo tipo de aviso, diferente de _check_coverage: não é a base estar
    concentrada num balde só do stub, é a base elegível pra essa variável de
    banner ser bem menor que o Total da tabela -- geralmente lógica de pulo
    (a pergunta não foi feita pra todo mundo). Achado real: "utiliza água de
    poço" tinha base de 64.216 contra Total de 85.201 (75,4%) -- sem esse
    aviso, só dava pra perceber somando a linha Base Amostra na mão, que foi
    exatamente como esse caso apareceu.

    Texto em linguagem simples de propósito -- "lógica de pulo" é jargão de
    quem constrói questionário, não de quem lê o banner.
    """
    if total_n <= 0:
        return None
    coverage = banner_n / total_n
    if coverage < threshold:
        return (
            f"Só {banner_n} de {total_n} pessoas do Total responderam essa "
            f"pergunta ({coverage:.0%}) -- provavelmente ela não foi feita "
            f"pra todo mundo. Os percentuais estão certos, só valem pra "
            f"esse grupo menor, não pro Total inteiro."
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
    total_n: int,
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
    cell_n_table = (
        joined.groupby(["category_stub", "category_banner"])["resp_id"]
        .nunique()
        .unstack("category_banner")
        .reindex(columns=base_weighted.index)
        .fillna(0)
        .astype(int)
    )

    pct = cell_table.divide(base_weighted, axis=1) * 100
    pct = pct.fillna(0.0)

    coverage_warning = _check_coverage(stub_long, get_label(meta, stub_key)) or _check_base_coverage(len(both_elig), total_n)

    return BannerBlock(
        banner_key=banner_key,
        banner_label=get_label(meta, banner_key),
        pct=pct,
        cell_n=cell_n_table.reindex(index=pct.index, fill_value=0),
        cell_weighted=cell_table.fillna(0.0),
        base_n=base_n.reindex(pct.columns).fillna(0).astype(int),
        small_n_flag=(base_n.reindex(pct.columns).fillna(0) < small_n_threshold),
        coverage_warning=coverage_warning,
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
    total_cell_n = total_base_pairs.groupby("category")["resp_id"].nunique()
    total_pct = (total_cell / total_weighted * 100).to_frame("Total")
    total_block = BannerBlock(
        banner_key=total_meta_key,
        banner_label="Total",
        pct=total_pct,
        cell_n=total_cell_n.reindex(total_pct.index).fillna(0).astype(int).to_frame("Total"),
        cell_weighted=total_cell.reindex(total_pct.index).fillna(0.0).to_frame("Total"),
        base_n=pd.Series({"Total": total_base_pairs["resp_id"].nunique()}),
        small_n_flag=pd.Series({"Total": total_base_pairs["resp_id"].nunique() < small_n_threshold}),
    )
    blocks.append(total_block)

    for bk in banner_keys:
        block = _build_single_block(
            data, meta, stub_key, bk, weights, na_handling, small_n_threshold,
            total_n=total_base_pairs["resp_id"].nunique(),
        )
        if block.pct.empty:
            # Acontece quando a variável de banner não tem nenhum respondente
            # elegível nesse recorte (ex.: pergunta de um branch de rota que
            # ninguém caiu nesse estudo/onda). Não é erro -- é sinal de que
            # essa variável não se aplica a esse recorte de dados.
            continue
        blocks.append(block)

    return blocks


def format_banner_table_full(blocks: list[BannerBlock]) -> pd.DataFrame:
    """
    Formato completo -- NA (contagem não ponderada), %LINHA e %COLUNA por
    célula, o padrão clássico de banner do SPSS/Quantum, com três linhas por
    categoria do stub em vez de uma.

    %COLUNA é o que já calculávamos (`pct`). %LINHA é novo: cada célula
    dividida pelo total ponderado da PRÓPRIA categoria do stub (o bloco
    "Total", que é sempre o primeiro de `blocks` por construção de
    `build_banner`) -- não pela base da coluna, que é o denominador do
    %COLUNA.
    """
    total_block = blocks[0]
    stub_categories = list(total_block.pct.index)
    row_totals_weighted = total_block.cell_weighted["Total"]

    n_frames, linha_frames, coluna_frames = [], [], []
    for b in blocks:
        cols = pd.MultiIndex.from_product([[b.banner_label], b.pct.columns])

        n_frame = b.cell_n.copy()
        n_frame.columns = cols
        n_frames.append(n_frame)

        coluna_frame = b.pct.copy()
        coluna_frame.columns = cols
        coluna_frames.append(coluna_frame)

        linha_frame = b.cell_weighted.divide(row_totals_weighted, axis=0) * 100
        linha_frame = linha_frame.fillna(0.0)
        linha_frame.columns = cols
        linha_frames.append(linha_frame)

    na_table = pd.concat(n_frames, axis=1).fillna(0).astype(int)
    linha_table = pd.concat(linha_frames, axis=1).fillna(0.0).round(1)
    coluna_table = pd.concat(coluna_frames, axis=1).fillna(0.0).round(1)

    rows: dict[tuple[str, str], pd.Series] = {}
    for cat in stub_categories:
        rows[(cat, "NA")] = na_table.loc[cat]
        rows[(cat, "%LINHA")] = linha_table.loc[cat]
        rows[(cat, "%COLUNA")] = coluna_table.loc[cat]

    base_na = na_table.sum(axis=0)
    base_total_na = base_na[("Total", "Total")]
    rows[("Base Amostra", "NA")] = base_na
    rows[("Base Amostra", "%LINHA")] = (base_na / base_total_na * 100).round(1) if base_total_na else base_na * 0.0
    rows[("Base Amostra", "%COLUNA")] = pd.Series(100.0, index=base_na.index).round(1)

    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index)
    return table


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


def small_n_mask_full(blocks: list[BannerBlock]) -> pd.DataFrame:
    """
    Mesma ideia de small_n_mask, mas pro formato NA/%LINHA/%COLUNA de
    format_banner_table_full -- a marcação de N pequeno vale pra coluna
    inteira (é a base daquela coluna que é pequena, não uma métrica
    específica), então se repete nas 3 linhas de cada categoria do stub e
    também nas 3 linhas de "Base Amostra".
    """
    total_block = blocks[0]
    stub_categories = list(total_block.pct.index)
    col_masks = []
    for b in blocks:
        cols = pd.MultiIndex.from_product([[b.banner_label], b.pct.columns])
        col_masks.append(pd.Series(b.small_n_flag.reindex(b.pct.columns).values, index=cols))
    flat = pd.concat(col_masks)

    row_labels = [(cat, metric) for cat in stub_categories for metric in ("NA", "%LINHA", "%COLUNA")]
    row_labels += [("Base Amostra", metric) for metric in ("NA", "%LINHA", "%COLUNA")]
    index = pd.MultiIndex.from_tuples(row_labels)
    return pd.DataFrame(np.tile(flat.values, (len(index), 1)), index=index, columns=flat.index)
