"""
app.py — Interface do gerador de banner.

PAPEL NO PROJETO
-----------------
Fina de propósito: toda a lógica pesada (classificação de variável,
cruzamento ponderado, unpivot de múltipla resposta, cálculo de índice)
está em `metadata.py`, `crosstab_engine.py` e `indices.py`, cada um
testado isoladamente. Este arquivo só orquestra widgets do Streamlit e
desenha o resultado -- se uma mudança aqui está alterando um NÚMERO, não
só a apresentação dele, provavelmente ela deveria estar num dos outros
três módulo.

BANCO FIXO
-----------
Lê sempre o mesmo `.parquet`, commitado no repositório ao lado deste
arquivo (gerado uma vez por `convert_to_parquet.py`, fora do app -- ver
README). Sem uploader nem campo de caminho na UI -- não faz sentido pra
um app com um cliente/estudo fixo, e evita reintroduzir a leitura de xlsx
cru (lenta, minutos pro banco de produção) dentro do container do
Streamlit Cloud, que tem teto de memória e tempo de execução.

ESTRUTURA DESTE ARQUIVO
--------------------------
1. Carregamento (cache) -- `_load_and_classify`
2. Gráficos -- `_build_chart`, compartilhado pelas duas abas
3. Aba Cruzamento -- `_render_cruzamento_tab`
4. Aba Índices -- três visões (`_render_indices_overview/_individual/
   _quadrant`) chamadas por `_render_indices_tab`
5. Aba Estatística -- `_render_estatistica_tab` (descritivas das
   numéricas; a conta mora em `stats_engine.py`)
6. Aba Filtro SQL -- `_render_sql_tab`, e o filtro SQL em si
   (`_filtro_sql_atual`, aviso fixo `_render_aviso_sql`); a conta mora em
   `sql_filter.py`
7. Barra lateral -- `_render_sidebar`, devolve os valores escolhidos
8. `main()` -- amarra tudo

Rodar localmente:
    pixi run streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import stats_engine as se

from crosstab_engine import (
    banner_table_to_excel_bytes,
    build_banner,
    describe_block_base,
    format_banner_table_full,
    format_table_for_export,
    get_column_series,
    get_weights,
    linha_pct_frame,
    mentions_per_respondent,
    small_n_mask_full,
    table_to_excel_bytes,
    variables_with_data,
)
from indices import (
    compute_index_cross,
    compute_index_trend,
    compute_quadrant_data,
    indicator_media_map,
    media_values,
    neutral_imputation,
)
from metadata import VarType, category_color, classify_columns, crossable_variables, load_parquet_with_labels, sort_categories
from sql_filter import (
    LIMITE_SEGUNDOS,
    ErroFiltroSQL,
    calcular_mascara,
    colunas_citadas,
    como_escrever,
    limpar_condicao,
    mensagem_sem_linhas,
    resumo_valores,
    sem_acento_minusculo,
)

st.set_page_config(page_title="Gerador de Banner", layout="wide")

# Caminho fixo do banco, relativo a este arquivo -- funciona igual local e
# no Streamlit Cloud, porque nos dois casos o diretório de trabalho é a
# raiz do repositório clonado. BANNER_DATA_PATH sobrescreve isso via
# variável de ambiente, só pra testar outro arquivo localmente sem editar
# código -- nunca é setada em produção.
DATA_PATH = Path(os.environ.get(
    "BANNER_DATA_PATH",
    str(Path(__file__).parent / "data" / "processed" / "df_completo_2026.parquet"),
))


# ══════════════════════════════════════════════════════════════════════
#  CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Carregando o banco...")
def _load_and_classify(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Lê o parquet e classifica as colunas -- cacheado pelo Streamlit
    (`@st.cache_data`) porque, sem isso, QUALQUER interação (trocar o
    limiar de N pequeno, mudar um seletor) rerodaria o app inteiro e
    reclassificaria as 1164 colunas do zero a cada clique. O cache é
    invalidado automaticamente se `path` mudar.
    """
    data, full_labels, short_names = load_parquet_with_labels(path)
    meta = classify_columns(data, full_labels, short_names)
    return data, meta


@st.cache_data(show_spinner=False)
def _build_banner_cached(
    data: pd.DataFrame,
    meta: dict,
    stub_key: str,
    banner_keys: tuple[str, ...],
    na_handling: str,
    small_n_threshold: int,
    use_weighting: bool,
    use_universe: bool = True,
) -> list:
    """
    Wrapper cacheado sobre `crosstab_engine.build_banner` -- vive aqui em
    `app.py`, não dentro do motor, porque cache é decisão de orquestração
    de UI (depende de como o Streamlit reroda o script), não faz parte
    da lógica de cruzamento em si.

    POR QUE ISSO PRECISAVA EXISTIR: sem cache, `build_banner` recalculava
    do zero a CADA rerun do Streamlit -- inclusive quando a interação nem
    tinha relação com o cruzamento (ex.: mexer no slider de espaçamento
    de barra do gráfico). Isso já era desperdício com 1 cruzamento na
    tela; com o modo Comparação (2 cruzamentos simultâneos, ver
    `_render_comparacao_tab`), o mesmo desperdício dobra -- por isso essa
    função é pré-requisito do modo Comparação, não um extra dele.

    `banner_keys` é `tuple`, não `list`, porque `@st.cache_data` precisa
    hashear os argumentos pra decidir se reusa o resultado -- `list` não
    é hasheável, `tuple` é. Todo call site abaixo converte antes de
    chamar.

    `data` (o DataFrame inteiro) segue sendo hasheado pelo Streamlit a
    cada chamada -- tem custo, mas é o mesmo custo que `_load_and_classify`
    já paga hoje; não otimizamos isso agora (ex.: cache manual por
    `id(data)`) sem medir primeiro se o hash do df é o gargalo real ou
    se o ganho já vem todo do cache do cruzamento em si.
    """
    return build_banner(
        data, meta, stub_key, list(banner_keys), na_handling, small_n_threshold, use_weighting,
        use_universe=use_universe,
    )


@st.cache_data(show_spinner=False)
def _visible_keys_cached(filters_signature: tuple, _filtered_data: pd.DataFrame, _meta: dict, keys: tuple) -> tuple:
    """
    Quais variáveis cruzáveis têm ao menos uma resposta na base já filtrada
    (ver `crosstab_engine.variables_with_data`). Cacheado por ASSINATURA DOS
    FILTROS, não pelo DataFrame: o parâmetro `_filtered_data` começa com
    `_` justamente pra o Streamlit NÃO hashear o df (hashear ~100 mil linhas
    x 1.100 colunas a cada rerun custaria mais que o próprio cálculo). É
    seguro porque a base filtrada é função determinística dos filtros
    (cascata em ordem) sobre um parquet fixo.

    Sempre usa na_handling='keep': a lista de variáveis é montada na barra
    lateral ANTES do seletor de N/A existir naquele rerun, e uma variável
    que só tem 'N/A - ...' continua sendo uma pergunta que existe nesse
    recorte -- quem decide excluir N/A vê o bloco sumir do cruzamento (o
    motor já descarta bloco vazio), não a variável sumir da lista.
    """
    return tuple(sorted(variables_with_data(_filtered_data, _meta, list(keys), "keep")))


# ══════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ══════════════════════════════════════════════════════════════════════

CHART_STACKED = "Barra empilhada 100%"
CHART_TYPES = ["Barras", "Barra horizontal", CHART_STACKED, "Linha"]

# Abaixo disso o segmento é estreito demais pro rótulo caber dentro: o
# Plotly corta ou sobrepõe o texto. O valor continua no hover.
_STACKED_MIN_LABEL_PCT = 4.0
# Segmento que completa a barra até 100% (quem está no grupo mas tem a variável
# EM BRANCO na base). NÃO começa com "N/A" de propósito: categorias reais "N/A - ..."
# do indicador já são cinza (#9E9E9E) e ficavam idênticas a este, lado a lado.
_STACKED_GAP_LABEL = "Em branco na base"
_STACKED_GAP_COLOR = "#E3E3E3"  # cinza bem claro: distinto do cinza das categorias "N/A - ..."


def _stacked_sum_note(pct_df: pd.DataFrame, tolerance: float = 1.0) -> str | None:
    """
    Avisa quando as barras do empilhado NÃO somam 100%.

    O gráfico se chama "100%", mas %LINHA divide pela base do bloco (universo
    da pergunta, ou Total do stub se essa opção estiver desligada) e não
    pela base de quem respondeu o banner, então a soma só
    é 100% se todo mundo do grupo respondeu o banner E cada pessoa cai
    em uma única coluna. Duas situações quebram isso:
      - banner de múltipla escolha: a pessoa conta em 2+ colunas -> soma > 100%;
      - cobertura baixa (nem todo mundo respondeu o banner) -> soma < 100%.
    Em vez de esconder isso, o aviso diz a causa -- uma barra que
    "deveria" fechar em 100% e não fecha parece bug de cálculo.
    """
    sums = pct_df.sum(axis=1)
    if sums.empty or ((sums - 100).abs() <= tolerance).all():
        return None
    lo, hi = sums.min(), sums.max()
    return (
        f"⚠️ As barras não fecham em 100% (somam de {lo:.1f}% a {hi:.1f}%). "
        "Acima de 100%: o banner é de múltipla escolha (a mesma pessoa conta em mais de uma cor). "
        "Abaixo de 100%: nem todo mundo do grupo respondeu o banner (veja a cobertura acima)."
    )


def _is_mr_key(meta: dict, key: str) -> bool:
    """True se `key` é um grupo de múltipla escolha (chave = código do grupo, não coluna)."""
    return any(m.var_type == VarType.MR_OPTION and m.mr_group == key for m in meta.values())


def _prepare_stacked(
    pct_linha: pd.DataFrame, meta: dict, banner_key: str, banner_label: str, stub_label: str,
) -> tuple[pd.DataFrame, str | None, str, str]:
    """
    Faz as barras do empilhado fecharem 100% SEM mudar nenhum número.

    Histórico (importa pra não repetir): o %LINHA da tabela divide pela
    base do grupo INTEIRO do stub, que inclui quem não respondeu o banner
    ou caiu em "N/A" (esses não viram segmento). Então as fatias visíveis
    somam < 100% (ex.: 77% do ano 2023 respondeu o indicador). Primeiro
    tentei re-normalizar cada barra pela soma das fatias -- barras fechavam
    100%, mas os números deixavam de bater com a tabela (BOM 39,3% virava
    45,6%), e o gráfico perdia a confiança do analista. Agora os valores
    são exatamente os da tabela e o que falta vira um segmento "Em branco
    na base" (cinza claro): a barra é 100% do GRUPO, e a lacuna aparece em
    vez de ser escondida.

    Por que NÃO é uma categoria "N/A": a pergunta pode ter uma categoria
    real "N/A - Não citou ter visto..." (resposta de verdade, coluna
    própria, cinza #9E9E9E). A lacuna é outra coisa -- gente do grupo que
    tem a variável NULA na base, em nenhuma coluna da tabela. Mesmo cinza
    deixava as duas indistinguíveis (visto em produção, IACOM 2022/2025:
    6,8% de "N/A" real ao lado de 13,8% de lacuna).

    Desde que a base do %LINHA passou a ser o "universo da pergunta"
    (`crosstab_engine.infer_question_universe`), grupos onde a pergunta nem
    existe saem do denominador e não viram lacuna: o que sobra de "Em
    branco" é não-resposta de verdade dentro do universo. Com a opção
    desligada (Total inteiro), a lacuna volta a misturar as duas coisas.

    Banner de múltipla escolha NÃO ganha segmento de lacuna: a soma conta
    menções (passa de 100%), então "o que falta" não tem leitura de
    pessoas. Fica como está, só com o aviso.

    Retorna (df, nota, título do eixo X, contexto do hover).
    """
    axis = f"% dentro de cada grupo de {stub_label}"
    if _is_mr_key(meta, banner_key):
        return pct_linha, _stacked_sum_note(pct_linha), axis, "desse grupo"

    gap = (100.0 - pct_linha.sum(axis=1)).clip(lower=0.0)
    gap = gap.where(gap > 0.05, 0.0)  # sobra de arredondamento não vira segmento
    if not (gap > 0).any():
        return pct_linha, None, axis, "desse grupo"

    out = pct_linha.copy()
    out[_STACKED_GAP_LABEL] = gap
    note = (
        f"ℹ️ **Em branco na base** (cinza claro): {gap.min():.1f}% a {gap.max():.1f}% das pessoas de cada grupo "
        f"não têm valor em {banner_label}, então não entram em nenhuma coluna da tabela. "
        "É diferente de uma categoria \"N/A - ...\" (essa é uma resposta e tem coluna própria). "
        "As demais fatias são exatamente os %LINHA da tabela. "
        "Se parte desse grupo nem foi perguntada (a pergunta não existe pra ela), confira a opção "
        "\"Base do %LINHA\" na barra lateral."
    )
    return out, note, axis, "desse grupo"


def _build_stacked_chart(
    pct_df: pd.DataFrame,
    title: str,
    value_axis_title: str,
    legend_title: str,
    category_label: str,
    value_context: str,
    bargap: float,
) -> go.Figure:
    """
    Barra horizontal empilhada: uma barra por categoria do stub, um
    segmento por categoria do banner, % escrito dentro de cada segmento.

    Diferente de "Barra horizontal" (barras lado a lado, uma por série),
    aqui as séries se somam na mesma barra -- por isso só faz sentido com
    %LINHA, onde cada linha da tabela soma ~100%.

    Decisões:
    - `autorange="reversed"` no eixo Y: o Plotly desenha a primeira
      categoria embaixo; aqui ela fica no topo (mesma ordem da tabela e da
      figura de referência). "Barra horizontal" mantém o comportamento
      antigo de propósito -- não foi pedido mexer nela.
    - Rótulo só em segmentos >= `_STACKED_MIN_LABEL_PCT`; texto vazio nos
      demais (o hover mostra o valor).
    - Contorno branco fino separando os segmentos, no lugar de cantos
      arredondados: `marker.cornerradius` só existe em versões recentes do
      Plotly e daria erro em ambiente mais antigo.
    - Eixo X vai até 100, ou até a maior soma se passar disso (múltipla
      escolha) -- fixar em 100 cortaria as barras.
    """
    fig = go.Figure()
    cats = pct_df.index.astype(str)
    for col in pct_df.columns:
        vals = pct_df[col]
        labels = [f"{v:.1f}%" if v >= _STACKED_MIN_LABEL_PCT else "" for v in vals]
        fig.add_trace(go.Bar(
            name=str(col), y=cats, x=vals, orientation="h",
            text=labels, textposition="inside", insidetextanchor="middle", textangle=0,
            hovertemplate=f"{col}<br>{category_label}: %{{y}}<br>%{{x:.1f}}% {value_context}<extra></extra>",
            marker=dict(color=_STACKED_GAP_COLOR if col == _STACKED_GAP_LABEL else category_color(str(col)),
                        line=dict(color="white", width=1.5)),
        ))

    x_max = max(100.0, float(pct_df.sum(axis=1).max())) if len(pct_df) else 100.0
    fig.update_layout(
        barmode="stack",
        bargap=bargap,
        title=title,
        # Sem isto o Plotly ENCOLHE o texto até caber (segmento de 4% num
        # painel estreito virava um rótulo minúsculo e ilegível). Com
        # mode="hide", o rótulo que não cabe em 10px some -- o valor segue
        # no hover. Complementa o corte fixo de `_STACKED_MIN_LABEL_PCT`,
        # que não sabe a largura real do gráfico.
        uniformtext=dict(minsize=10, mode="hide"),
        xaxis=dict(title=value_axis_title, range=[0, x_max], ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        # traceorder="normal": em barra empilhada o Plotly inverte a ordem
        # da legenda por padrão, e ela ficava "Detrator, Neutro, Promotor"
        # -- o contrário da ordem dos segmentos. y=-0.35 (e não -0.2 como
        # nos outros tipos) porque aqui o título do eixo X ocupa a faixa
        # logo abaixo das barras e colava na legenda.
        legend=dict(
            title=legend_title, orientation="h", traceorder="normal",
            yanchor="top", y=-0.35, xanchor="center", x=0.5,
        ),
        margin=dict(b=130),
        height=max(340, 48 * len(cats) + 220),
    )
    return fig


def _build_chart(
    pct_df: pd.DataFrame,
    chart_type: str,
    title: str,
    value_axis_title: str,
    legend_title: str,
    category_label: str,
    value_context: str,
    bargap: float = 0.2,
    bargroupgap: float = 0.1,
) -> go.Figure:
    """
    Monta a figura pros três tipos de gráfico (Barras, Barra horizontal,
    Linha) num só lugar -- existe pra não duplicar esse branch em cada
    chamada (era assim antes, com %LINHA e %COLUNA cada um com sua
    própria cópia da lógica, e ficou fácil os dois divergirem por
    engano).

    `category_label` é o nome do que está no eixo de categorias (ex.:
    nome do stub); `value_context` é uma frase curta pro hover explicando
    o que o número representa (ex.: "desse grupo", "de quem respondeu
    isso").

    `bargap`/`bargroupgap` controlam largura/espaçamento de barra --
    ignorados pelo Plotly quando `chart_type` é "Linha" (não tem barra),
    não precisam de tratamento condicional aqui. Largura de barra em si
    não é exposta como parâmetro direto de propósito: o Plotly já deriva
    ela do espaço sobrando depois do gap, e deixar as duas configuráveis
    (largura fixa E gap) cria combinação que desalinha barra entre séries
    -- mexer só no gap já cobre "quero barra mais grossa/mais fina" sem
    esse risco.

    DUAS DECISÕES DE UX:

    1. Legenda embaixo do gráfico, horizontal, não à direita (padrão do
       Plotly) -- à direita, a legenda cria uma faixa vertical de largura
       fixa que sobra pouco espaço de fato pro gráfico quando ele já está
       dividido ao meio (dois gráficos lado a lado, %LINHA e %COLUNA) --
       com 5 categorias de escala, a legenda vertical podia ocupar mais
       largura que o próprio gráfico.
    2. Cor por categoria reconhecida (`metadata.category_color`) --
       verde/vermelho reforça visualmente a mesma ordem "melhor pro pior"
       que `sort_categories` já aplica nos dados, em vez da paleta
       categórica padrão do Plotly (cores sem relação de intensidade
       entre si, ex. azul/laranja/verde escolhidas só pra serem
       distintas). Categoria fora do vocabulário conhecido (a maioria das
       nominais) não recebe cor fixa, fica com a paleta padrão.
    """
    if chart_type == CHART_STACKED:
        return _build_stacked_chart(pct_df, title, value_axis_title, legend_title, category_label, value_context, bargap)

    fig = go.Figure()
    horizontal = chart_type == "Barra horizontal"
    for col in pct_df.columns:
        cats = pct_df.index.astype(str)
        vals = pct_df[col]
        color = category_color(str(col))
        if horizontal:
            hover = f"{col}<br>{category_label}: %{{y}}<br>%{{x:.1f}}% {value_context}<extra></extra>"
            fig.add_trace(go.Bar(
                name=str(col), y=cats, x=vals, orientation="h", hovertemplate=hover,
                marker_color=color,
            ))
        else:
            hover = f"{category_label}: %{{x}}<br>{col}<br>%{{y:.1f}}% {value_context}<extra></extra>"
            if chart_type == "Barras":
                fig.add_trace(go.Bar(name=str(col), x=cats, y=vals, hovertemplate=hover, marker_color=color))
            else:  # Linha
                fig.add_trace(go.Scatter(
                    name=str(col), x=cats, y=vals, mode="lines+markers", hovertemplate=hover,
                    line=dict(color=color) if color else None,
                    marker=dict(color=color) if color else None,
                ))

    layout = dict(
        barmode="group",
        bargap=bargap,
        bargroupgap=bargroupgap,
        title=title,
        legend=dict(
            title=legend_title, orientation="h",
            yanchor="top", y=-0.2, xanchor="center", x=0.5,
        ),
        margin=dict(b=100),  # espaço extra embaixo pra legenda não cortar
    )
    if horizontal:
        layout["xaxis_title"] = value_axis_title
    else:
        layout["yaxis_title"] = value_axis_title
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  ABA: CRUZAMENTO
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _question_years_cached(_data: pd.DataFrame, _meta: dict, stub_key: str, data_signature: tuple) -> tuple:
    """
    Anos (coluna ANO) em que a pergunta `stub_key` tem ao menos uma
    resposta, no banco INTEIRO (sem filtro). `data_signature` (shape do
    banco) é quem invalida o cache -- `_data`/`_meta` têm underscore pra o
    Streamlit não hashear o DataFrame de 119 mil linhas a cada chamada.

    POR QUE EXISTE: no banco 2026 há perguntas feitas só em 2026 (138) e
    perguntas que existiam e não foram feitas em 2026 (223). A base do
    %LINHA já está certa nesses casos (regra de universo), mas nada na
    tela dizia ao usuário que a pergunta só existe em alguns anos -- quem
    lê "Total" tende a achar que é o banco todo.

    Devolve () se o banco não tem coluna ANO.
    """
    if "ANO" not in _data.columns:
        return ()
    members = [m.name for m in _meta.values() if m.var_type == VarType.MR_OPTION and m.mr_group == stub_key]
    cols = members if members else [stub_key]
    # MR: opção marcada = True (bool); SR/indicador: qualquer valor não nulo.
    # `notna()` serve pros dois porque, no parquet, MR não marcada é False
    # em coluna bool -- por isso MR precisa de `.any()` sobre o valor, não sobre notna.
    if members:
        answered = _data[cols].fillna(False).astype(bool).any(axis=1)
    else:
        answered = _data[cols[0]].notna()
    years = answered.groupby(_data["ANO"].astype(str), observed=True).any()
    return tuple(sorted(y for y, ok in years.items() if ok))


def _availability_note(data: pd.DataFrame, meta: dict, stub_key: str) -> str | None:
    """
    Frase "pergunta feita só em X" quando a pergunta não cobre todos os
    anos do banco; None quando cobre todos (nada a avisar).
    """
    all_years = tuple(sorted(data["ANO"].astype(str).unique())) if "ANO" in data.columns else ()
    if not all_years:
        return None
    years = _question_years_cached(data, meta, stub_key, tuple(data.shape))
    if not years or len(years) == len(all_years):
        return None
    missing = [y for y in all_years if y not in years]
    return f"Pergunta aplicada apenas em {', '.join(years)} (não feita em {', '.join(missing)})."


def _filter_note_lines(active_filters: dict, meta: dict, max_valores: int | None = None) -> list[str]:
    """
    Uma linha de texto por filtro ativo (SQL primeiro, depois a cascata da
    barra lateral, na ordem em que foram aplicados). Vai pra tela, embaixo
    do banner, e pras Notas do CSV/Excel: o arquivo circula sem a tela, e
    sem essa linha ninguém sabe de que recorte saiu o número.

    `max_valores` corta listas longas na tela (ex.: 90 cidades mantidas);
    na exportação vai tudo (`None`).
    """
    linhas = []
    for key, valores in active_filters.items():
        if key == SQL_FILTER_KEY:
            linhas.append(f"Filtro SQL: {valores[0]}")
            continue
        label = meta[key].label if key in meta else key
        vals = [str(v) for v in valores]
        if max_valores is not None and len(vals) > max_valores:
            vals = vals[:max_valores] + [f"… (+{len(valores) - max_valores})"]
        linhas.append(f"Filtro de base — {label}: manter {', '.join(vals)}")
    return linhas


def _render_block_notes(blocks: list, availability: str | None = None, filtros: list[str] | None = None) -> None:
    """
    Legenda de base + avisos por variável de banner, acima da tabela --
    compartilhada pela aba Cruzamento e pelos painéis do Comparar (antes
    cada uma tinha a sua cópia, e a base do %LINHA não aparecia em lugar
    nenhum).

    Mostra, por banner: quantos responderam, contra QUAL base (a base do
    %LINHA do bloco, não o Total do stub) e, quando o universo da pergunta
    tirou gente da base, quem saiu. É essa frase que impede a leitura
    errada "as barras deveriam fechar 100% do ano inteiro": o ano inteiro
    inclui gente a quem a pergunta nunca foi feita.
    """
    for linha in filtros or []:
        st.caption(f"🔎 {linha}")
    if availability:
        st.caption(f"📅 {availability}")
    for b in blocks[1:]:
        own_n = int(b.base_n.sum())
        base = b.base_people
        pct = own_n / base * 100 if base else 0
        st.caption(f"**{b.banner_label}**: resposta de {own_n:,} de {base:,} ({pct:.1f}%) — {describe_block_base(b)}")
        if b.universe_excluded_groups and len(b.universe_excluded_groups) > 3:
            with st.expander(f"Ver todos os {len(b.universe_excluded_groups)} recortes fora da base de {b.banner_label}"):
                st.markdown("\n".join(f"- {name}: {n:,} pessoas" for name, n in b.universe_excluded_groups))
        if b.coverage_warning:
            st.warning(f"**{b.banner_label}**: {b.coverage_warning}")


def _export_notes(blocks: list, use_weighting: bool, availability: str | None = None, filtros: list[str] | None = None) -> list[str]:
    """Linhas de nota que vão junto do CSV/Excel -- os filtros aplicados, a base de cada %LINHA, a convenção de peso e, se houver, os anos em que a pergunta foi aplicada."""
    notes = list(filtros or [])
    notes += [availability] if availability else []
    notes += [f"{b.banner_label}: {describe_block_base(b)}" for b in blocks[1:]]
    notes.append("Peso amostral aplicado aos percentuais." if use_weighting else "Percentuais sem peso amostral (contagem bruta).")
    return notes


def _render_cruzamento_tab(
    data: pd.DataFrame,
    meta: dict,
    options: dict,
    filtered_data: pd.DataFrame,
    stub_key: str,
    banner_keys: list[str],
    na_handling: str,
    small_n_threshold: int,
    active_filters: dict,
    use_weighting: bool,
    use_universe: bool = True,
) -> None:
    """
    Desenha a aba de cruzamento inteira: tabela NA/%LINHA/%COLUNA,
    avisos de cobertura, legenda explicativa, gráficos (%LINHA e
    %COLUNA lado a lado) e botão de exportar CSV.

    `data` (sem filtro) só é usado pra montar a coluna "Total geral (sem
    filtro)" de comparação quando `active_filters` não está vazio --
    todo o resto do cálculo usa `filtered_data`.
    """
    if not banner_keys:
        st.warning("Selecione ao menos uma variável de banner na barra lateral.")
        return

    blocks = _build_banner_cached(filtered_data, meta, stub_key, tuple(banner_keys), na_handling, small_n_threshold, use_weighting, use_universe)
    if not blocks:
        st.warning(
            "Nenhuma das variáveis de banner selecionadas tem respondente elegível "
            "cruzado com o stub escolhido nesse conjunto de dados (ou nesse filtro de base)."
        )
        return

    table = format_banner_table_full(blocks)
    mask = small_n_mask_full(blocks)

    unfiltered_blocks = None
    if active_filters:
        # Total sem filtro nenhum -- só pra comparação lado a lado com o
        # Total já filtrado que o resto da tabela usa. banner_keys=[] faz
        # build_banner devolver só o bloco Total, sem cruzar nada. Mesma
        # `use_weighting` do lado filtrado -- as duas colunas precisam da
        # mesma convenção, senão a comparação lado a lado mistura ponderado
        # com não-ponderado silenciosamente.
        unfiltered_blocks = _build_banner_cached(data, meta, stub_key, (), na_handling, small_n_threshold, use_weighting)
        unfiltered_table = format_banner_table_full(unfiltered_blocks)
        unfiltered_mask = small_n_mask_full(unfiltered_blocks)
        # reindex pro índice de `table` (o lado FILTRADO) -- sem isso, pd.concat(axis=1)
        # faz union dos dois índices por padrão, e como o lado sem filtro tem TODAS as
        # categorias do stub (ex.: 13 UFs) contra só as que sobrevivem ao filtro (ex.: 2),
        # a tabela final herdava linha extra pras categorias filtradas fora, preenchida
        # com None -- exatamente o "CEARÁ aparece vazio" visto em produção. O filtro que
        # o usuário aplicou é quem decide quais linhas existem; "sem filtro" só empresta
        # os números delas, nunca acrescenta categoria nova.
        unfiltered_table = unfiltered_table.reindex(table.index)
        unfiltered_mask = unfiltered_mask.reindex(table.index)
        unfiltered_table.columns = pd.MultiIndex.from_tuples(
            [("Total geral (sem filtro)", c[1]) for c in unfiltered_table.columns]
        )
        unfiltered_mask.columns = unfiltered_table.columns
        table = pd.concat([unfiltered_table, table], axis=1)
        mask = pd.concat([unfiltered_mask, mask], axis=1)

    # --- legendas de cobertura por variável de banner ---
    _render_block_notes(blocks, _availability_note(data, meta, stub_key), _filter_note_lines(active_filters, meta, max_valores=10))

    st.subheader(f"{options[stub_key]}")
    if use_weighting:
        st.caption("⚖️ %LINHA e %COLUNA aplicando peso amostral (PESO) — NA continua contagem bruta.")

    with st.expander("❓ Como ler esses números"):
        st.markdown(
            "- **NA** — quantas pessoas de verdade estão nessa célula (contagem, não ponderada).\n"
            "- **%LINHA** — dentro dessa categoria do stub (a variável de linha, à esquerda), "
            "que % foi pra cada opção da pergunta. Responde \"como esse grupo se comporta\".\n"
            "- **%COLUNA** — o inverso: dentro dessa opção da pergunta, que % é de cada categoria "
            "do stub. Responde \"quem escolheu essa opção\" — o perfil, não o comportamento.\n"
            "- **Base Amostra** — o total de cada coluna, somando todas as categorias do stub.\n"
            "- **Total** — a base inteira do stub (todo mundo, independente da pergunta de banner "
            "escolhida). Nunca é a base específica de uma pergunta — pra isso, veja a legenda "
            "logo acima da tabela (\"resposta de X de Y\").\n"
            "- **Base do %LINHA** — contra quem o %LINHA é calculado. Por padrão é o *universo da "
            "pergunta*: grupos onde ela quase não tem resposta (ex.: um índice que não existe numa "
            "operação) saem da base, pra não contar como \"não respondeu\" quem nunca foi "
            "perguntado. A legenda acima da tabela diz quem saiu; a chave \"Base do %LINHA\" na "
            "barra lateral volta ao Total inteiro.\n"
            "- **Total geral (sem filtro)** — só aparece se você aplicou um filtro de base; é o "
            "Total de antes do filtro, pra comparação lado a lado.\n"
            "- **Células em amarelo** — a base ali é menor que o limiar definido na barra lateral; "
            "leia esse percentual com cautela, poucos respondentes sustentam esse número.\n"
            "- **Aviso amarelo acima da tabela** — sinaliza base concentrada numa categoria só do "
            "stub, ou cobertura baixa da pergunta (nem todo mundo respondeu) — não é erro de conta.\n"
            "- Se %LINHA de uma categoria não soma perto de 100%, é sinal de que nem todo mundo "
            "da base respondeu a pergunta (com o universo ligado, isso é não-resposta de verdade; "
            "com o Total inteiro, pode ser pergunta que nem existia pro grupo).\n"
            "- **%COLUNA pode somar mais de 100%** quando a variável de linha (stub) é de "
            "múltipla resposta: quem marcou 2+ opções é contado em 2+ categorias ao mesmo "
            "tempo, então a soma passa a refletir *menções*, não pessoas. Não é erro de conta "
            "— é o esperado sempre que o stub permite mais de uma resposta por pessoa.\n"
            "- **Gráfico de %LINHA** compara grupos entre si; **gráfico de %COLUNA** mostra o "
            "perfil de quem escolheu cada opção — geralmente %LINHA é a leitura mais direta."
        )

    def _highlight_small_n(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        for col in mask.columns:
            styles.loc[mask.index, col] = mask[col].map(
                lambda flagged: "background-color: #fff3cd; color: #000000" if flagged else ""
            )
        return styles

    # linha "NA" mostra contagem inteira; %LINHA/%COLUNA mostram 1 casa decimal
    na_rows = table.index[table.index.get_level_values(1) == "NA"]
    pct_rows = table.index[table.index.get_level_values(1) != "NA"]
    styled = (
        table.style
        .apply(_highlight_small_n, axis=None)
        .format("{:,.0f}", subset=pd.IndexSlice[na_rows, :])
        .format("{:.1f}", subset=pd.IndexSlice[pct_rows, :])
    )
    st.dataframe(styled, width='stretch')
    st.caption(
        "NA = contagem não ponderada · %LINHA = % dentro da categoria do stub · "
        "%COLUNA = % dentro da categoria do banner. Células em amarelo: base "
        "abaixo do limiar definido — leia o percentual com cautela."
    )
    linha_multiplicidade = _render_multiplicidade(meta, stub_key, blocks, unfiltered_blocks, table.columns)

    # --- gráficos: %LINHA e %COLUNA lado a lado, por variável de banner ---
    st.subheader("Gráficos")
    chart_type = st.radio(
        "Tipo de gráfico",
        options=CHART_TYPES,
        horizontal=True,
        help=(
            "Barra horizontal ajuda quando os nomes das categorias são longos "
            "(ex.: texto de pergunta aberta codificada) e ficam cortados no eixo "
            "vertical. Barra empilhada 100% coloca todas as opções do banner numa "
            "única barra por categoria do stub, com o % escrito dentro de cada "
            "pedaço -- existe só para %LINHA (é o que soma 100% por barra). "
            "Linha só faz sentido quando as categorias do stub têm "
            "ordem natural (ex.: Ano, faixa etária, escala de concordância) -- "
            "pra categorias sem ordem (ex.: Gênero, Região), a linha sugere uma "
            "tendência que não existe."
        ),
    )
    bargap, bargroupgap = 0.2, 0.1
    stacked = chart_type == CHART_STACKED
    if stacked:
        # Empilhada não tem "grupo" de barras lado a lado -- só o espaço
        # entre barras faz sentido, então o segundo slider some.
        bargap = st.slider(
            "Espaço entre barras", 0.0, 0.9, 0.2, 0.05,
            help="Maior = barras mais finas, mais espaço entre categorias do eixo.",
        )
        st.info("Barra empilhada 100% existe só para %LINHA — o gráfico de %COLUNA não é desenhado neste tipo.")
    elif chart_type in ("Barras", "Barra horizontal"):
        col_gap1, col_gap2 = st.columns(2)
        with col_gap1:
            bargap = st.slider(
                "Espaço entre grupos", 0.0, 0.9, 0.2, 0.05,
                help="Maior = barras mais finas, mais espaço entre categorias do eixo.",
            )
        with col_gap2:
            bargroupgap = st.slider(
                "Espaço dentro do grupo", 0.0, 0.9, 0.1, 0.05,
                help="Maior = mais separação entre as barras de um mesmo grupo (ex.: entre Ótimo/Bom/Regular lado a lado).",
            )
    for b in blocks[1:]:  # blocks[0] é sempre "Total" -- não faz gráfico próprio, já está implícito em cada um dos outros
        # %LINHA: "como esse grupo do stub se distribui nessa pergunta" --
        # leitura mais natural pra comparar grupos entre si. `linha_pct_frame`
        # é a MESMA função que monta a tabela e a exportação: o gráfico não
        # recalcula o número, só lê o que a tabela mostra.
        pct_linha = linha_pct_frame(b, blocks[0]).fillna(0.0)
        # %COLUNA: "quem compõe cada resposta" -- já vem pronto em b.pct.
        pct_coluna = b.pct

        if stacked:
            st_df, note, axis_title, ctx = _prepare_stacked(
                pct_linha, meta, b.banner_key, b.banner_label, options[stub_key],
            )
            st.plotly_chart(
                _build_chart(
                    st_df, chart_type,
                    title=f"{options[stub_key]} por {b.banner_label} — %LINHA",
                    value_axis_title=axis_title,
                    legend_title=b.banner_label,
                    category_label=options[stub_key],
                    value_context=ctx,
                    bargap=bargap, bargroupgap=bargroupgap,
                ),
                width='stretch',
            )
            if note:
                st.caption(note)
            continue

        col_linha, col_coluna = st.columns(2)

        with col_linha:
            fig1 = _build_chart(
                pct_linha, chart_type,
                title=f"{options[stub_key]} por {b.banner_label} — %LINHA",
                value_axis_title=f"% dentro de cada grupo de {options[stub_key]}",
                legend_title=b.banner_label,
                category_label=options[stub_key],
                value_context="desse grupo",
                bargap=bargap, bargroupgap=bargroupgap,
            )
            st.plotly_chart(fig1, width='stretch')
            st.caption("Como cada grupo se distribui nessa pergunta — compare grupos entre si.")

        with col_coluna:
            fig2 = _build_chart(
                pct_coluna, chart_type,
                title=f"{options[stub_key]} por {b.banner_label} — %COLUNA",
                value_axis_title="% de quem escolheu cada opção",
                legend_title=b.banner_label,
                category_label=options[stub_key],
                value_context="de quem respondeu isso",
                bargap=bargap, bargroupgap=bargroupgap,
            )
            st.plotly_chart(fig2, width='stretch')
            st.caption("Quem compõe cada resposta — o perfil de quem escolheu cada opção.")

    _render_export_controls(
        table, mask, stub_key, banner_keys, active_filters, na_handling, key_prefix="cruzamento",
        notes=_export_notes(blocks, use_weighting, _availability_note(data, meta, stub_key), _filter_note_lines(active_filters, meta))
        + ([linha_multiplicidade] if linha_multiplicidade else []),
        extra_signature=(use_weighting, use_universe),
    )


def _render_multiplicidade(meta: dict, stub_key: str, blocks: list, unfiltered_blocks: list | None, colunas) -> str | None:
    """
    Índice de multiplicidade (média de opções marcadas por respondente)
    embaixo do banner, quando o stub é de MÚLTIPLA ESCOLHA -- uma linha,
    alinhada às colunas da tabela. Devolve a mesma informação como uma
    linha de texto pras Notas da exportação (ou None se o stub não é MR).
    A conta é `mentions_per_respondent`: soma do %COLUNA ÷ 100, então
    bate com a tabela por construção.
    """
    if not _is_mr_key(meta, stub_key):
        return None
    valores = mentions_per_respondent(blocks)
    if unfiltered_blocks is not None:
        sem_filtro = mentions_per_respondent(unfiltered_blocks)
        sem_filtro.index = pd.MultiIndex.from_tuples([("Total geral (sem filtro)", c[1]) for c in sem_filtro.index])
        valores = pd.concat([sem_filtro, valores])
    valores.index = pd.MultiIndex.from_tuples(list(valores.index))
    # mesmas colunas da tabela do banner (que só tem uma coluna por categoria, não as 3 linhas NA/%)
    linha = valores.reindex(pd.MultiIndex.from_tuples(list(colunas)))
    st.markdown("**Índice de multiplicidade** — média de opções marcadas por respondente")
    st.dataframe(
        linha.to_frame("Média de menções por respondente").T.style.format("{:.2f}", na_rep="—"),
        width="stretch", placeholder="—",
    )
    st.caption(
        "Entre quem respondeu a pergunta (marcou pelo menos uma opção): total de opções marcadas ÷ pessoas. "
        "É a soma do %COLUNA ÷ 100. O banco não separa quem pulou a pergunta de quem não marcou nada, "
        "por isso o índice nunca fica abaixo de 1."
    )
    partes = [
        f"{c[1] if c[0] == c[1] else f'{c[0]} {c[1]}'} " + f"{v:.2f}".replace(".", ",")
        for c, v in linha.items() if pd.notna(v)
    ]
    return "Média de menções por respondente (índice de multiplicidade): " + " · ".join(partes)


def _render_export_controls(
    table: pd.DataFrame,
    mask: pd.DataFrame,
    stub_key: str,
    banner_keys: list[str],
    active_filters: dict,
    na_handling: str,
    key_prefix: str,
    notes: list[str] | None = None,
    extra_signature: tuple = (),
) -> None:
    """
    Fluxo de exportação (CSV/Excel) com o passo de "preparar antes de
    baixar" -- extraído de `_render_cruzamento_tab` pra ser chamado por
    mais de um lugar (modo Sozinho E cada painel do modo Comparação,
    ver `_render_comparacao_tab`) sem duplicar a lógica em si.

    `key_prefix` é OBRIGATÓRIO e é o motivo desta função existir como
    função, não como copiar-colar do bloco original: todo `key=` de
    widget aqui dentro é derivado dele (`f"{key_prefix}_export_ready"`
    etc.). Sem esse prefixo, dois painéis do modo Comparação (A e B)
    compartilhariam a MESMA chave de session_state -- apertar "preparar"
    no painel A marcaria o B como pronto também, e um download pisaria
    no outro. É o mesmo tipo de bug de `key=` colidindo que já foi
    corrigido no filtro em cascata da sidebar (ver comentário em
    `_render_sidebar`); a correção aqui é preventiva, não uma reação a
    um bug visto em produção.
    """
    col_csv, col_xlsx = st.columns(2)
    export_ready_key = f"{key_prefix}_export_ready"
    export_signature_key = f"{key_prefix}_export_signature"
    # assinatura simples do estado atual (filtro + stub + banner) -- se mudar
    # desde a última preparação, a exportação anterior fica obsoleta e
    # exigimos recomputar antes de liberar o download de novo.
    # `extra_signature` carrega as chaves que mudam o NÚMERO sem mudar
    # stub/banner/filtro (peso amostral, base do %LINHA) -- sem elas, ligar
    # o peso deixava a exportação antiga marcada como "pronta" e o download
    # entregava a tabela do estado anterior.
    current_signature = (stub_key, tuple(banner_keys), tuple(sorted(active_filters.items())), na_handling, extra_signature)
    if st.session_state.get(export_signature_key) != current_signature:
        st.session_state[export_ready_key] = False

    if not st.session_state.get(export_ready_key):
        st.button(
            "Preparar exportação (CSV/Excel)",
            key=f"{key_prefix}_prepare_export",
            help=(
                "Fixa a tabela atual antes de gerar os arquivos -- clicar direto em "
                "'Baixar' sem esse passo pode, em alguns casos, capturar um estado de "
                "filtro anterior ao seu ajuste mais recente (comportamento documentado "
                "do Streamlit em botões de download com dado calculado dinamicamente)."
            ),
        )
        if st.session_state.get(f"{key_prefix}_prepare_export"):
            st.session_state[export_ready_key] = True
            st.session_state[export_signature_key] = current_signature
            st.rerun()
        return

    csv_text = format_table_for_export(table).to_csv()
    if notes:
        # Notas depois de uma linha em branco, uma por linha: o CSV circula
        # sem a tela, então a base do %LINHA precisa viajar junto com os números.
        csv_text += "\n" + "\n".join(['"Notas"'] + ['"' + n.replace('"', "'") + '"' for n in notes]) + "\n"
    csv_bytes = csv_text.encode("utf-8")
    xlsx_bytes = banner_table_to_excel_bytes(table, mask, notes)
    base_name = _crosstab_file_name(stub_key, banner_keys)

    with col_csv:
        st.download_button(
            "Baixar banner (CSV)",
            data=csv_bytes,
            file_name=f"{base_name}.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_csv",
        )
    with col_xlsx:
        st.download_button(
            "Baixar banner (Excel)",
            data=xlsx_bytes,
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_download_xlsx",
        )


def _crosstab_file_name(stub_key: str, banner_keys: list[str]) -> str:
    """
    Nome de arquivo derivado do cruzamento em si (ex.: "REGIAO_por_GENERO"),
    em vez do "banner.csv" fixo de antes -- baixar vários cruzamentos
    seguidos não sobrescrevia um no outro silenciosamente na pasta de
    Downloads, mas também não dava pra saber qual era qual sem abrir cada
    um. Usa a CHAVE técnica da variável (ex. "REGIAO"), não o rótulo
    completo exibido na UI -- o rótulo pode ter espaço/acento/parênteses
    que precisariam de sanitização extra pra virar nome de arquivo válido
    em todo SO, e a chave já é um identificador limpo por natureza (é
    assim que a variável existe no banco).

    Com vários banners (ex. ["GENERO", "CLASSE"]), junta com "_e_" -- pra
    cruzamento de 4+ banners simultâneos, o nome ficaria muito longo, mas
    esse é um limite aceitável no PoC (não truncamos ativamente; revisar
    se aparecer nome de arquivo estourando limite de SO na prática).
    """
    banner_part = "_e_".join(banner_keys) if banner_keys else "Total"
    return f"{stub_key}_por_{banner_part}"


# ══════════════════════════════════════════════════════════════════════
#  ABA: COMPARAR (dois cruzamentos lado a lado)
# ══════════════════════════════════════════════════════════════════════

_PERSIST_KEY = "_cmp_panel_state"
# Dict `{persist_key: valor}` fora do namespace de qualquer widget --
# sobrevive independente de qual painel está sendo desenhado neste rerun.
# Ver `_persisted_widget_default`.


def _persisted_widget_default(persist_key: str, options: list, fallback_index: int = 0):
    """
    Índice inicial de um selectbox/multiselect que precisa sobreviver a
    ficar "escondido" (o painel não é desenhado naquele rerun). Existe
    porque `st.session_state[key]` de um widget SOME quando o widget não
    é renderizado num rerun -- não fica parado esperando, como uma
    variável Python comum ficaria (comportamento verificado empiricamente
    com `streamlit.testing.v1.AppTest`, ver ways-of-working: testar
    contra execução real, não assumir). Guardar o valor num dict PRÓPRIO
    em `session_state` (`_PERSIST_KEY`, fora do namespace de qualquer
    widget) é o que faz o valor sobreviver ao painel sumir e reaparecer.
    """
    saved = st.session_state.get(_PERSIST_KEY, {}).get(persist_key)
    if saved in options:
        return options.index(saved)
    return fallback_index


def _save_persisted(persist_key: str, value) -> None:
    """Grava o valor atual de um widget no dict de persistência entre reruns."""
    st.session_state.setdefault(_PERSIST_KEY, {})[persist_key] = value


def _render_comparison_panel(
    panel_label: str,
    key_prefix: str,
    data: pd.DataFrame,
    filtered_data: pd.DataFrame,
    meta: dict,
    options: dict,
    na_handling: str,
    small_n_threshold: int,
    active_filters: dict,
    use_weighting: bool,
    bargap: float,
    bargroupgap: float,
    use_universe: bool = True,
) -> None:
    """
    Um painel (A, B, C ou D) do modo Comparação -- stub/banner/tipo de
    gráfico são escolhidos aqui, independentes dos demais painéis; filtro
    de base e peso amostral vêm de fora (parâmetros `filtered_data`/
    `use_weighting`), porque foram decididos como COMPARTILHADOS entre
    todos os painéis, não configuráveis por painel (decisão de Ro: evita
    a sidebar duplicar filtro em cascata múltiplas vezes, e mantém as
    colunas comparáveis sobre a mesma base).

    Só UM gráfico é desenhado (o escolhido no selectbox), não os dois
    (%LINHA e %COLUNA) como no modo Sozinho -- é o corte que faz cada
    painel caber em 1/N da largura de tela sem virar rolagem infinita.
    A tabela numérica, em contraste, continua completa (NA/%LINHA/
    %COLUNA) por decisão explícita de Ro -- é a fonte auditável, o
    gráfico é só leitura rápida.

    `key_prefix` (ex.: "a", "b", "c", "d") evita todo widget e toda
    entrada de `session_state` deste painel colidir com a de outro --
    mesmo princípio de `_render_export_controls`.

    PERSISTÊNCIA AO ESCONDER O PAINEL (Ro: reduzir de 4 pra 2 painéis e
    voltar pra 4 deve trazer C/D como estavam, não resetados): usa
    `_persisted_widget_default`/`_save_persisted` em vez de confiar só em
    `key=` -- ver docstring de `_persisted_widget_default` pro motivo.
    """
    st.markdown(f"#### Painel {panel_label}")

    stub_options = list(options.keys())
    # `options` aqui já é a lista SEM as variáveis sem resposta na base
    # filtrada (ver `_render_sidebar`). Se o filtro mudou e a variável que
    # este painel tinha escolhida saiu da lista, o valor guardado no widget
    # deixa de ser uma opção válida -- mesma família do bug de estado do
    # filtro em cascata. Limpar a chave ANTES de desenhar faz o widget cair
    # no `index` calculado abaixo (primeira opção, pois o valor persistido
    # também não está mais em `stub_options`) em vez de quebrar.
    stub_widget_key = f"{key_prefix}_stub"
    gone_stub = st.session_state.get(stub_widget_key)
    if gone_stub is not None and gone_stub not in options:
        del st.session_state[stub_widget_key]
        st.caption(f"⚠️ '{gone_stub}' saiu da lista (sem resposta nessa base); linha trocada.")
    stub_key = st.selectbox(
        "Variável de linha (stub)",
        options=stub_options,
        format_func=lambda k: options[k],
        index=_persisted_widget_default(f"{key_prefix}_stub", stub_options),
        key=stub_widget_key,
    )
    _save_persisted(f"{key_prefix}_stub", stub_key)

    banner_options = [k for k in options if k != stub_key]
    saved_banner = st.session_state.get(_PERSIST_KEY, {}).get(f"{key_prefix}_banner", [])
    default_banner = [k for k in saved_banner if k in banner_options]
    banner_widget_key = f"{key_prefix}_banner"
    if banner_widget_key in st.session_state:
        # mesmo motivo do stub acima: valor do widget com opção que saiu da lista
        st.session_state[banner_widget_key] = [k for k in st.session_state[banner_widget_key] if k in banner_options]
    banner_keys = st.multiselect(
        "Variáveis de banner (colunas)",
        options=banner_options,
        format_func=lambda k: options[k],
        # `default=` junto de um valor já escrito no session_state gera aviso do
        # Streamlit (e o default seria ignorado de qualquer forma) -- só passa
        # default na primeira vez, quando o widget ainda não tem estado.
        default=None if banner_widget_key in st.session_state else default_banner,
        key=banner_widget_key,
    )
    _save_persisted(f"{key_prefix}_banner", banner_keys)

    # Dois seletores lado a lado (e não empilhados) pra não gastar mais
    # altura do painel -- a preocupação com poluição visual é justamente
    # o motivo de o modo Comparação mostrar só UM gráfico por vez.
    col_side, col_type = st.columns(2)

    chart_options = ["%LINHA", "%COLUNA"]
    with col_side:
        chart_side = st.selectbox(
            "📊 Gráfico",
            options=chart_options,
            index=_persisted_widget_default(f"{key_prefix}_chart_side", chart_options),
            key=f"{key_prefix}_chart_side",
            help=(
                "%LINHA compara os grupos do stub entre si (leitura mais direta pra "
                "comparação); %COLUNA mostra o perfil de quem escolheu cada opção. "
                "A tabela abaixo sempre mostra as duas, independente dessa escolha -- "
                "isso só decide qual vira gráfico."
            ),
        )
    _save_persisted(f"{key_prefix}_chart_side", chart_side)

    # Mesmas três opções da aba Cruzamento (`_render_cruzamento_tab`), pra
    # as duas abas oferecerem o mesmo repertório. "Barra horizontal" é o
    # equivalente em Plotly (`go.Bar(orientation="h")`) do `Axes.barh` do
    # matplotlib -- mesma figura, mas mantém hover e zoom e não exige uma
    # segunda biblioteca de gráfico. Antes, esta aba tinha "Barras" fixo.
    chart_type_options = CHART_TYPES
    with col_type:
        chart_type = st.selectbox(
            "Tipo",
            options=chart_type_options,
            index=_persisted_widget_default(f"{key_prefix}_chart_type", chart_type_options),
            key=f"{key_prefix}_chart_type",
            help=(
                "Barra horizontal ajuda quando os nomes das categorias são longos "
                "e ficam cortados no eixo vertical. Linha só faz sentido quando as "
                "categorias do stub têm ordem natural (ex.: Ano, faixa etária). "
                "Barra empilhada 100% só funciona com %LINHA."
            ),
        )
    _save_persisted(f"{key_prefix}_chart_type", chart_type)

    if not banner_keys:
        st.info("Selecione ao menos uma variável de banner pra este painel.")
        return

    blocks = _build_banner_cached(filtered_data, meta, stub_key, tuple(banner_keys), na_handling, small_n_threshold, use_weighting, use_universe)
    if not blocks:
        st.warning("Nenhum respondente elegível cruzado nesse conjunto de dados.")
        return

    table = format_banner_table_full(blocks)
    mask = small_n_mask_full(blocks)

    _render_block_notes(blocks, _availability_note(data, meta, stub_key), _filter_note_lines(active_filters, meta, max_valores=10))

    def _highlight_small_n(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        for col in mask.columns:
            styles.loc[mask.index, col] = mask[col].map(
                lambda flagged: "background-color: #fff3cd; color: #000000" if flagged else ""
            )
        return styles

    na_rows = table.index[table.index.get_level_values(1) == "NA"]
    pct_rows = table.index[table.index.get_level_values(1) != "NA"]
    styled = (
        table.style
        .apply(_highlight_small_n, axis=None)
        .format("{:,.0f}", subset=pd.IndexSlice[na_rows, :])
        .format("{:.1f}", subset=pd.IndexSlice[pct_rows, :])
    )
    # container com scroll próprio -- a alternativa (deixar a tabela livre)
    # empurra a largura da coluna do painel além do que `st.columns(2)`
    # reserva, e o layout dos dois painéis lado a lado quebra. Rolar uma
    # tabela dentro de uma caixa de altura fixa é preferível a isso.
    st.dataframe(styled, width='stretch', height=320)

    stacked_needs_linha = chart_type == CHART_STACKED and chart_side != "%LINHA"
    if stacked_needs_linha:
        # Em vez de cair em silêncio pra outro tipo (o analista acharia que
        # está vendo empilhado), explica e diz o que fazer.
        st.info("Barra empilhada 100% só existe para %LINHA — troque o seletor 📊 Gráfico para %LINHA.")
    for b in ([] if stacked_needs_linha else blocks[1:]):
        if chart_side == "%LINHA":
            pct_df = linha_pct_frame(b, blocks[0]).fillna(0.0)
            value_axis_title = f"% dentro de cada grupo de {options[stub_key]}"
            value_context = "desse grupo"
            stacked_note = None
            if chart_type == CHART_STACKED:
                pct_df, stacked_note, value_axis_title, value_context = _prepare_stacked(
                    pct_df, meta, b.banner_key, b.banner_label, options[stub_key],
                )
        else:
            pct_df = b.pct
            value_axis_title = "% de quem escolheu cada opção"
            value_context = "de quem respondeu isso"

        fig = _build_chart(
            pct_df, chart_type,
            title=f"{options[stub_key]} por {b.banner_label} — {chart_side}",
            value_axis_title=value_axis_title,
            legend_title=b.banner_label,
            category_label=options[stub_key],
            value_context=value_context,
            bargap=bargap, bargroupgap=bargroupgap,
        )
        st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_chart_{b.banner_key}")
        if chart_type == CHART_STACKED and stacked_note:
            st.caption(stacked_note)

    _render_export_controls(
        table, mask, stub_key, banner_keys, active_filters, na_handling, key_prefix=key_prefix,
        notes=_export_notes(blocks, use_weighting, _availability_note(data, meta, stub_key), _filter_note_lines(active_filters, meta)),
        extra_signature=(use_weighting, use_universe),
    )


def _render_comparacao_tab(
    data: pd.DataFrame,
    meta: dict,
    options: dict,
    filtered_data: pd.DataFrame,
    na_handling: str,
    small_n_threshold: int,
    active_filters: dict,
    use_weighting: bool,
    use_universe: bool = True,
) -> None:
    """
    De 2 a 4 cruzamentos independentes num grid 2x2, pra comparação
    visual direta -- a hipótese de UX sendo testada neste PoC (ver
    decisão com Ro: o problema real não é "caber mais informação no
    dashboard", é "comparar cortes sem depender de memória de curto
    prazo trocando de aba").

    DECISÕES DE ESCOPO, já fechadas com Ro antes deste código existir:
    - Quantidade de painéis é escolhida pelo analista (2/3/4).
    - GRID 2x2, não uma fileira única de `st.columns(n_panels)`: com 4
      painéis numa linha só, cada um herda 1/4 da largura -- pouco pra
      uma tabela banner (várias colunas de categoria por natureza), os
      rótulos truncam. Em 2 linhas de `st.columns(2)`, cada painel
      herda 1/2 da largura -- a mesma largura já validada como
      confortável com N=2 (ver `_grid_layout`).
    - N=2 -> uma linha só, `st.columns(2)`, sem grid (idêntico ao
      comportamento original antes de existir escolha de quantidade).
    - N=3 -> A e B na primeira linha; C sozinho na segunda linha, com a
      4ª vaga VAZIA ao lado dele (decisão de Ro: preferiu manter os 3
      painéis do mesmo tamanho a esticar C pra ocupar a linha toda).
    - N=4 -> 2 linhas de `st.columns(2)`, todas as 4 vagas preenchidas.
    - SEM slider de largura por painel (decisão explícita de Ro) -- os N
      painéis sempre dividem a largura em partes iguais dentro de cada
      linha. "Redimensionar arrastando" não existe em Streamlit puro
      (`st.columns` só aceita pesos definidos em código, não arraste de
      mouse); um slider de peso substituto foi avaliado e descartado --
      a confusão adicionada seria maior que o ganho.
    - Painel oculto (ex.: C/D quando n_panels=2) MANTÉM sua configuração
      -- persistido manualmente via `_PERSIST_KEY`/`_save_persisted`,
      NÃO só por `key=` do widget. Verificado com AppTest que
      `session_state[key]` de um widget some quando ele não é desenhado
      num rerun -- por isso o dict próprio existe, ver docstring de
      `_persisted_widget_default`. Voltar pra n_panels=4 faz C/D
      reaparecerem como estavam.
    - Filtro de base e peso amostral são COMPARTILHADOS entre todos os
      painéis (vêm de `filtered_data`/`use_weighting`, calculados uma vez
      só em `_render_sidebar`) -- só stub, banner e escolha de gráfico
      são independentes por painel.
    - Espaçamento de barra (`bargap`/`bargroupgap`) é compartilhado entre
      todos os painéis -- suposição minha, não pedido explícito de Ro;
      reavaliar se ele quiser gap independente por painel.
    - Tabela numérica sempre completa (NA/%LINHA/%COLUNA); só o GRÁFICO é
      cortado pra uma leitura por vez.
    - Exportação CSV/Excel mantida em cada painel (decisão de Ro: quem
      está comparando pode querer levar um dos lados pro relatório sem
      sair do modo comparação).
    """
    st.caption(
        "Cruzamentos lado a lado, sobre a mesma base filtrada e o mesmo peso amostral "
        "da barra lateral. Stub, banner e tipo de gráfico são escolhidos por painel."
    )

    n_panels = st.radio(
        "Quantidade de painéis",
        options=[2, 3, 4],
        horizontal=True,
        key="cmp_n_panels",
        help=(
            "A tabela banner tem várias colunas de categoria por natureza -- por isso o "
            "grid é sempre 2 colunas (nunca 4 numa fileira só), pra cada painel manter "
            "largura suficiente pra tabela não truncar. Reduzir a quantidade não apaga a "
            "configuração dos painéis escondidos -- eles reaparecem como estavam se você "
            "aumentar de novo."
        ),
    )

    bargap, bargroupgap = 0.2, 0.1
    with st.expander(f"Ajustar espaçamento de barra (aplicado aos {n_panels} painéis)"):
        col_gap1, col_gap2 = st.columns(2)
        with col_gap1:
            bargap = st.slider("Espaço entre grupos", 0.0, 0.9, 0.2, 0.05, key="cmp_bargap")
        with col_gap2:
            bargroupgap = st.slider("Espaço dentro do grupo", 0.0, 0.9, 0.1, 0.05, key="cmp_bargroupgap")

    panel_labels = ["A", "B", "C", "D"][:n_panels]
    # Grid 2 colunas, quantas linhas N exigir -- N=3 é o único caso com
    # vaga sobrando (3 não é múltiplo de 2); decisão de Ro: fica vazia,
    # não esticamos o painel ímpar pra ocupar a linha toda.
    for row_start in range(0, n_panels, 2):
        row_labels = panel_labels[row_start:row_start + 2]
        row_columns = st.columns(2)
        for label, col in zip(row_labels, row_columns):
            with col:
                _render_comparison_panel(
                    label, label.lower(), data, filtered_data, meta, options, na_handling,
                    small_n_threshold, active_filters, use_weighting, bargap, bargroupgap,
                    use_universe,
                )
        # vaga vazia de N=3: row_columns[1] existe mas não recebe painel
        # nenhum -- a coluna do Streamlit já foi criada e ocupa espaço
        # reservado, então o vazio aparece como espaço em branco mesmo,
        # não como coluna colapsada.


# ══════════════════════════════════════════════════════════════════════
#  ABA: ÍNDICES
# ══════════════════════════════════════════════════════════════════════
# Três visões independentes (visão geral, individual, quadrante), cada
# uma na sua própria função -- eram um único if/elif/else grande antes;
# separadas fica mais fácil achar/mudar uma visão sem reler as outras
# duas.

def _indices_overview_table(filtered_data, meta, indicator_names, media_map, segment_key, weights) -> pd.DataFrame:
    """
    Uma linha por índice, uma coluna por categoria do segmento, valor =
    média ponderada -- o mesmo dado do gráfico de "Visão geral", só que em
    grade em vez de linha, pra quem quer o número exato de cada célula
    sem precisar passar o mouse em cada ponto.
    """
    rows = {}
    for ind in indicator_names:
        media_col = media_map.get(ind)
        if not media_col:
            continue
        trend = compute_index_trend(filtered_data, meta, media_col, segment_key, weights)
        rows[meta[ind].label] = trend["media"]
    return pd.DataFrame(rows).T


def _render_indices_overview(filtered_data, meta, options, indicator_names, media_map, segment_key, weights, notes: list[str] | None = None) -> None:
    """Todos os índices juntos, mesma escala 1-5 -- alternável entre linha de tendência ("painel de saúde") e tabela de médias."""
    view = st.radio("Mostrar como", ["Gráfico", "Tabela"], horizontal=True, key="indices_overview_view")

    table = _indices_overview_table(filtered_data, meta, indicator_names, media_map, segment_key, weights)
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "Baixar índices (CSV)",
            data=(table.to_csv() + ("\n" + "\n".join(['"Notas"'] + ['"' + n.replace('"', "'") + '"' for n in notes]) + "\n" if notes else "")).encode("utf-8"),
            file_name=f"indices_por_{segment_key}.csv",
            mime="text/csv",
            key="download_indices_overview_csv",
        )
    with col_xlsx:
        st.download_button(
            "Baixar índices (Excel)",
            data=table_to_excel_bytes(table, sheet_name="Índices", default_format="#,##0.00", notes=notes),
            file_name=f"indices_por_{segment_key}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_indices_overview_xlsx",
        )

    if view == "Tabela":
        st.dataframe(table.style.format("{:.2f}", na_rep="—"), width='stretch')
        st.caption("Média ponderada (escala 1-5) por índice e categoria de " + options[segment_key] + ". '—' = sem dado suficiente nessa categoria.")
        return

    fig = go.Figure()
    for ind in indicator_names:
        media_col = media_map.get(ind)
        if not media_col:
            continue
        trend = compute_index_trend(filtered_data, meta, media_col, segment_key, weights)
        hover = (
            f"{ind}<br>{options[segment_key]}: %{{x}}<br>"
            "Média: %{y:.2f}<br>Cobertura: %{customdata[0]:.0f}% (n=%{customdata[1]})<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=trend.index.astype(str), y=trend["media"], name=ind, mode="lines+markers",
            customdata=trend[["cobertura_pct", "n"]].values, hovertemplate=hover,
        ))
    fig.update_layout(
        title=f"Todos os índices por {options[segment_key]}",
        yaxis_title="Média (escala 1-5)",
        legend_title="Índice",
    )
    st.plotly_chart(fig, width='stretch')


def _render_indices_individual(filtered_data, meta, options, indicator_names, media_map, segment_key, weights) -> None:
    """Um índice escolhido, com gráfico de tendência e tabela de média/cobertura/N por categoria do segmento."""
    selected = st.selectbox("Escolha o índice", options=indicator_names, format_func=lambda k: meta[k].label)
    media_col = media_map.get(selected)
    trend = compute_index_trend(filtered_data, meta, media_col, segment_key, weights)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend.index.astype(str), y=trend["media"], name="Média", mode="lines+markers",
        customdata=trend[["cobertura_pct", "n"]].values,
        hovertemplate=(
            f"{options[segment_key]}: %{{x}}<br>Média: %{{y:.2f}}<br>"
            "Cobertura: %{customdata[0]:.0f}% (n=%{customdata[1]})<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"{selected} por {options[segment_key]}",
        yaxis_title="Média (escala 1-5)",
    )
    st.plotly_chart(fig, width='stretch')

    display = trend.copy()
    display.columns = ["Média", "Cobertura %", "N"]
    st.dataframe(
        display.style.format({"Média": "{:.2f}", "Cobertura %": "{:.1f}", "N": "{:,.0f}"}),
        width='stretch',
    )
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "Baixar (CSV)",
            data=display.to_csv().encode("utf-8"),
            file_name=f"{selected}_por_{segment_key}.csv",
            mime="text/csv",
            key="download_indice_individual_csv",
        )
    with col_xlsx:
        st.download_button(
            "Baixar (Excel)",
            data=table_to_excel_bytes(
                display, sheet_name=selected,
                column_formats={"Média": "#,##0.00", "Cobertura %": "#,##0.0", "N": "#,##0"},
            ),
            file_name=f"{selected}_por_{segment_key}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_indice_individual_xlsx",
        )


def _render_indices_quadrant(filtered_data, meta, indicator_names, media_map, segment_key, weights) -> None:
    """
    Bolha por índice, dois eixos possíveis (escolhidos na tela):
    Tendência x Nível atual (reaproveita `compute_index_trend`, sem
    cálculo novo) ou Importância x Desempenho (correlação ponderada
    contra um índice de referência escolhível, padrão IM). Tamanho da
    bolha = cobertura -- ver `indices.compute_quadrant_data` pro cálculo
    completo e o porquê de cada convenção.
    """
    axis_mode = st.radio(
        "Eixos do quadrante",
        ["Tendência x Nível atual", "Importância x Desempenho"],
        horizontal=True,
    )
    reference_indicator = "IM"
    if axis_mode == "Importância x Desempenho":
        reference_indicator = st.selectbox(
            "Índice de referência (o que os outros tentam explicar)",
            options=indicator_names,
            index=indicator_names.index("IM") if "IM" in indicator_names else 0,
            format_func=lambda k: meta[k].label,
            help="Importância = correlação ponderada entre cada índice e esse de referência, pessoa por pessoa. Não é causa, é aproximação padrão de mercado.",
        )

    quad = compute_quadrant_data(filtered_data, meta, media_map, weights, wave_key=segment_key, reference_indicator=reference_indicator)

    if axis_mode == "Tendência x Nível atual":
        plot_df = quad.dropna(subset=["nivel", "tendencia"])
        x_col, y_col = "nivel", "tendencia"
        x_title, y_title = "Nível atual (média)", f"Tendência (variação desde a onda anterior de {segment_key})"
        y_zero = 0.0
    else:
        plot_df = quad.dropna(subset=["nivel", "importancia"])
        x_col, y_col = "nivel", "importancia"
        x_title, y_title = "Desempenho (nível atual)", f"Importância (correlação com {reference_indicator})"
        y_zero = plot_df["importancia"].median() if not plot_df.empty else 0.0

    if plot_df.empty:
        st.info(
            f"Não deu pra calcular nenhum ponto -- confere se '{segment_key}' tem pelo menos duas ondas "
            "na base filtrada (pra tendência) ou se o índice de referência tem dado suficiente "
            "(pra importância)."
        )
        return

    x_mid = plot_df[x_col].median()
    sizes = plot_df["n"].clip(lower=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df[x_col], y=plot_df[y_col], mode="markers+text",
        text=plot_df.index, textposition="top center",
        marker=dict(size=sizes, sizemode="area", sizeref=2. * sizes.max() / (40. ** 2), sizemin=6),
        customdata=plot_df[["cobertura_pct", "n"]].values,
        hovertemplate=(
            "%{text}<br>" + f"{x_title}: %{{x:.2f}}<br>{y_title}: %{{y:.2f}}<br>"
            "Cobertura: %{customdata[0]:.0f}% (n=%{customdata[1]})<extra></extra>"
        ),
    ))
    fig.add_vline(x=x_mid, line_dash="dash", line_color="gray")
    fig.add_hline(y=y_zero, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Quadrante de índices" + (f" (referência: {reference_indicator})" if axis_mode == "Importância x Desempenho" else ""),
        xaxis_title=x_title, yaxis_title=y_title,
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Tamanho da bolha = quantas pessoas responderam esse índice (cobertura) -- "
        "bolha pequena, leia a posição com mais cautela. Linhas tracejadas dividem "
        "os quadrantes pela mediana (nível/importância) ou por zero (tendência)."
    )
    if len(plot_df) < len(quad):
        faltando = [i for i in quad.index if i not in plot_df.index]
        st.caption(f"Fora do gráfico por falta de dado suficiente: {', '.join(faltando)}")


_LINHAS_CRUZADO_PADRAO = "CIDADES"
_MAX_DESTAQUES = 3
_MAX_LINHAS_COLORIDAS = 8  # até aqui, cada linha ganha uma cor; acima, cinza + até 3 destacadas


def _render_indices_cross(
    filtered_data: pd.DataFrame,
    meta: dict,
    options: dict,
    indicator_names: list[str],
    media_map: dict,
    segment_key: str,
    weights: pd.Series,
    small_n_threshold: int,
    notes: list[str] | None = None,
) -> None:
    """
    Visão "Duas variáveis": UM índice cruzado por duas variáveis ao mesmo
    tempo -- linhas à escolha (padrão CIDADES), colunas = "Segmentar por"
    (padrão ANO). Pedido do Ro (26/11) pra fazer no app o que o HTML das
    34 cidades PLENA fazia: cidade por ano.

    Tabela com Total (linha e coluna) e células de n pequeno em amarelo,
    como no banner. No gráfico, com muitas linhas (130 cidades) cada uma
    colorida vira um emaranhado; por isso, acima de
    `_MAX_LINHAS_COLORIDAS`, as linhas ficam cinza e o usuário destaca até
    3 -- o mesmo desenho do HTML.
    """
    linhas_opcoes = [k for k, m in meta.items() if m.var_type == VarType.SR and k in options and k != segment_key]
    if not linhas_opcoes:
        st.info("Não há outra variável de resposta única pra cruzar.")
        return
    col_i, col_l = st.columns(2)
    with col_i:
        selected = st.selectbox("Índice", options=indicator_names, format_func=lambda k: meta[k].label, key="indices_cross_indice")
    with col_l:
        padrao = linhas_opcoes.index(_LINHAS_CRUZADO_PADRAO) if _LINHAS_CRUZADO_PADRAO in linhas_opcoes else 0
        row_key = st.selectbox(
            "Linhas", options=linhas_opcoes, index=padrao, format_func=lambda k: options.get(k, k), key="indices_cross_linhas",
            help=f"As colunas são a variável de \"Segmentar por\", acima ({options.get(segment_key, segment_key)}).",
        )
    media_col = media_map[selected]
    cross = compute_index_cross(filtered_data, meta, media_col, row_key, segment_key, weights)
    if cross.empty:
        st.info("Nenhuma combinação com entrevista nessa base.")
        return

    # margens: Total por coluna (todas as linhas) e por linha (todas as colunas), mesma conta
    por_coluna = compute_index_trend(filtered_data, meta, media_col, segment_key, weights)
    por_linha = compute_index_trend(filtered_data, meta, media_col, row_key, weights)
    valores = media_values(filtered_data, media_col)
    ok = valores.notna()
    geral = float((valores[ok] * weights[ok]).sum() / weights[ok].sum()) if ok.any() else np.nan

    linhas = sort_categories([str(v) for v in cross.index.get_level_values(0).unique()])
    colunas = sort_categories([str(v) for v in cross.index.get_level_values(1).unique()])
    plano = cross.copy()
    plano.index = pd.MultiIndex.from_tuples([(str(a), str(b)) for a, b in plano.index])
    medias = plano["media"].unstack().reindex(index=linhas, columns=colunas)
    ns = plano["n"].unstack().reindex(index=linhas, columns=colunas)
    por_linha.index = por_linha.index.map(str)
    por_coluna.index = por_coluna.index.map(str)
    medias["Total"] = por_linha["media"].reindex(linhas)
    ns["Total"] = por_linha["n"].reindex(linhas)
    medias.loc["Total"] = list(por_coluna["media"].reindex(colunas)) + [geral]
    ns.loc["Total"] = list(por_coluna["n"].reindex(colunas)) + [int(ok.sum())]
    medias = medias.reindex(["Total"] + linhas)
    ns = ns.reindex(["Total"] + linhas)
    medias.index.name, medias.columns.name = options.get(row_key, row_key), options.get(segment_key, segment_key)

    regra = neutral_imputation(media_col)
    st.caption(
        f"**{meta[selected].label}** — média do índice (escala 1-5) por {options.get(row_key, row_key)} e "
        f"{options.get(segment_key, segment_key)}. Célula em amarelo = menos de {small_n_threshold} entrevistas "
        "(limiar da barra lateral). '—' = combinação sem entrevista."
        + (f" Regra do índice: quem não respondeu entra como {regra:g}." if regra is not None else "")
    )
    st.caption(
        "💡 O Total de cada coluna muda conforme as linhas que existem naquela coluna (ex.: cidades que entraram "
        "ou saíram num ano). Pra comparar ano contra ano sem esse efeito, aplique antes o filtro SQL das cidades "
        "presentes em todos os anos (está nos exemplos da aba 🔎 Filtro SQL)."
    )

    view = st.radio("Mostrar como", ["Gráfico", "Tabela"], horizontal=True, key="indices_cross_view")
    if view == "Tabela":
        pequeno = ns.lt(small_n_threshold) & ns.notna()

        def _amarelo(_: pd.DataFrame) -> pd.DataFrame:
            return pequeno.map(lambda f: "background-color: #fff3cd; color: #000000" if f else "")

        st.dataframe(medias.style.apply(_amarelo, axis=None).format("{:.2f}", na_rep="—"), width="stretch", placeholder="—")
        if st.checkbox("Mostrar o n de cada célula", key="indices_cross_mostrar_n"):
            st.dataframe(ns.style.format("{:,.0f}", na_rep="—"), width="stretch", placeholder="—")
    else:
        _render_indices_cross_chart(medias.drop(columns="Total"), ns.drop(columns="Total"), small_n_threshold,
                                    meta[selected].label, options.get(segment_key, segment_key))

    # exportação: grade das médias (Excel) e tabela longa com n e cobertura (CSV)
    notas = list(notes or []) + [
        f"Índice: {meta[selected].label}. Linhas: {options.get(row_key, row_key)}. Colunas: {options.get(segment_key, segment_key)}.",
        f"Células com menos de {small_n_threshold} entrevistas: leia com cautela.",
    ] + ([f"Regra do índice: quem não respondeu entra como {regra:g}."] if regra is not None else [])
    longa = cross.reset_index().rename(columns={"media": "média", "cobertura_pct": "cobertura (%)"})
    nome = f"{selected}_por_{row_key}_e_{segment_key}"
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "Baixar com n e cobertura (CSV)",
            data=(longa.to_csv(index=False) + "\n" + "\n".join(['"Notas"'] + ['"' + n.replace('"', "'") + '"' for n in notas]) + "\n").encode("utf-8"),
            file_name=f"{nome}.csv", mime="text/csv", key="download_indices_cross_csv",
        )
    with col_xlsx:
        st.download_button(
            "Baixar a tabela (Excel)",
            data=table_to_excel_bytes(medias, sheet_name="Índice", default_format="#,##0.00", notes=notas),
            file_name=f"{nome}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_indices_cross_xlsx",
        )


def _render_indices_cross_chart(medias: pd.DataFrame, ns: pd.DataFrame, small_n_threshold: int, titulo: str, eixo_x: str) -> None:
    """Uma linha por categoria das linhas ao longo das colunas; Total tracejado. Muitas linhas: cinza + até 3 destacadas."""
    paleta = _paleta_est()
    categorias = [r for r in medias.index if r != "Total"]
    colorir_todas = len(categorias) <= _MAX_LINHAS_COLORIDAS
    destaques: list[str] = []
    if not colorir_todas:
        destaques = st.multiselect(
            f"Destacar no gráfico (até {_MAX_DESTAQUES})", options=categorias, max_selections=_MAX_DESTAQUES,
            key="indices_cross_destaques", help="As outras ficam em cinza, de fundo, pra dar a referência sem embaralhar.",
        )
    cores = {c: paleta["serie"][i] for i, c in enumerate(categorias if colorir_todas else destaques)}
    x = list(medias.columns)
    fig = go.Figure()

    def _linha(nome: str, cor: str, largura: float, tracejado: bool = False, legenda: bool = True, por_cima: bool = True) -> None:
        y = medias.loc[nome].to_numpy(dtype=float)
        n = ns.loc[nome].to_numpy(dtype=float)
        # ponto vazado = célula com n pequeno (mesmo aviso do amarelo na tabela)
        simbolos = ["circle-open" if (not np.isnan(v) and v < small_n_threshold) else "circle" for v in n]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=nome, showlegend=legenda, connectgaps=False,
            line=dict(color=cor, width=largura, dash="dash" if tracejado else "solid"),
            marker=dict(size=7 if por_cima else 4, color=cor, symbol=simbolos),
            customdata=np.nan_to_num(n, nan=0).astype(int),
            hovertemplate=f"{nome} · " + "%{x}: %{y:.2f} (n=%{customdata})<extra></extra>",
            opacity=1.0 if por_cima else 0.9,
        ))

    if not colorir_todas:
        for c in categorias:
            if c not in destaques:
                _linha(c, paleta["contexto"], 1, legenda=False, por_cima=False)
    for c in (categorias if colorir_todas else destaques):
        _linha(c, cores[c], 2.5)
    _linha("Total", paleta["tinta"], 3, tracejado=True)
    fig.update_layout(title=titulo, height=480, xaxis=dict(type="category", title=eixo_x), yaxis_title="média (1-5)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"), margin=dict(t=90))
    st.plotly_chart(fig, width="stretch", key="indices_cross_fig")
    texto = "Total = tracejado. Ponto vazado = menos de " + f"{small_n_threshold} entrevistas naquela célula."
    if not colorir_todas:
        texto += f" Linhas em cinza = as outras {len(categorias) - len(destaques)} (passe o mouse pra ver o nome)."
    st.caption(texto)


def _render_indices_tab(
    data: pd.DataFrame,
    meta: dict,
    options: dict,
    filtered_data: pd.DataFrame,
    notes: list[str] | None = None,
    use_weighting: bool = True,
    small_n_threshold: int = 30,
) -> None:
    """
    Ponto de entrada da aba Índices: monta os controles comuns às quatro
    visões (segmento, seletor de visão) e despacha pra
    `_render_indices_overview`/`_individual`/`_cross`/`_quadrant` conforme
    escolhido.

    Peso: desde 26/11 segue a chave da barra lateral, como o resto do app
    (antes era sempre ponderado). Decisão do Ro, pra um índice não ter um
    número aqui e outro na aba Estatística com o peso desligado.
    """
    full_media_map = indicator_media_map(meta)
    if not full_media_map:
        st.info("Esse banco não tem variáveis de indicador (_c) com companion numérico (_media).")
        return
    # Mesma regra das listas de stub/banner: índice sem NENHUM valor na base
    # filtrada some (ex.: IACOM com OPERACAO = PPPs). Conta o `_media` CRU,
    # sem a imputação neutra de `indices._NEUTRAL_IMPUTATION` -- ela preenche
    # nota 3 pra todo mundo sem valor e faria o IACOM parecer ter dado em PPPs.
    media_map = {
        ind: col for ind, col in full_media_map.items()
        if pd.to_numeric(get_column_series(filtered_data, col), errors="coerce").notna().any()
    }
    hidden_indices = [i for i in full_media_map if i not in media_map]
    if hidden_indices:
        st.caption(f"🙈 Índices ocultos (nenhum valor nessa base): {', '.join(hidden_indices)}.")
    indicator_names = list(media_map.keys())
    if not indicator_names:
        st.info("Nenhum índice tem valor nessa base. Ajuste o filtro de base.")
        return

    segment_options = [k for k, m in meta.items() if m.var_type == VarType.SR and k in options]
    default_idx = segment_options.index("ANO") if "ANO" in segment_options else 0

    col1, col2 = st.columns([2, 1])
    with col1:
        segment_key = st.selectbox(
            "Segmentar por", options=segment_options, index=default_idx,
            format_func=lambda k: options.get(k, k),
            help="Só variáveis de resposta única -- segmentar índice por múltipla resposta não tem definição óbvia (a pessoa contaria em mais de um segmento ao mesmo tempo).",
        )
    with col2:
        view_mode = st.radio("Visualização", ["Visão geral", "Índice individual", "Duas variáveis", "Quadrante"], horizontal=True)

    weights = get_weights(filtered_data, meta, use_weighting)
    st.caption("⚖️ Médias ponderadas pelo PESO." if use_weighting
               else "Sem peso: médias simples (o peso está desligado na barra lateral).")

    with st.expander("❓ Como ler esse painel"):
        st.markdown(
            "- Cada índice (`IACOM`, `IMC`, `IM`...) é uma **média** -- ponderada pelo PESO quando o peso está ligado -- (não % de "
            "categoria) na escala 1-5, calculada a partir do companion numérico "
            "(`_media`) que fica por trás da versão categórica que aparece na aba "
            "Cruzamento.\n"
            "- **Cobertura** = % de quem, dentro daquele segmento, tem valor nesse "
            "índice. Indicadores de baixa incidência (ex.: avaliação de atendimento, "
            "quando a maioria nunca contatou a empresa) têm cobertura naturalmente "
            "baixa -- isso não é erro, é a pergunta não se aplicando a todo mundo.\n"
            "- Passe o mouse sobre um ponto pra ver a cobertura e o N junto com a média."
        )

    if view_mode == "Visão geral":
        _render_indices_overview(filtered_data, meta, options, indicator_names, media_map, segment_key, weights, notes)
    elif view_mode == "Índice individual":
        _render_indices_individual(filtered_data, meta, options, indicator_names, media_map, segment_key, weights)
    elif view_mode == "Duas variáveis":
        _render_indices_cross(filtered_data, meta, options, indicator_names, media_map, segment_key, weights, small_n_threshold, notes)
    else:  # Quadrante
        _render_indices_quadrant(filtered_data, meta, indicator_names, media_map, segment_key, weights)


# ══════════════════════════════════════════════════════════════════════
#  DADOS ORIGINAIS
# ══════════════════════════════════════════════════════════════════════

_INDICATOR_BIN_EDGES = [-float("inf"), 1.5, 2.5, 3.5, 4.5, float("inf")]
_INDICATOR_BIN_LABELS = [
    "Média 1 (<=1,4)", "Média 2 (>=1,5)", "Média 3 (>=2,5)",
    "Média 4 (>=3,5)", "Média 5 (>=4,5)",
]


def _categorical_counts(series: pd.Series) -> pd.Series:
    """Contagem bruta por categoria, ordenada pra leitura (não alfabética), com nulo como mais uma linha."""
    counts = series.value_counts(dropna=True)
    counts = counts.reindex(sort_categories(list(counts.index)))
    counts["(nulo)"] = int(series.isna().sum())
    return counts


def _binned_media_counts(series: pd.Series) -> pd.Series:
    """
    Contagem bruta por faixa de um companion `_media`, nos MESMOS cortes
    que geram a categórica `_c` irmã (<1,5 / 1,5-2,5 / 2,5-3,5 / 3,5-4,5
    / >=4,5) -- não é um corte novo, é o que o dado já usa. Serve de
    auditoria cruzada: comparado com `_categorical_counts` da
    categórica irmã, os números têm que bater linha a linha -- foi
    assim que o bug do ISDE apareceu (categórica com distribuição real,
    média 100% nula).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    binned = pd.cut(numeric, bins=_INDICATOR_BIN_EDGES, labels=_INDICATOR_BIN_LABELS, right=False)
    counts = binned.value_counts(dropna=True).reindex(_INDICATOR_BIN_LABELS, fill_value=0)
    counts["(nulo)"] = int(numeric.isna().sum())
    return counts


def _render_dados_tab(data: pd.DataFrame, meta: dict, filtered_data: pd.DataFrame) -> None:
    """
    Aba de DIAGNÓSTICO de dados brutos -- não é visualização, é "como
    estão os dados": nulo por coluna, linha duplicada, describe. Existe
    pra achar problema tipo o do ISDE (companion `_media` vazio apesar
    da categórica ter valor real) direto na tela, sem sair do app pra
    rodar script. Temporária de propósito -- por isso sem estilo, só a
    conta.

    Não usa `crossable_variables`/`options` pelo mesmo motivo da versão
    anterior desta aba: aquela lista some com WEIGHT/IDENTIFIER/
    SCALE_MEDIA, e é justamente o companion `_media` que mais precisa
    ser diagnosticado.
    """
    st.subheader("Diagnóstico de dados")

    usar_filtro = st.checkbox(
        "Aplicar o filtro de base (barra lateral e Filtro SQL)", value=True,
        help="Desmarque pra diagnosticar a base completa, ignorando o filtro em '1. Filtro de base' e o da aba Filtro SQL.",
    )
    base = filtered_data if usar_filtro else data
    st.caption(f"{len(base)} respondentes disponíveis nessa visão · {len(base.columns)} colunas no total.")

    label_por_coluna = {c: (meta[c].label if c in meta else c) for c in base.columns}
    colunas = st.multiselect(
        "Colunas pra diagnosticar",
        options=list(base.columns),
        format_func=lambda c: f"{c} — {label_por_coluna[c]}" if label_por_coluna[c] != c else c,
        help="Digite pra buscar por código (ex.: ISDE_media) ou por texto da pergunta.",
    )
    if not colunas:
        st.info("Escolha ao menos uma coluna acima -- ex.: uma categórica (`ISDE`) e sua `_media` companion juntas, pra comparar direto.")
        return

    subset = base[colunas]

    st.markdown("**Nulos por coluna**")
    nulos = subset.isnull().sum().rename("nulos")
    nulos_pct = (subset.isnull().mean() * 100).round(1).rename("nulos_%")
    st.dataframe(pd.concat([nulos, nulos_pct], axis=1), width='stretch')

    st.markdown("**Distribuição por coluna** (contagem bruta, não ponderada)")
    for col in colunas:
        m = meta.get(col)
        with st.expander(f"{col} — {label_por_coluna[col]}"):
            if m is not None and m.var_type == VarType.SCALE_MEDIA:
                st.dataframe(_binned_media_counts(base[col]).rename("contagem").to_frame(), width='stretch')
                sibling = m.scale_base
                if sibling and sibling in base.columns and sibling in meta and meta[sibling].var_type == VarType.INDICATOR:
                    st.caption(
                        f"Comparação com a categórica irmã '{sibling}' -- as faixas têm que bater linha a "
                        f"linha com as categorias 'Média próxima ou igual a N' abaixo, e '(nulo)' aqui "
                        f"corresponde à categoria 'N/A - ...' lá (são o mesmo grupo de gente, contado de "
                        f"dois jeitos diferentes). Se não bater, é sinal de inconsistência -- foi assim "
                        f"que o problema do ISDE apareceu."
                    )
                    st.dataframe(_categorical_counts(base[sibling]).rename(sibling).to_frame(), width='stretch')
            elif m is not None and m.var_type in (VarType.SR, VarType.MR_OPTION, VarType.INDICATOR):
                st.dataframe(_categorical_counts(base[col]).rename("contagem").to_frame(), width='stretch')
            else:
                st.caption("Coluna numérica sem indicador associado -- sem faixa aqui, só o describe() acima.")

    st.markdown("**Linhas duplicadas** (considerando só as colunas escolhidas acima)")
    st.write(f"{int(subset.duplicated().sum())} de {len(subset)} linhas")
    st.caption("Poucas colunas categóricas naturalmente geram muita 'duplicata' -- isso não é bug sozinho, é combinação de categoria se repetindo. Só desconfie se escolher colunas suficientes pra cada linha ser praticamente única (ex.: incluindo um ID).")

    st.markdown("**describe()**")
    st.dataframe(subset.describe(include="all").T, width='stretch')


    with st.expander("Ver linhas cruas (opcional)"):
        n_linhas = st.number_input(
            "Quantas linhas mostrar",
            min_value=1, max_value=max(1, min(len(base), 5000)), value=max(1, min(100, len(base))), step=50,
            help="Limitado de propósito -- mostrar as 100k+ linhas de uma vez pode estourar a memória do app.",
        )
        st.dataframe(subset.head(int(n_linhas)), width='stretch')


# ══════════════════════════════════════════════════════════════════════
#  ESTATÍSTICA (aba 📐 Estatística — parte Descritivas)
# ══════════════════════════════════════════════════════════════════════
# A conta mora em `stats_engine.py` (ponderada, testada em `tests/`); aqui
# fica a escolha do usuário e o desenho. Decisões do Ro (25/11): uma ou
# mais numéricas, quebra opcional por uma categórica, o usuário marca os
# cálculos e escolhe o gráfico -- e o app recomenda o gráfico certo pelo
# tipo da variável e explica quando a escolha não serve, porque quem usa
# nem sempre sabe qual estatística cabe em qual variável.

GRAFICO_RECOMENDADO = "Recomendado"
GRAFICO_BARRAS = "Barras de distribuição"
GRAFICO_HIST = "Histograma"
GRAFICO_BOX = "Box plot"
GRAFICO_DISPERSAO = "Dispersão"
_GRAFICOS_EST = [GRAFICO_RECOMENDADO, GRAFICO_BARRAS, GRAFICO_HIST, GRAFICO_BOX, GRAFICO_DISPERSAO]
_SEM_QUEBRA = "(sem quebra)"
_MAX_GRUPOS_GRAFICO = 12   # acima disso o gráfico vira ilegível; a tabela mostra todos
_MAX_GRUPOS_HIST = 6       # histograma por grupo = um painel por grupo
_PONTOS_DISPERSAO = 5000

# Paleta validada (skill dataviz, `references/palette.md`): categórica em
# ordem fixa e rampa azul sequencial, com versão própria pro tema escuro.
_PALETA_EST = {
    "light": {
        "serie": ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"],
        # rampa azul, passos 250 a 650 de 100 em 100: degraus iguais, pra nota 4 não se confundir com a 5
        # (mais claro que 250 some no fundo branco)
        "rampa": ["#84b9f3", "#519ced", "#257de0", "#1b64be", "#0f4a96"],
        "claras": 2,  # quantas cores do começo da rampa pedem texto escuro em cima
        "vao": "#ffffff",
        "contexto": "#cfcfcf",  # linhas de fundo (as cidades não destacadas)
        "tinta": "#191919",
    },
    "dark": {
        "serie": ["#338ced", "#e76624", "#00a888", "#d28e05", "#e24d8b", "#0aac23", "#9887ef", "#eb6263"],
        # no escuro a rampa inverte: valor maior = mais claro (mais contraste com o fundo); nada mais escuro que o passo 600
        "rampa": ["#1757ac", "#2370cd", "#338ced", "#6aabf1", "#9dc8f7"],
        "claras": -2,  # as 2 ÚLTIMAS cores são claras e pedem texto escuro
        "vao": "#0e1117",
        "contexto": "#3d3d3d",
        "tinta": "#efefef",
    },
}


def _paleta_est() -> dict:
    """Paleta do tema que o Streamlit está usando (claro/escuro); sem informação, a clara."""
    try:
        tipo = st.context.theme.type
    except Exception:
        tipo = None
    return _PALETA_EST["dark" if tipo == "dark" else "light"]


def _cor_texto(i: int, k: int, paleta: dict) -> str:
    """Texto escuro sobre as cores claras da rampa, branco sobre as escuras (contraste do rótulo dentro da barra)."""
    claras = paleta["claras"]
    clara = i < claras if claras > 0 else i >= k + claras
    return "#191919" if clara else "#ffffff"


def _rampa(k: int, paleta: dict) -> list[str]:
    """k cores da rampa sequencial, espaçadas por igual (valor menor -> primeira cor)."""
    rampa = paleta["rampa"]
    if k <= 1:
        return [rampa[len(rampa) // 2]]
    return [rampa[round(i * (len(rampa) - 1) / (k - 1))] for i in range(k)]


def _fmt_valor(v: float) -> str:
    return f"{v:.0f}" if float(v).is_integer() else f"{v:g}"


def _grafico_da_variavel(escolha: str, tipo: str, tem_grupo: bool, n_distintos: int) -> tuple[str, str | None]:
    """
    (gráfico que vai ser desenhado, aviso ou None). O usuário escolhe; o
    app desenha o que faz sentido pro tipo e diz o porquê quando troca.
    """
    recomendado = GRAFICO_BARRAS if tipo == se.TIPO_ESCALA else (GRAFICO_BOX if tem_grupo else GRAFICO_HIST)
    if escolha == GRAFICO_RECOMENDADO:
        return recomendado, None
    if escolha == GRAFICO_BARRAS and tipo == se.TIPO_CONTINUA:
        return recomendado, (f"Barras de distribuição servem pra escala; esta variável tem {n_distintos} valores "
                             f"diferentes e as barras ficariam ilegíveis. Mostrando o recomendado ({recomendado.lower()}).")
    if escolha == GRAFICO_HIST and tipo == se.TIPO_ESCALA:
        return GRAFICO_BARRAS, "Na escala, o histograma é o próprio gráfico de barras (uma barra por nota)."
    if escolha == GRAFICO_BOX and tipo == se.TIPO_ESCALA:
        return GRAFICO_BOX, ("Cuidado: na escala, a caixa costuma ir de 3 a 4 em quase todo grupo e esconde a "
                             "diferença (dos 15 índices, 10 têm a caixa igual). Compare pela média e pelas barras.")
    return escolha, None


def _grupos_do_grafico(tabela: pd.DataFrame, limite: int) -> list[str]:
    """Grupos que entram no gráfico: todos se couberem, senão os de maior n (a tabela continua com todos)."""
    grupos = [g for g in tabela.index if g != "Total"]
    if not grupos:
        return ["Total"]
    if len(grupos) <= limite:
        return grupos
    return list(tabela.loc[grupos, "n"].sort_values(ascending=False).index[:limite])


def _fig_barras(valores: pd.Series, pesos: pd.Series, grupos: pd.Series | None, nomes: list[str], titulo: str) -> go.Figure:
    paleta = _paleta_est()
    if grupos is None:
        dist = se.distribuicao(valores, pesos)
        fig = go.Figure(go.Bar(
            x=[_fmt_valor(v) for v in dist.index], y=dist.values, marker_color=paleta["serie"][0],
            text=[f"{p:.1f}%" for p in dist.values], textposition="outside",
            hovertemplate="valor %{x}: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(title=titulo, yaxis_title="% das entrevistas", xaxis_title="valor", height=380,
                          showlegend=False, xaxis_type="category")
        return fig
    # com quebra: uma barra 100% por grupo, nota baixa -> alta na rampa sequencial
    dists = {g: se.distribuicao(valores[(grupos == g).to_numpy()], pesos[(grupos == g).to_numpy()]) for g in nomes}
    todos = sorted({v for d in dists.values() for v in d.index})
    cores = _rampa(len(todos), paleta)
    fig = go.Figure()
    for i, (v, cor) in enumerate(zip(todos, cores)):
        pcts = [float(dists[g].get(v, 0.0)) for g in nomes]
        # cor do texto pela POSIÇÃO da cor na rampa usada (a rampa pode ter sido espaçada pra k notas)
        pos_na_rampa = paleta["rampa"].index(cor)
        fig.add_trace(go.Bar(
            y=nomes, x=pcts, name=_fmt_valor(v), orientation="h", marker_color=cor,
            marker_line=dict(width=1.5, color=paleta["vao"]),
            text=[f"{p:.0f}%" if p >= 6 else "" for p in pcts], textposition="inside", insidetextanchor="middle",
            insidetextfont=dict(color=_cor_texto(pos_na_rampa, len(paleta["rampa"]), paleta)),
            hovertemplate="%{y} · valor " + _fmt_valor(v) + ": %{x:.1f}%<extra></extra>",
        ))
    fig.update_layout(barmode="stack", title=titulo, xaxis_title="% das entrevistas", height=max(260, 140 + 40 * len(nomes)),
                      legend=dict(title_text="valor", orientation="h", traceorder="normal", y=1.02, yanchor="bottom", x=1, xanchor="right"),
                      yaxis=dict(autorange="reversed", type="category"), xaxis_range=[0, 100], margin=dict(t=90))
    return fig


def _fig_histograma(valores: pd.Series, pesos: pd.Series, grupos: pd.Series | None, nomes: list[str], titulo: str) -> go.Figure:
    from plotly.subplots import make_subplots

    paleta = _paleta_est()
    bordas, _ = se.histograma(valores, pesos)
    centros, larguras = (bordas[:-1] + bordas[1:]) / 2, np.diff(bordas)
    paineis = [("Total", None)] if grupos is None else [(g, (grupos == g).to_numpy()) for g in nomes]
    fig = make_subplots(rows=len(paineis), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[g for g, _ in paineis] if grupos is not None else None)
    for i, (g, mascara) in enumerate(paineis, start=1):
        x, w = se.limpar(valores if mascara is None else valores[mascara], pesos if mascara is None else pesos[mascara])
        soma, _ = np.histogram(x, bins=bordas, weights=w)
        pct = soma / w.sum() * 100 if len(x) else soma
        de, ate = bordas[:-1], bordas[1:]
        fig.add_trace(go.Bar(
            x=centros, y=pct, width=larguras, marker_color=paleta["serie"][0],
            marker_line=dict(width=1, color=paleta["vao"]), name=g, showlegend=False,
            customdata=np.column_stack([de, ate]),
            hovertemplate=f"{g} · " + "de %{customdata[0]:g} a %{customdata[1]:g}: %{y:.1f}%<extra></extra>",
        ), row=i, col=1)
    fig.update_yaxes(title_text="%", title_standoff=4)
    fig.update_layout(title=titulo, height=(380 if grupos is None else 70 + 150 * len(paineis)), bargap=0)
    return fig


def _fig_box(valores: pd.Series, pesos: pd.Series, grupos: pd.Series | None, nomes: list[str], titulo: str) -> go.Figure:
    paleta = _paleta_est()
    alvos = [("Total", None)] if grupos is None else [(g, (grupos == g).to_numpy()) for g in nomes]
    caixas = [(g, se.caixa(valores if m is None else valores[m], pesos if m is None else pesos[m])) for g, m in alvos]
    caixas = [(g, c) for g, c in caixas if c]
    fig = go.Figure(go.Box(
        x=[g for g, _ in caixas],
        q1=[c["q1"] for _, c in caixas], median=[c["mediana"] for _, c in caixas], q3=[c["q3"] for _, c in caixas],
        lowerfence=[c["bigode_inf"] for _, c in caixas], upperfence=[c["bigode_sup"] for _, c in caixas],
        mean=[c["media"] for _, c in caixas], boxmean=True, marker_color=paleta["serie"][0], name="", showlegend=False,
        hoverinfo="y",
    ))
    fora_x = [g for g, c in caixas for _ in c["fora"]]
    fora_y = [v for _, c in caixas for v in c["fora"]]
    if fora_y:
        fig.add_trace(go.Scatter(x=fora_x, y=fora_y, mode="markers", marker=dict(color=paleta["serie"][0], size=6, opacity=0.6),
                                 showlegend=False, hovertemplate="%{x}: %{y:g} (fora do bigode)<extra></extra>"))
    fig.update_layout(title=titulo, height=420, xaxis_type="category", yaxis_title="valor")
    return fig


def _fig_dispersao(serie_x: pd.Series, serie_y: pd.Series, grupos: pd.Series | None, nomes: list[str],
                   rotulo_x: str, rotulo_y: str, jitter_x: bool, jitter_y: bool) -> tuple[go.Figure, int, int]:
    """(figura, pontos desenhados, entrevistas com as duas respostas). Séries já lidas com a regra do índice."""
    paleta = _paleta_est()
    ambos = (serie_x.notna() & serie_y.notna()).to_numpy()
    pos = se.amostra_indices(int(ambos.sum()), _PONTOS_DISPERSAO)
    rng = np.random.default_rng(1)
    x = serie_x.to_numpy(dtype=float)[ambos][pos]
    y = serie_y.to_numpy(dtype=float)[ambos][pos]
    # escala tem poucos valores: sem tremer, milhares de pontos caem no mesmo lugar e parecem um só
    if jitter_x:
        x = x + rng.uniform(-0.18, 0.18, len(x))
    if jitter_y:
        y = y + rng.uniform(-0.18, 0.18, len(y))
    fig = go.Figure()
    g = None if grupos is None else grupos.loc[ambos].iloc[pos].astype("object").to_numpy()
    if g is not None and len(nomes) <= len(paleta["serie"]):
        for nome, cor in zip(nomes, paleta["serie"]):
            m = g == nome
            fig.add_trace(go.Scattergl(x=x[m], y=y[m], mode="markers", name=str(nome),
                                       marker=dict(color=cor, size=5, opacity=0.45)))
    else:
        fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", showlegend=False,
                                   marker=dict(color=paleta["serie"][0], size=5, opacity=0.35)))
    fig.update_layout(height=480, xaxis_title=rotulo_x, yaxis_title=rotulo_y,
                      title=f"{rotulo_y} x {rotulo_x}", legend_title_text="grupo" if g is not None else None)
    return fig, len(pos), int(ambos.sum())


def _tabela_estatisticas(tabela: pd.DataFrame, escolhidas: list[str]):
    """Só as colunas que o usuário marcou, na ordem de `ESTATISTICAS`, com rótulo e formato."""
    colunas: list[tuple[str, str]] = []
    for chave, rotulo in se.ESTATISTICAS:
        if chave not in escolhidas:
            continue
        if chave == "ic":
            colunas += [("ic_de", "IC 95% de"), ("ic_ate", "IC 95% até")]
        else:
            colunas.append((chave, rotulo))
    out = tabela[[c for c, _ in colunas]].rename(columns=dict(colunas))
    formatos = {r: ("{:,.0f}" if c in ("n", "n_efetivo") else "{:,.2f}") for c, r in colunas}
    return out.style.format(formatos, na_rep="—")


def _render_estatistica_tab(
    data: pd.DataFrame,
    meta: dict,
    options: dict,
    filtered_data: pd.DataFrame,
    use_weighting: bool,
    active_filters: dict,
) -> None:
    """
    Aba 📐 Estatística, parte Descritivas: variáveis numéricas (as `_media`),
    com quebra opcional por uma categórica. Usa a base já filtrada (SQL +
    barra lateral) e o mesmo peso do resto do app. Testes de hipótese são
    a próxima fase.
    """
    st.subheader("Estatística — descritivas das variáveis numéricas")
    st.caption("Testes de hipótese (qui-quadrado, t, z) entram na próxima fase.")

    numericas = [c for c in filtered_data.columns
                 if c in meta and meta[c].var_type == VarType.SCALE_MEDIA and filtered_data[c].notna().any()]
    if not numericas:
        st.info("Nenhuma variável numérica com dado nessa base.")
        return
    label = {c: meta[c].label for c in numericas}
    variaveis = st.multiselect(
        "Variáveis numéricas", options=numericas, key="est_vars",
        format_func=lambda c: f"{c} — {label[c]}" if label[c] != c else c,
        help="Digite pra buscar pelo código ou pelo texto da pergunta. As numéricas do banco são as colunas `_media`.",
        placeholder="ex.: F1_media (idade), IACOM_media, P50_media (nº de moradores)",
    )
    categoricas = [k for k in options if k in meta and meta[k].var_type in (VarType.SR, VarType.INDICATOR)]
    col_q, col_c = st.columns([1, 2])
    with col_q:
        quebra = st.selectbox("Quebrar por (opcional)", [_SEM_QUEBRA] + categoricas, key="est_quebra",
                              format_func=lambda k: k if k == _SEM_QUEBRA else options[k],
                              help="Uma linha por categoria na tabela, e o gráfico por grupo. Só resposta única e indicador.")
    with col_c:
        calculos = st.multiselect(
            "Cálculos", options=[k for k, _ in se.ESTATISTICAS], default=se.ESTATISTICAS_PADRAO, key="est_calc",
            format_func=lambda k: dict(se.ESTATISTICAS)[k],
        )
    grafico = st.radio(
        "Gráfico", _GRAFICOS_EST, horizontal=True, key="est_grafico",
        help=("Recomendado: barras pra escala; histograma pra contínua (box plot quando há quebra). "
              "Dispersão compara as DUAS primeiras variáveis escolhidas, uma contra a outra."),
    )

    peso_txt = ("⚖️ Ponderado pelo PESO (n e n efetivo são contagem de entrevistas)." if use_weighting
                else "Sem peso: contagem bruta (o peso está desligado na barra lateral).")
    st.caption(
        f"{peso_txt} Escala = até {se.LIMITE_ESCALA} valores diferentes (ex.: nota de 1 a 5); o resto é contínua. "
        "IC 95% = média ± t × erro padrão, com o peso considerado e (n efetivo − 1) graus de liberdade. "
        "Amostra por cota: erro padrão e IC são a convenção de mercado (como se fosse sorteio) e servem pra "
        "comparar a incerteza entre grupos, não como probabilidade de erro de verdade."
    )
    for linha in _filter_note_lines(active_filters, meta, max_valores=10):
        st.caption(f"🔎 {linha}")

    if not variaveis:
        st.info("Escolha ao menos uma variável numérica acima.")
        return
    if not calculos:
        st.info("Marque ao menos um cálculo.")
        return

    pesos = get_weights(filtered_data, meta, use_weighting)
    grupos = None
    ordem = None
    if quebra != _SEM_QUEBRA:
        grupos = get_column_series(filtered_data, quebra).astype("object")
        ordem = sort_categories([str(v) for v in pd.unique(grupos.dropna())])
        grupos = grupos.map(lambda v: str(v) if pd.notna(v) else None)

    for var in variaveis:
        # `media_values` = a mesma leitura da aba Índices (ex.: no IACOM, vazio vira 3) -- um índice, um número só
        valores = media_values(filtered_data, var)
        tipo = se.tipo_numerica(valores)
        n_distintos = int(valores.nunique())
        st.markdown(f"#### {var} — {label[var]}")
        regra = neutral_imputation(var)
        st.caption(f"Tipo: **{tipo}** ({n_distintos} valores diferentes)."
                   + (f" Regra do índice: quem não respondeu entra como {regra:g}, a mesma conta da aba Índices." if regra is not None else ""))
        tabela = se.descrever_por_grupo(valores, pesos, grupos, ordem_grupos=ordem)
        vazios = [g for g in tabela.index if g != "Total" and tabela.loc[g, "n"] == 0]
        if vazios:  # grupo sem nenhuma resposta = pergunta que não existiu ali (ex.: ano em que não foi feita)
            tabela = tabela.drop(index=vazios)
            st.caption(f"Sem resposta (fora da tabela e do gráfico): {', '.join(vazios)}.")
        st.dataframe(_tabela_estatisticas(tabela, calculos), width="stretch", placeholder="—")
        if grafico == GRAFICO_DISPERSAO:
            continue
        desenho, aviso = _grafico_da_variavel(grafico, tipo, grupos is not None, n_distintos)
        if aviso:
            st.caption(f"ℹ️ {aviso}")
        limite = _MAX_GRUPOS_HIST if desenho == GRAFICO_HIST else _MAX_GRUPOS_GRAFICO
        nomes = _grupos_do_grafico(tabela, limite)
        if grupos is not None and len(tabela) - 1 > len(nomes):
            st.caption(f"O gráfico mostra os {len(nomes)} grupos com mais entrevistas; a tabela mostra todos.")
        titulo = var if grupos is None else f"{var} por {options[quebra]}"
        if desenho == GRAFICO_BARRAS:
            fig = _fig_barras(valores, pesos, grupos, nomes, titulo)
        elif desenho == GRAFICO_HIST:
            fig = _fig_histograma(valores, pesos, grupos, nomes, titulo)
        else:
            fig = _fig_box(valores, pesos, grupos, nomes, titulo)
            st.caption("Linha cheia = mediana, tracejada = média; pontos = valores fora dos bigodes (1,5 × a altura da caixa).")
        st.plotly_chart(fig, width="stretch", key=f"est_fig_{var}")

    if grafico == GRAFICO_DISPERSAO:
        st.markdown("#### Dispersão")
        if len(variaveis) < 2:
            st.info("A dispersão compara duas variáveis: escolha pelo menos duas acima (a 1ª vai no eixo X, a 2ª no Y).")
            return
        vx, vy = variaveis[0], variaveis[1]
        nomes = []
        if grupos is not None:
            nomes = [g for g in ordem if g in set(grupos.dropna())]
        serie_x, serie_y = media_values(filtered_data, vx), media_values(filtered_data, vy)
        fig, n_pontos, n_ambos = _fig_dispersao(
            serie_x, serie_y, grupos, nomes, vx, vy,
            se.tipo_numerica(serie_x) == se.TIPO_ESCALA, se.tipo_numerica(serie_y) == se.TIPO_ESCALA,
        )
        st.plotly_chart(fig, width="stretch", key="est_fig_dispersao")
        notas = [f"Amostra aleatória de {_fmt_int(n_pontos)} de {_fmt_int(n_ambos)} entrevistas com as duas respostas "
                 "(com 100 mil pontos o gráfico vira uma mancha); cada ponto é uma entrevista, sem peso."]
        if grupos is not None and len(nomes) > len(_paleta_est()["serie"]):
            notas.append(f"Com {len(nomes)} grupos não dá pra separar por cor; todos os pontos estão na mesma cor.")
        if se.tipo_numerica(serie_x) == se.TIPO_ESCALA or se.tipo_numerica(serie_y) == se.TIPO_ESCALA:
            notas.append("Variável de escala: os pontos foram tremidos um pouco (±0,18) pra não caírem todos no mesmo lugar.")
        st.caption(" ".join(notas))


# ══════════════════════════════════════════════════════════════════════
#  FILTRO SQL
# ══════════════════════════════════════════════════════════════════════
# A conta (checar, rodar no DuckDB, traduzir erro) mora em `sql_filter.py`;
# aqui fica só a orquestração: guardar a condição aplicada, cachear a
# máscara, desenhar a aba e o aviso.
#
# ESTADO: a condição APLICADA fica em `st.session_state[_SQL_APLICADA]`,
# uma chave comum (não de widget). É ela que vale pro app -- não o texto
# da caixa, que pode estar pela metade. Por isso a barra lateral, que o
# Streamlit desenha ANTES das abas, já enxerga o filtro: ela lê essa
# chave no começo da rodada, e os botões Aplicar/Limpar mudam a chave
# num callback, que roda antes do script.
#
# ORDEM: o SQL corta primeiro, sobre o banco inteiro; a cascata da barra
# lateral parte do que sobrou (ver `sql_filter.py` pro porquê).

SQL_FILTER_KEY = "__sql__"  # chave reservada em `active_filters`; código de coluna nunca começa com "__"
_SQL_APLICADA = "sql_aplicada"
_SQL_TEXTO = "sql_texto"
_SQL_ERRO = "sql_erro"
_SQL_ERRO_DETALHE = "sql_erro_detalhe"
_SQL_COLUNAS = "sql_colunas"
_SQL_COLUNAS_PADRAO = ("ID_UNICO", "ANO", "OPERACAO", "CIDADES")
SQL_LINHAS_MAX = 1000  # decisão do Ro (25/11): de 1 a 1.000 linhas na tabela da aba, pra página não pesar

_SQL_EXEMPLOS = [
    ("Operação e anos", "OPERACAO = 'PLENA' AND ANO IN ('2024', '2025')"),
    ("Quem marcou uma opção de múltipla escolha", '"P46.1-Outros"'),
    ("Número: peso e média", '"P21.7_media" >= 4 AND PESO > 0.5'),
    ("Texto parecido (sem diferenciar maiúscula)", "lower(CIDADES) LIKE '%rio%'"),
    ("Sem resposta", "P25 IS NULL"),
    ("Cidades PLENA presentes nos 5 anos",
     "CIDADES IN (SELECT CIDADES FROM dados WHERE OPERACAO = 'PLENA'\n"
     "            GROUP BY CIDADES HAVING COUNT(DISTINCT ANO) = 5)\nAND OPERACAO = 'PLENA'"),
]


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _colunas_ocultas(meta: dict) -> tuple[str, ...]:
    """Colunas retiradas do app de propósito (VarType.EXCLUDED, ex.: IMCE/IMCT): fora da tabela `dados` do SQL."""
    return tuple(sorted(k for k, m in meta.items() if m.var_type == VarType.EXCLUDED))


@st.cache_data(show_spinner="Rodando o filtro SQL...", max_entries=32)
def _sql_mask_cached(data_path: str, condicao: str, colunas_ocultas: tuple, _data: pd.DataFrame):
    """
    Máscara da condição, cacheada pelo TEXTO da condição + caminho do banco.
    `_data` não é hasheado (sublinhado): hashear 119 mil x 1.600 colunas a
    cada rodada custaria mais que a própria consulta; o banco é função do
    caminho, que já está na chave (mesma premissa de `_load_and_classify`).
    """
    return calcular_mascara(_data, condicao, colunas_ocultas)


def _colunas_tabela_padrao(data: pd.DataFrame) -> list[str]:
    return [c for c in _SQL_COLUNAS_PADRAO if c in data.columns]


def _on_aplicar_sql(data: pd.DataFrame, meta: dict) -> None:
    """
    Callback do botão Aplicar. Só troca a condição aplicada se ela rodou
    E deixou alguém na base: com zero entrevistas a barra lateral pararia
    o app (`st.stop`) e a aba sumiria junto, sem jeito de desfazer. Em
    qualquer falha o filtro anterior continua valendo -- o aviso no topo
    mostra qual é.
    """
    st.session_state[_SQL_ERRO_DETALHE] = None
    cond = limpar_condicao(st.session_state.get(_SQL_TEXTO))
    if cond is None:
        _on_limpar_sql()
        return
    ocultas = _colunas_ocultas(meta)
    try:
        mask = _sql_mask_cached(str(DATA_PATH), cond, ocultas, data)
    except ErroFiltroSQL as exc:
        st.session_state[_SQL_ERRO] = str(exc)
        st.session_state[_SQL_ERRO_DETALHE] = exc.detalhe
        return
    except ImportError:
        st.session_state[_SQL_ERRO] = "O DuckDB não está instalado neste ambiente: rode `pixi install` (ou `pip install duckdb`)."
        return
    if not mask.any():
        st.session_state[_SQL_ERRO] = mensagem_sem_linhas(data, cond)
        return
    st.session_state[_SQL_APLICADA] = cond
    st.session_state[_SQL_ERRO] = None
    # já mostra na tabela da aba as colunas que a condição usa
    colunas = list(st.session_state.get(_SQL_COLUNAS) or _colunas_tabela_padrao(data))
    for c in colunas_citadas(cond, list(data.columns)):
        if c not in colunas and c not in ocultas:
            colunas.append(c)
    st.session_state[_SQL_COLUNAS] = colunas


def _on_limpar_sql() -> None:
    """Tira o filtro SQL do app. O texto da caixa fica, pra dar pra corrigir e aplicar de novo."""
    st.session_state[_SQL_APLICADA] = None
    st.session_state[_SQL_ERRO] = None
    st.session_state[_SQL_ERRO_DETALHE] = None


def _filtro_sql_atual(data: pd.DataFrame, meta: dict):
    """
    (condição aplicada, máscara) -- ou (None, None) sem filtro. Se a
    condição aplicada deixar de rodar (ex.: o banco foi trocado e a coluna
    sumiu), o filtro sai e a aba mostra por quê, em vez de quebrar o app.
    """
    cond = st.session_state.get(_SQL_APLICADA)
    if not cond:
        return None, None
    try:
        mask = _sql_mask_cached(str(DATA_PATH), cond, _colunas_ocultas(meta), data)
    except (ErroFiltroSQL, ImportError) as exc:
        _on_limpar_sql()
        st.session_state[_SQL_ERRO] = f"O filtro SQL que estava aplicado deixou de funcionar e foi retirado: {exc}"
        return None, None
    if not mask.any():
        _on_limpar_sql()
        st.session_state[_SQL_ERRO] = "O filtro SQL que estava aplicado não deixa mais ninguém na base e foi retirado."
        return None, None
    return cond, mask


def _render_aviso_sql(cond: str, n: int, total: int) -> None:
    """
    Aviso fixo no topo de todas as abas enquanto houver filtro SQL. Existe
    porque o filtro mora numa aba: sem isso, dava pra ir pro Cruzamento e
    ler número filtrado sem lembrar do filtro. Desenhado ANTES da barra
    lateral, pra o Limpar existir mesmo quando a cascata zera a base e o
    app para (`st.stop`).
    """
    col_txt, col_btn = st.columns([5, 1], vertical_alignment="center")
    with col_txt:
        st.info(f"🔎 **Filtro SQL ativo** — {_fmt_int(n)} de {_fmt_int(total)} entrevistas. Vale pro app inteiro "
                f"(aba 🔎 Filtro SQL); os filtros da barra lateral partem delas.")
        st.code(cond, language="sql", wrap_lines=True)
    with col_btn:
        st.button("Limpar filtro SQL", key="sql_limpar_topo", on_click=_on_limpar_sql, width="stretch")


def _tabela_busca_codigos(data: pd.DataFrame, meta: dict, busca: str, ocultas: tuple, limite: int = 50) -> pd.DataFrame:
    """Código | como escrever | pergunta | tipo | valores, das colunas que casam com a busca (código ou texto da pergunta, sem acento)."""
    tipo_legivel = {
        VarType.SR: "texto (resposta única)",
        VarType.INDICATOR: "texto (indicador)",
        VarType.MR_OPTION: "verdadeiro/falso (múltipla escolha)",
        VarType.SCALE_MEDIA: "número (média)",
        VarType.WEIGHT: "número (peso)",
        VarType.IDENTIFIER: "identificador",
    }
    # cada palavra da busca tem que aparecer (em qualquer ordem): "esgoto piorou" acha a P25
    termos = sem_acento_minusculo(busca).split()
    linhas = []
    for c in data.columns:
        if c in ocultas:
            continue
        label = meta[c].label if c in meta else c
        texto = sem_acento_minusculo(c) + " " + sem_acento_minusculo(label)
        if not all(t in texto for t in termos):
            continue
        linhas.append(c)
        if len(linhas) >= limite:
            break
    return pd.DataFrame([
        {
            "código": c,
            "como escrever": como_escrever(c),
            "pergunta": meta[c].label if c in meta else c,
            "tipo": tipo_legivel.get(meta[c].var_type, "") if c in meta else "",
            "valores": resumo_valores(data[c]),
        }
        for c in linhas
    ])


def _render_sql_tab(data: pd.DataFrame, meta: dict, filtered_data: pd.DataFrame, sql_mask) -> None:
    """
    Aba Filtro SQL: escrever a condição, achar o código da pergunta e ver a
    tabela. Decisões do Ro (25/11): é pra quem já sabe SQL (sem montador
    visual), mas erro em português; tabela com colunas escolhidas e de 1 a
    1.000 linhas; escolher se a tabela mostra a base final ou só o SQL.
    Com a caixa vazia, o app funciona como antes.
    """
    st.subheader("Filtro SQL")
    st.caption(
        "Escreva só a **condição** (o que viria depois do WHERE). A tabela se chama `dados` e cada coluna é o "
        "**código** da pergunta (busca logo abaixo). Texto vai entre aspas simples: `'PLENA'`; nome de coluna com "
        "ponto, hífen ou espaço vai entre aspas duplas: `\"P29.1\"`. ANO é texto: `ANO = '2025'`. "
        f"A condição roda sobre o banco inteiro e, ao clicar **Aplicar**, vale pro app todo. Limite: {LIMITE_SEGUNDOS} s."
    )
    # O widget some do session_state se o script parar antes de desenhá-lo
    # (ex.: `st.stop` da barra lateral); a condição aplicada não some, então
    # a caixa volta preenchida com ela.
    if _SQL_TEXTO not in st.session_state:
        st.session_state[_SQL_TEXTO] = st.session_state.get(_SQL_APLICADA) or ""
    st.text_area("Condição SQL", key=_SQL_TEXTO, height=130,
                 placeholder="OPERACAO = 'PLENA' AND ANO IN ('2024', '2025')")
    col_a, col_l, col_status = st.columns([1, 1, 4], vertical_alignment="center")
    with col_a:
        st.button("Aplicar", type="primary", key="sql_aplicar", on_click=_on_aplicar_sql, args=(data, meta), width="stretch")
    with col_l:
        st.button("Limpar filtro", key="sql_limpar", on_click=_on_limpar_sql, width="stretch")
    aplicada = st.session_state.get(_SQL_APLICADA)
    with col_status:
        if aplicada and sql_mask is not None:
            st.success(f"Filtro ativo: {_fmt_int(int(sql_mask.sum()))} de {_fmt_int(len(data))} entrevistas.")
        else:
            st.caption("Sem filtro SQL: o app usa o banco inteiro (mais os filtros da barra lateral).")
    erro = st.session_state.get(_SQL_ERRO)
    if erro:
        st.error(erro + (" O filtro anterior continua valendo." if aplicada else ""))
        if st.session_state.get(_SQL_ERRO_DETALHE):
            with st.expander("Mensagem original do DuckDB"):
                st.code(st.session_state[_SQL_ERRO_DETALHE], language=None, wrap_lines=True)
    if aplicada and limpar_condicao(st.session_state.get(_SQL_TEXTO)) != aplicada:
        st.caption("✏️ O texto da caixa mudou e ainda não foi aplicado; o que vale é a condição do aviso no topo.")

    with st.expander("Exemplos (copie pelo ícone no canto do código)"):
        for titulo, exemplo in _SQL_EXEMPLOS:
            st.markdown(f"**{titulo}**")
            st.code(exemplo, language="sql")
        st.caption("Múltipla escolha é verdadeiro/falso: `NOT \"P46.1-Outros\"` inclui também quem nunca recebeu a pergunta.")

    ocultas = _colunas_ocultas(meta)
    with st.expander("Procurar o código da pergunta"):
        busca = st.text_input("Parte do código ou do texto da pergunta", key="sql_busca",
                              placeholder="ex.: esgoto, P46.1, concessionaria")
        tabela = _tabela_busca_codigos(data, meta, busca, ocultas)
        if tabela.empty:
            st.caption("Nenhuma coluna com esse texto.")
        else:
            st.caption("Até 50 resultados. Use a coluna **como escrever** na condição (dá pra selecionar e copiar).")
            st.dataframe(tabela, hide_index=True, width="stretch",
                         column_config={"pergunta": st.column_config.TextColumn(width="large")})

    st.markdown("**Tabela**")
    base_final = st.checkbox(
        "Somar os filtros da barra lateral (base final, a que o cruzamento usa)", value=True, key="sql_base_final",
        help="Só muda o que ESTA tabela mostra. O app usa sempre o filtro SQL e os da barra lateral juntos.",
    )
    if base_final:
        base = filtered_data
    else:
        base = data[sql_mask] if sql_mask is not None else data
    opcoes = [c for c in data.columns if c not in ocultas]
    if _SQL_COLUNAS not in st.session_state:
        st.session_state[_SQL_COLUNAS] = _colunas_tabela_padrao(data)
    else:  # coluna que saiu do banco (troca de banco com o app aberto) derrubaria o multiselect
        st.session_state[_SQL_COLUNAS] = [c for c in st.session_state[_SQL_COLUNAS] if c in opcoes]
    label_por_coluna = {c: (meta[c].label if c in meta else c) for c in opcoes}
    col_cols, col_n = st.columns([4, 1])
    with col_cols:
        colunas = st.multiselect(
            "Colunas", options=opcoes, key=_SQL_COLUNAS,
            format_func=lambda c: f"{c} — {label_por_coluna[c]}" if label_por_coluna[c] != c else c,
            help="Digite pra buscar por código ou pelo texto da pergunta.",
        )
    with col_n:
        n_linhas = st.number_input("Linhas", min_value=1, max_value=SQL_LINHAS_MAX, value=SQL_LINHAS_MAX, step=50, key="sql_n_linhas")
    if not colunas:
        st.info("Escolha ao menos uma coluna.")
        return
    st.caption(f"Mostrando {_fmt_int(min(int(n_linhas), len(base)))} de {_fmt_int(len(base))} entrevistas "
               f"({'base final' if base_final else 'só o filtro SQL'}). Passe o mouse no nome da coluna pra ver a pergunta.")
    st.dataframe(
        base[colunas].head(int(n_linhas)),
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.Column(c, help=label_por_coluna[c]) for c in colunas},
    )


# ══════════════════════════════════════════════════════════════════════
#  BARRA LATERAL
# ══════════════════════════════════════════════════════════════════════

def _render_sidebar(
    data: pd.DataFrame,
    meta: dict,
    options: dict,
    sql_condition: str | None = None,
    sql_base: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, list[str], str, int, dict, bool, bool, dict]:
    """
    Monta a barra lateral inteira (filtro de base, seleção de stub/
    banner, regras) e já aplica o filtro escolhido, devolvendo tudo que
    `main()` precisa pra chamar as duas abas:

        (filtered_data, stub_key, banner_keys, na_handling, small_n_threshold,
         active_filters, use_weighting, use_universe, visible_options)

    `visible_options` é `options` sem as variáveis que não têm NENHUMA
    resposta na base já filtrada -- é o que alimenta as listas de stub e
    banner (aqui e nas abas), pra ninguém escolher uma pergunta que não
    existe no recorte (ex.: IACOM com OPERACAO = PPPs). O seletor do
    próprio filtro de base continua com a lista completa: filtrar por uma
    variável não depende de ela ter resposta no recorte anterior (a
    cascata já mostra só os valores que sobraram).

    O filtro de base é aplicado ANTES de qualquer cruzamento -- é só um
    recorte de linhas do df. Nada em `crosstab_engine.py` precisa saber
    que um filtro existe; ele só vê um df menor.

    Filtro SQL (aba Filtro SQL): quando existe, `sql_base` é o banco já
    cortado por ele e a cascata parte dali -- o SQL é o primeiro filtro.
    Ele entra em `active_filters` com a chave reservada `SQL_FILTER_KEY`,
    e por isso vale em tudo que já usa `active_filters`: a assinatura do
    cache de variáveis visíveis, a da exportação, a coluna "Total geral
    (sem filtro)" e as notas.
    """
    with st.sidebar:
        st.header("1. Filtro de base (opcional)")
        if sql_condition:
            st.caption(f"🔎 **Filtro SQL ativo**: {_fmt_int(len(sql_base))} de {_fmt_int(len(data))} entrevistas "
                       f"(aba 🔎 Filtro SQL). Os filtros abaixo partem delas.")
            st.button("Limpar filtro SQL", key="sql_limpar_lateral", on_click=_on_limpar_sql)
        filter_keys = st.multiselect(
            "Restringir a base a...",
            options=list(options.keys()),
            format_func=lambda k: options[k],
            help=(
                "Filtro por bloco de múltipla resposta ainda não é suportado -- "
                "só variáveis de resposta única ou indicador."
            ),
        )

        if filter_keys:
            st.caption(
                "Filtros em cascata: cada um abaixo já considera só as linhas que "
                "sobraram do filtro anterior, na ordem em que aparecem aqui."
            )

        active_filters: dict[str, list[str]] = {SQL_FILTER_KEY: [sql_condition]} if sql_condition else {}
        # encolhe a cada iteração -- é o que faz a cascata funcionar; com filtro SQL, já começa cortado por ele
        cascading_data = sql_base if (sql_condition and sql_base is not None) else data
        for fk in filter_keys:
            fmeta = meta.get(fk)
            if fmeta is None or fmeta.var_type not in (VarType.SR, VarType.INDICATOR):
                st.warning(f"'{options[fk]}' é múltipla resposta -- filtro ignorado nessa versão.")
                continue
            col = get_column_series(cascading_data, fmeta.name)
            available = sorted(v for v in col.dropna().unique().tolist())

            prev_key = f"filter_{fk}"
            prev_choice = st.session_state.get(prev_key)
            # Bug real encontrado em produção: Streamlit documenta que "if you use a key
            # and change a widget's default value, there will be no change to the widget's
            # state" -- ou seja, com key= já presente no session_state, mudar `default=`
            # (que é o que a cascata faz a cada rerun, conforme `available` encolhe) NÃO
            # reseta o valor interno do widget. A tela pode mostrar só os chips que ainda
            # cabem em `available`, mas o valor devolvido por st.multiselect() pode
            # carregar categoria de uma seleção anterior que já não é mais válida --
            # exatamente o "filtro não corta" visto em produção. Corrigimos escrevendo o
            # session_state manualmente ANTES de desenhar o widget, e nunca passando
            # `default=` junto de `key=` (a causa raiz do comportamento documentado acima).
            if prev_choice is None:
                st.session_state[prev_key] = available
            else:
                dropped = [v for v in prev_choice if v not in available]
                if dropped:
                    st.caption(
                        f"Em '{options[fk]}': {', '.join(str(v) for v in dropped)} não "
                        f"sobrevive(m) ao(s) filtro(s) anterior(es) e foi(ram) removido(s) da seleção."
                    )
                    st.session_state[prev_key] = [v for v in prev_choice if v in available]

            chosen = st.multiselect(f"Manter em '{options[fk]}'", options=available, key=prev_key)
            active_filters[fk] = chosen

            if not available:
                st.warning(f"'{options[fk]}' não tem nenhum valor restante após os filtros anteriores.")
                continue
            cascading_data = cascading_data[col.isin(chosen)] if chosen else cascading_data.iloc[0:0]

        st.header("2. Cruzamento")
        # Só variáveis com resposta na base filtrada. `cascading_data` já é a
        # base depois de todos os filtros; a assinatura (filtros em ordem)
        # é a chave do cache de `_visible_keys_cached`.
        if cascading_data.empty:
            st.warning("O filtro de base zerou a amostra. Ajuste as seleções na barra lateral.")
            st.stop()
        filters_signature = tuple((fk, tuple(map(str, chosen))) for fk, chosen in active_filters.items())
        visible_set = set(_visible_keys_cached(filters_signature, cascading_data, meta, tuple(options.keys())))
        visible_options = {k: label for k, label in options.items() if k in visible_set}
        hidden = [k for k in options if k not in visible_set]
        if hidden:
            st.caption(f"🙈 {len(hidden)} variáveis ocultas: nenhuma resposta nessa base.")
            with st.expander("Ver quais"):
                st.markdown("\n".join(f"- {options[k]}" for k in hidden))
        if not visible_options:
            st.warning("Nenhuma variável tem resposta nessa base. Ajuste o filtro de base.")
            st.stop()

        # `key=` explícito + correção do estado ANTES de desenhar. Sem isso, cada
        # mudança de filtro que escondesse/mostrasse uma variável mudava a
        # lista de opções -- e o Streamlit trata widget sem key com opções
        # diferentes como OUTRO widget, voltando stub pra primeira variável e
        # esvaziando o banner a cada clique no filtro. Com key, o valor
        # sobrevive; o que precisa de cuidado é o valor que saiu da lista
        # (inválido), tratado aqui explicitamente e avisado ao analista.
        stub_options = list(visible_options.keys())
        prev_stub = st.session_state.get("sb_stub")
        if prev_stub is not None and prev_stub not in visible_options:
            st.session_state["sb_stub"] = stub_options[0]
            st.caption(f"⚠️ '{options.get(prev_stub, prev_stub)}' não tem resposta nessa base; linha trocada para '{options[stub_options[0]]}'.")
        stub_key = st.selectbox(
            "Variável de linha (stub)",
            options=stub_options,
            format_func=lambda k: visible_options[k],
            key="sb_stub",
        )
        banner_options = [k for k in visible_options if k != stub_key]
        prev_banner = st.session_state.get("sb_banner")
        if prev_banner is not None:
            kept = [k for k in prev_banner if k in banner_options]
            if kept != prev_banner:
                lost = [options.get(k, k) for k in prev_banner if k not in banner_options]
                st.session_state["sb_banner"] = kept
                st.caption(f"⚠️ Saiu(ram) do banner (sem resposta nessa base, ou virou o stub): {', '.join(lost)}.")
        banner_keys = st.multiselect(
            "Variáveis de banner (colunas)",
            options=banner_options,
            format_func=lambda k: visible_options[k],
            key="sb_banner",
        )

        st.header("3. Regras")
        use_weighting = st.toggle(
            "Aplicar peso amostral (PESO)",
            value=True,
            help=(
                "Ligado por padrão. Desligado, o app inteiro (%LINHA/%COLUNA, Índices, "
                "Estatística) usa contagem bruta, sem peso amostral. O peso amostral "
                "é calibrado pra representar o universo TOTAL do estudo; dentro de um "
                "recorte já filtrado (ex.: uma diretoria específica), reaplicar esse "
                "mesmo peso pode inflar/distorcer a proporção entre categorias, porque "
                "ele não foi recalibrado pra esse subgrupo menor. Ligue só se você "
                "sabe que precisa do peso mesmo dentro do recorte atual -- ex.: "
                "comparando o recorte contra o total nacional lado a lado."
            ),
        )
        use_universe = st.toggle(
            "Base do %LINHA = universo da pergunta",
            value=True,
            help=(
                "Ligado (padrão): o %LINHA de cada banner divide só por quem estava num grupo "
                "onde aquela pergunta existe. Exemplo: o índice IACOM não existe em PPPs; com "
                "isso ligado, os ~20 mil de PPPs saem do denominador e as barras de cada ano "
                "fecham 100%. Desligado: volta ao Total inteiro do stub -- quem nunca foi "
                "perguntado entra como 'não respondeu' e a soma fica abaixo de 100%. "
                "A regra é automática: um grupo (ex.: uma operação, um ano, ano x UF) sai da "
                "base se menos de 5% dele respondeu a pergunta e ela tem resposta em outros "
                "grupos. Ela não conhece o desenho do questionário -- a legenda acima da "
                "tabela mostra exatamente quem saiu, e vale conferir."
            ),
        )
        na_handling = st.radio(
            "Categoria 'N/A - ...' em indicadores",
            options=["keep", "exclude"],
            format_func=lambda v: "Manter como categoria" if v == "keep" else "Excluir da base",
            help=(
                "Muda o DENOMINADOR de todos os percentuais dos indicadores, não só o "
                "rótulo -- 'Excluir da base' recalcula %LINHA/%COLUNA só sobre quem "
                "respondeu de fato, então o mesmo indicador pode mostrar números "
                "diferentes dependendo dessa escolha. Indicadores de baixa incidência "
                "(ex.: avaliação de atendimento, quando a maioria nunca contatou) ficam "
                "com N/A dominante se mantido, e com base muito pequena (BASE menor, "
                "célula pode ficar amarela) se excluído. Escolha por variável ainda não "
                "está implementado nesta versão — é o próximo incremento."
            ),
        )
        small_n_threshold = st.number_input(
            "Alertar células com base menor que", min_value=1, value=30, step=5
        )

    # `cascading_data` já é o resultado de aplicar todos os filtros em sequência
    # (calculado dentro do loop acima, pro cascateamento de opções funcionar) --
    # reusar em vez de refazer o corte do zero evita duas fontes de verdade.
    filtered_data = cascading_data

    if active_filters:
        st.caption(f"Base após filtro: {len(filtered_data)} de {len(data)} respondentes.")
        if filtered_data.empty:
            st.warning("O filtro de base zerou a amostra. Ajuste as seleções na barra lateral.")
            st.stop()

    return filtered_data, stub_key, banner_keys, na_handling, small_n_threshold, active_filters, use_weighting, use_universe, visible_options


# ══════════════════════════════════════════════════════════════════════
#  ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("Cruza-Tudo")
    st.caption(
        "Cruza qualquer variável de conteúdo (stub) contra qualquer variável de perfil (banner), "
        "com múltipla resposta tratada automaticamente — sem pivotar nada manualmente antes."
    )

    try:
        data, meta = _load_and_classify(str(DATA_PATH))
    except Exception as exc:  # noqa: BLE001 -- é uma fronteira de UI, queremos mensagem legível, não traceback
        st.error(f"Não consegui carregar {DATA_PATH.name}: {exc}")
        st.stop()

    entries = crossable_variables(meta)
    options = {e["key"]: e["label"] for e in entries}
    n_mr = sum(1 for e in entries if e["var_type"] == VarType.MR_OPTION)

    st.caption(
        f"{len(data)} respondentes · {len(options)} variáveis cruzáveis "
        f"({n_mr} são blocos de múltipla resposta, unpivotados automaticamente)."
    )

    # Filtro SQL: primeiro corte, antes da barra lateral (ver seção FILTRO SQL).
    sql_condition, sql_mask = _filtro_sql_atual(data, meta)
    sql_base = data[sql_mask] if sql_mask is not None else None
    if sql_condition:
        _render_aviso_sql(sql_condition, int(sql_mask.sum()), len(data))

    (
        filtered_data, stub_key, banner_keys, na_handling, small_n_threshold,
        active_filters, use_weighting, use_universe, visible_options,
    ) = _render_sidebar(data, meta, options, sql_condition, sql_base)

    # Aviso de peso: logo abaixo do cabeçalho, vale para todas as abas.
    if use_weighting:
        st.info("⚖️ **Peso amostral LIGADO** — percentuais ponderados pelo PESO (NA segue contagem bruta).")
    else:
        st.warning("⚖️ **Peso amostral DESLIGADO** — o app inteiro (Cruzamento, Comparar, Índices e Estatística) em contagem bruta, sem ponderar. Pra ponderar, ligue em \"3. Regras\" na barra lateral.")

    if not banner_keys:
        st.warning("Selecione ao menos uma variável de banner na barra lateral (aba Cruzamento).")

    # "Filtro SQL" é a ÚLTIMA aba de propósito: o Streamlit abre na primeira,
    # então o app continua abrindo no Cruzamento, como antes.
    # `key` fixa a identidade das abas: sem ele, o aviso do filtro SQL que
    # aparece/some ACIMA das abas mudava a posição delas na página, e o
    # Streamlit voltava pra primeira aba a cada Aplicar/Limpar (visto no teste).
    # Diagnóstico é a ÚLTIMA (decisão do Ro, 25/11): é a aba de olhar o dado
    # cru, não de análise. Estatística fica junto das abas de análise.
    tab_cruzamento, tab_comparacao, tab_indices, tab_estatistica, tab_sql, tab_dados = st.tabs(
        ["📊 Cruzamento", "🔀 Comparar", "📈 Índices", "📐 Estatística", "🔎 Filtro SQL", "🩺 Diagnóstico de dados"],
        key="abas_principais",
    )
    with tab_cruzamento:
        _render_cruzamento_tab(data, meta, visible_options, filtered_data, stub_key, banner_keys, na_handling, small_n_threshold, active_filters, use_weighting, use_universe)
    with tab_comparacao:
        _render_comparacao_tab(data, meta, visible_options, filtered_data, na_handling, small_n_threshold, active_filters, use_weighting, use_universe)
    with tab_indices:
        _render_indices_tab(data, meta, visible_options, filtered_data, _filter_note_lines(active_filters, meta),
                            use_weighting, small_n_threshold)
    with tab_estatistica:
        _render_estatistica_tab(data, meta, visible_options, filtered_data, use_weighting, active_filters)
    with tab_sql:
        _render_sql_tab(data, meta, filtered_data, sql_mask)
    with tab_dados:
        _render_dados_tab(data, meta, filtered_data)


if __name__ == "__main__":
    main()
