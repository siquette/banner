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
três módulos.

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
5. Barra lateral -- `_render_sidebar`, devolve os valores escolhidos
6. `main()` -- amarra tudo

Rodar localmente:
    pixi run streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crosstab_engine import (
    build_banner,
    format_banner_table_full,
    format_table_for_export,
    get_column_series,
    get_weights,
    small_n_mask_full,
)
from indices import compute_index_trend, compute_quadrant_data, indicator_media_map
from metadata import VarType, category_color, classify_columns, crossable_variables, load_parquet_with_labels

st.set_page_config(page_title="Gerador de Banner", layout="wide")

# Caminho fixo do banco, relativo a este arquivo -- funciona igual local e
# no Streamlit Cloud, porque nos dois casos o diretório de trabalho é a
# raiz do repositório clonado. BANNER_DATA_PATH sobrescreve isso via
# variável de ambiente, só pra testar outro arquivo localmente sem editar
# código -- nunca é setada em produção.
DATA_PATH = Path(os.environ.get("BANNER_DATA_PATH", str(Path(__file__).parent / "df_completo_v2_corrigido.parquet")))


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


# ══════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ══════════════════════════════════════════════════════════════════════

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

    blocks = build_banner(filtered_data, meta, stub_key, banner_keys, na_handling, small_n_threshold)
    if not blocks:
        st.warning(
            "Nenhuma das variáveis de banner selecionadas tem respondente elegível "
            "cruzado com o stub escolhido nesse conjunto de dados (ou nesse filtro de base)."
        )
        return

    table = format_banner_table_full(blocks)
    mask = small_n_mask_full(blocks)

    if active_filters:
        # Total sem filtro nenhum -- só pra comparação lado a lado com o
        # Total já filtrado que o resto da tabela usa. banner_keys=[] faz
        # build_banner devolver só o bloco Total, sem cruzar nada.
        unfiltered_blocks = build_banner(data, meta, stub_key, [], na_handling, small_n_threshold)
        unfiltered_table = format_banner_table_full(unfiltered_blocks)
        unfiltered_mask = small_n_mask_full(unfiltered_blocks)
        unfiltered_table.columns = pd.MultiIndex.from_tuples(
            [("Total geral (sem filtro)", c[1]) for c in unfiltered_table.columns]
        )
        unfiltered_mask.columns = unfiltered_table.columns
        table = pd.concat([unfiltered_table, table], axis=1)
        mask = pd.concat([unfiltered_mask, mask], axis=1)

    # --- legendas de cobertura por variável de banner ---
    total_n_ref = blocks[0].base_n["Total"]
    for b in blocks[1:]:
        own_n = b.base_n.sum()
        pct = own_n / total_n_ref * 100 if total_n_ref else 0
        st.caption(f"**{b.banner_label}**: resposta de {own_n:,} de {total_n_ref:,} ({pct:.1f}%)")
        if b.coverage_warning:
            st.warning(f"**{b.banner_label}**: {b.coverage_warning}")

    st.subheader(f"{options[stub_key]}")

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
            "- **Total geral (sem filtro)** — só aparece se você aplicou um filtro de base; é o "
            "Total de antes do filtro, pra comparação lado a lado.\n"
            "- **Células em amarelo** — a base ali é menor que o limiar definido na barra lateral; "
            "leia esse percentual com cautela, poucos respondentes sustentam esse número.\n"
            "- **Aviso amarelo acima da tabela** — sinaliza base concentrada numa categoria só do "
            "stub, ou cobertura baixa da pergunta (nem todo mundo respondeu) — não é erro de conta.\n"
            "- Se %LINHA de uma categoria não soma perto de 100%, é sinal de que nem todo mundo "
            "daquele grupo respondeu a pergunta.\n"
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

    # --- gráficos: %LINHA e %COLUNA lado a lado, por variável de banner ---
    st.subheader("Gráficos")
    chart_type = st.radio(
        "Tipo de gráfico",
        options=["Barras", "Barra horizontal", "Linha"],
        horizontal=True,
        help=(
            "Barra horizontal ajuda quando os nomes das categorias são longos "
            "(ex.: texto de pergunta aberta codificada) e ficam cortados no eixo "
            "vertical. Linha só faz sentido quando as categorias do stub têm "
            "ordem natural (ex.: Ano, faixa etária, escala de concordância) -- "
            "pra categorias sem ordem (ex.: Gênero, Região), a linha sugere uma "
            "tendência que não existe."
        ),
    )
    bargap, bargroupgap = 0.2, 0.1
    if chart_type in ("Barras", "Barra horizontal"):
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
    row_totals_weighted = blocks[0].cell_weighted["Total"]  # base ponderada de cada categoria do stub, pro %LINHA
    for b in blocks[1:]:  # blocks[0] é sempre "Total" -- não faz gráfico próprio, já está implícito em cada um dos outros
        # %LINHA: "como esse grupo do stub se distribui nessa pergunta" --
        # leitura mais natural pra comparar grupos entre si.
        pct_linha = b.cell_weighted.divide(row_totals_weighted, axis=0) * 100
        pct_linha = pct_linha.fillna(0.0)
        # %COLUNA: "quem compõe cada resposta" -- já vem pronto em b.pct.
        pct_coluna = b.pct

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

    st.download_button(
        "Baixar banner (CSV)",
        data=format_table_for_export(table).to_csv().encode("utf-8"),
        file_name="banner.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════
#  ABA: ÍNDICES
# ══════════════════════════════════════════════════════════════════════
# Três visões independentes (visão geral, individual, quadrante), cada
# uma na sua própria função -- eram um único if/elif/else grande antes;
# separadas fica mais fácil achar/mudar uma visão sem reler as outras
# duas.

def _render_indices_overview(filtered_data, meta, options, indicator_names, media_map, segment_key, weights) -> None:
    """Linha de tendência de TODOS os índices juntos, mesma escala 1-5 -- visão de "painel de saúde"."""
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


def _render_indices_tab(data: pd.DataFrame, meta: dict, options: dict, filtered_data: pd.DataFrame) -> None:
    """
    Ponto de entrada da aba Índices: monta os controles comuns às três
    visões (segmento, seletor de visão) e despacha pra
    `_render_indices_overview`/`_individual`/`_quadrant` conforme
    escolhido.
    """
    indicator_names = [m.name for m in meta.values() if m.var_type == VarType.INDICATOR]
    media_map = indicator_media_map(meta)
    if not indicator_names or not media_map:
        st.info("Esse banco não tem variáveis de indicador (_c) com companion numérico (_media).")
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
        view_mode = st.radio("Visualização", ["Visão geral", "Índice individual", "Quadrante"], horizontal=True)

    weights = get_weights(filtered_data, meta)

    with st.expander("❓ Como ler esse painel"):
        st.markdown(
            "- Cada índice (`IACOM`, `IMC`, `IM`...) é uma **média ponderada** (não % de "
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
        _render_indices_overview(filtered_data, meta, options, indicator_names, media_map, segment_key, weights)
    elif view_mode == "Índice individual":
        _render_indices_individual(filtered_data, meta, options, indicator_names, media_map, segment_key, weights)
    else:  # Quadrante
        _render_indices_quadrant(filtered_data, meta, indicator_names, media_map, segment_key, weights)


# ══════════════════════════════════════════════════════════════════════
#  BARRA LATERAL
# ══════════════════════════════════════════════════════════════════════

def _render_sidebar(data: pd.DataFrame, meta: dict, options: dict) -> tuple[pd.DataFrame, str, list[str], str, int, dict]:
    """
    Monta a barra lateral inteira (filtro de base, seleção de stub/
    banner, regras) e já aplica o filtro escolhido, devolvendo tudo que
    `main()` precisa pra chamar as duas abas:

        (filtered_data, stub_key, banner_keys, na_handling, small_n_threshold, active_filters)

    O filtro de base é aplicado ANTES de qualquer cruzamento -- é só um
    recorte de linhas do df. Nada em `crosstab_engine.py` precisa saber
    que um filtro existe; ele só vê um df menor.
    """
    with st.sidebar:
        st.header("1. Filtro de base (opcional)")
        filter_keys = st.multiselect(
            "Restringir a base a...",
            options=list(options.keys()),
            format_func=lambda k: options[k],
            help=(
                "Filtro por bloco de múltipla resposta ainda não é suportado -- "
                "só variáveis de resposta única ou indicador."
            ),
        )

        active_filters: dict[str, list[str]] = {}
        for fk in filter_keys:
            fmeta = meta.get(fk)
            if fmeta is None or fmeta.var_type not in (VarType.SR, VarType.INDICATOR):
                st.warning(f"'{options[fk]}' é múltipla resposta -- filtro ignorado nessa versão.")
                continue
            col = get_column_series(data, fmeta.name)
            available = sorted(v for v in col.dropna().unique().tolist())
            chosen = st.multiselect(f"Manter em '{options[fk]}'", options=available, default=available, key=f"filter_{fk}")
            active_filters[fk] = chosen

        st.header("2. Cruzamento")
        stub_key = st.selectbox(
            "Variável de linha (stub)",
            options=list(options.keys()),
            format_func=lambda k: options[k],
        )
        banner_keys = st.multiselect(
            "Variáveis de banner (colunas)",
            options=[k for k in options if k != stub_key],
            format_func=lambda k: options[k],
        )

        st.header("3. Regras")
        na_handling = st.radio(
            "Categoria 'N/A - ...' em indicadores",
            options=["keep", "exclude"],
            format_func=lambda v: "Manter como categoria" if v == "keep" else "Excluir da base",
            help=(
                "Indicadores de baixa incidência (ex.: avaliação de atendimento, "
                "quando a maioria nunca contatou) ficam com N/A dominante se mantido, "
                "e com base muito pequena se excluído. Escolha por variável ainda não "
                "está implementado nesta versão — é o próximo incremento."
            ),
        )
        small_n_threshold = st.number_input(
            "Alertar células com base menor que", min_value=1, value=30, step=5
        )

    filtered_data = data
    for fk, chosen in active_filters.items():
        fmeta = meta[fk]
        col = get_column_series(filtered_data, fmeta.name)
        filtered_data = filtered_data[col.isin(chosen)]

    if active_filters:
        st.caption(f"Base após filtro: {len(filtered_data)} de {len(data)} respondentes.")
        if filtered_data.empty:
            st.warning("O filtro de base zerou a amostra. Ajuste as seleções na barra lateral.")
            st.stop()

    return filtered_data, stub_key, banner_keys, na_handling, small_n_threshold, active_filters


# ══════════════════════════════════════════════════════════════════════
#  ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("Gerador de tabela banner")
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

    filtered_data, stub_key, banner_keys, na_handling, small_n_threshold, active_filters = _render_sidebar(data, meta, options)

    if not banner_keys:
        st.warning("Selecione ao menos uma variável de banner na barra lateral (aba Cruzamento).")

    tab_cruzamento, tab_indices = st.tabs(["📊 Cruzamento", "📈 Índices"])
    with tab_cruzamento:
        _render_cruzamento_tab(data, meta, options, filtered_data, stub_key, banner_keys, na_handling, small_n_threshold, active_filters)
    with tab_indices:
        _render_indices_tab(data, meta, options, filtered_data)


if __name__ == "__main__":
    main()
