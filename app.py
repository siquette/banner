"""
Interface do gerador de banner. Fina de propósito: toda a lógica pesada
(classificação de variável, cruzamento ponderado, unpivot de múltipla
resposta) está em metadata.py e crosstab_engine.py, testados isoladamente.
Este arquivo só orquestra widgets e desenha o resultado.

Banco fixo: lê sempre o mesmo .parquet, commitado no repo ao lado deste
arquivo (gerado uma vez por convert_to_parquet.py, fora do app -- ver
README). Sem uploader nem campo de caminho -- não faz sentido pra um app
com um cliente/estudo fixo, e evita reintroduzir a leitura de xlsx cru
(lenta, ~35s pra 14,5MB) dentro do container do Streamlit Cloud.

Rodar localmente:
    pixi run streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metadata import (
    VarType,
    classify_columns,
    crossable_variables,
    load_parquet_with_labels,
)
from crosstab_engine import (
    build_banner,
    format_banner_table_full,
    format_table_for_export,
    get_column_series,
    small_n_mask_full,
)

st.set_page_config(page_title="Gerador de Banner", layout="wide")

# Caminho fixo do banco, relativo a este arquivo -- funciona igual local e
# no Streamlit Cloud, porque nos dois casos o diretório de trabalho é a
# raiz do repositório clonado. BANNER_DATA_PATH sobrescreve isso via
# variável de ambiente, só pra testar outro arquivo localmente sem editar
# código -- nunca é setada em produção.
DATA_PATH = Path(os.environ.get("BANNER_DATA_PATH", str(Path(__file__).parent / "df_completo_v2_corrigido.parquet")))


@st.cache_data(show_spinner="Carregando o banco...")
def _load_and_classify(path: str):
    data, full_labels, short_names = load_parquet_with_labels(path)
    meta = classify_columns(data, full_labels, short_names)
    return data, meta


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

    # O filtro de base é aplicado ANTES de qualquer cruzamento -- é só um
    # recorte de linhas do df. Nada em crosstab_engine.py precisa saber que
    # um filtro existe; ele só vê um df menor.
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

    if not banner_keys:
        st.warning("Selecione ao menos uma variável de banner na barra lateral.")
        st.stop()

    blocks = build_banner(filtered_data, meta, stub_key, banner_keys, na_handling, small_n_threshold)
    if not blocks:
        st.warning(
            "Nenhuma das variáveis de banner selecionadas tem respondente elegível "
            "cruzado com o stub escolhido nesse conjunto de dados (ou nesse filtro de base)."
        )
        st.stop()

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
    st.dataframe(styled, use_container_width=True)
    st.caption(
        "NA = contagem não ponderada · %LINHA = % dentro da categoria do stub · "
        "%COLUNA = % dentro da categoria do banner. Células em amarelo: base "
        "abaixo do limiar definido — leia o percentual com cautela."
    )

    st.subheader("Gráficos")
    chart_type = st.radio(
        "Tipo de gráfico",
        options=["Barras", "Linha"],
        horizontal=True,
        help=(
            "Linha só faz sentido quando as categorias do stub têm ordem natural "
            "(ex.: Ano, faixa etária, escala de concordância). Pra categorias sem "
            "ordem (ex.: Gênero, Região), a linha sugere uma tendência que não "
            "existe -- use Barras nesses casos."
        ),
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
            fig1 = go.Figure()
            for col in pct_linha.columns:
                hover = (
                    f"{options[stub_key]}: %{{x}}<br>"
                    f"{b.banner_label}: {col}<br>"
                    "%{y:.1f}% desse grupo<extra></extra>"
                )
                if chart_type == "Barras":
                    fig1.add_trace(go.Bar(name=str(col), x=pct_linha.index.astype(str), y=pct_linha[col], hovertemplate=hover))
                else:
                    fig1.add_trace(go.Scatter(name=str(col), x=pct_linha.index.astype(str), y=pct_linha[col], mode="lines+markers", hovertemplate=hover))
            fig1.update_layout(
                barmode="group",
                title=f"{options[stub_key]} por {b.banner_label} — %LINHA",
                yaxis_title=f"% dentro de cada grupo de {options[stub_key]}",
                legend_title=b.banner_label,
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Como cada grupo se distribui nessa pergunta — compare grupos entre si.")

        with col_coluna:
            fig2 = go.Figure()
            for col in pct_coluna.columns:
                hover = (
                    f"{b.banner_label}: {col}<br>"
                    f"{options[stub_key]}: %{{x}}<br>"
                    "%{y:.1f}% de quem respondeu isso<extra></extra>"
                )
                if chart_type == "Barras":
                    fig2.add_trace(go.Bar(name=str(col), x=pct_coluna.index.astype(str), y=pct_coluna[col], hovertemplate=hover))
                else:
                    fig2.add_trace(go.Scatter(name=str(col), x=pct_coluna.index.astype(str), y=pct_coluna[col], mode="lines+markers", hovertemplate=hover))
            fig2.update_layout(
                barmode="group",
                title=f"{options[stub_key]} por {b.banner_label} — %COLUNA",
                yaxis_title="% de quem escolheu cada opção",
                legend_title=b.banner_label,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Quem compõe cada resposta — o perfil de quem escolheu cada opção.")

    st.download_button(
        "Baixar banner (CSV)",
        data=format_table_for_export(table).to_csv().encode("utf-8"),
        file_name="banner.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
