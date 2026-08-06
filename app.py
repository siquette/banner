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
    get_label,
    load_parquet_with_labels,
)
from crosstab_engine import build_banner, format_banner_table_full, get_column_series, small_n_mask_full

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

    for b in blocks:
        if b.coverage_warning:
            st.warning(f"**{b.banner_label}**: {b.coverage_warning}")

    st.subheader(f"{options[stub_key]}")

    def _highlight_small_n(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        for col in mask.columns:
            styles.loc[mask.index, col] = mask[col].map(
                lambda flagged: "background-color: #fff3cd" if flagged else ""
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

    st.subheader("Gráfico")
    chart_banner_label = get_label(meta, banner_keys[0])
    chosen_block = next((b for b in blocks if b.banner_label == chart_banner_label), blocks[-1] if len(blocks) > 1 else blocks[0])
    pct = chosen_block.pct

    fig = go.Figure()
    for col in pct.columns:
        fig.add_trace(go.Bar(name=str(col), x=pct.index.astype(str), y=pct[col]))
    fig.update_layout(
        barmode="group",
        title=f"{options[stub_key]} por {chart_banner_label}",
        yaxis_title="%",
        legend_title=chart_banner_label,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Baixar banner (CSV)",
        data=table.to_csv().encode("utf-8"),
        file_name="banner.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
