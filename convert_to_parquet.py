"""
convert_to_parquet.py — Conversão única de xlsx (cabeçalho duplo) pra
parquet otimizado + rótulos.

PAPEL NO PROJETO
-----------------
Script standalone, roda FORA do Streamlit, nunca importado por `app.py`.
É o único jeito de atualizar o banco: rodar isso de novo e commitar o
`.parquet` + `.labels.json` resultantes no repositório.

POR QUE ISSO É UM SCRIPT SEPARADO
------------------------------------
Ler um xlsx largo com openpyxl custa minutos, não segundos -- medido
~35s pra 14,5MB de teste (3060 linhas x 1164 colunas); pros ~156MB reais
do banco de produção (106.217 linhas), ~7min. Pagar esse custo dentro do
Streamlit, a cada sessão, era o que tornava o app impraticável. Aqui você
paga uma vez, offline, e o app passa a ler o resultado -- parquet é
colunar e ~76x mais rápido pra ler de volta, e a otimização de memória
feita aqui (ver `_optimize_memory`) derruba o uso de RAM em ~82%,
essencial pro teto de 1GB do Streamlit Community Cloud.

USO
----
    python convert_to_parquet.py /caminho/df_completo.xlsx [nome_da_aba]

Gera, ao lado do xlsx:
    df_completo.parquet        -- os dados, colunas com nome curto, já
                                   otimizados pra memória
    df_completo.labels.json    -- {short_names, full_labels}

O sidecar de rótulos existe porque parquet só guarda nome de coluna, não
as duas linhas de cabeçalho do Excel -- sem ele, a linha 1 (pergunta
completa) se perderia na conversão, e o banner voltaria a mostrar código
cru em vez de pergunta legível.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
# _MEDIA_PATTERN e _WEIGHT_NAMES são "privados" de metadata.py (prefixo
# _), mas importados aqui de propósito -- são os MESMOS padrões que
# classify_columns usa pra reconhecer coluna de peso e companion numérico.
# Reimportar em vez de duplicar a regex garante que os dois módulos nunca
# divirjam sobre o que conta como "_media" ou "coluna de peso".
from metadata import VarType, classify_columns, load_raw_with_double_header, _MEDIA_PATTERN, _WEIGHT_NAMES


# ══════════════════════════════════════════════════════════════════════
#  AJUSTE DE TIPO E OTIMIZAÇÃO DE MEMÓRIA
# ══════════════════════════════════════════════════════════════════════

def _coerce_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    pyarrow (o motor do parquet) exige tipo consistente por coluna. Uma
    planilha de pesquisa real mistura número e texto na mesma coluna com
    frequência (ex.: CRIANCAS com "0 - Nenhuma" numa linha e um número
    puro noutra) -- não é bug do arquivo, é como Excel deixa acontecer, e
    foi exatamente isso que quebrou a primeira tentativa de benchmark
    deste projeto.

    Regra: PESO e colunas "_media" viram numéricas de verdade, porque
    entram em conta matemática (soma ponderada, média). Todo o resto vira
    texto -- é o que são, semanticamente: categoria, mesmo quando o
    rótulo parece número (ex.: uma faixa de renda codificada como "1",
    "2", "3" ainda é categoria, não teria sentido somar/tirar média).
    """
    out = data.copy()
    for col in out.columns:
        is_weight = col.upper() in _WEIGHT_NAMES
        is_media = bool(_MEDIA_PATTERN.match(col))
        if is_weight or is_media:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = out[col].astype("string")
    return out


def _optimize_memory(data: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Sem isso, o banco de produção (106 mil linhas x 1164 colunas) projeta
    ~1,3GB só pra existir na RAM -- acima do teto de 1GB do Streamlit
    Community Cloud sozinho, antes de qualquer conta rodar em cima.
    Medido: metade dessa memória vem dos blocos de múltipla resposta
    (quase 50% das 1164 colunas), guardando o texto inteiro da opção
    repetido em toda linha marcada, quando `to_long()`
    (`crosstab_engine.py`) só olha `.notna()` -- nunca lê o conteúdo da
    célula MR. Virar booleano não perde nenhuma informação que o motor
    usa, e cai o custo de string pra ~1 bit por célula.

    SR e indicador viram "category": mesmas strings, guardadas uma vez só
    e referenciadas por código -- transparente pro resto do código
    (`.astype(str)`, `.str.match(...)` continuam funcionando igual em
    cima de category).

    Medido: 81,6% de redução na amostra pequena de teste; 83% no banco
    real de produção (1268MB -> 221MB).
    """
    out = data.copy()
    for name, m in meta.items():
        col = out[name]
        if isinstance(col, pd.DataFrame):  # nome curto duplicado não desambiguado
            col = col.iloc[:, 0]
        if m.var_type == VarType.MR_OPTION:
            out[name] = col.notna()
        elif m.var_type in (VarType.SR, VarType.INDICATOR):
            out[name] = col.astype("category")
    return out


# ══════════════════════════════════════════════════════════════════════
#  CONVERSÃO
# ══════════════════════════════════════════════════════════════════════

def convert(xlsx_path: str, sheet_name: str = "Dados") -> None:
    """
    Lê o xlsx bruto, ajusta tipo, classifica, otimiza memória, e escreve
    `.parquet` + `.labels.json` ao lado do arquivo de origem. Imprime
    progresso e métricas (tempo de leitura, redução de memória, tamanho
    final) -- é intencional que rode como script visível no terminal, não
    silencioso, já que a etapa de leitura demora minutos e a pessoa
    precisa saber que não travou.
    """
    xlsx_path = Path(xlsx_path)

    t0 = time.time()
    print(f"Lendo {xlsx_path.name} -- essa é a parte lenta, só acontece agora...")
    data, full_labels, short_names = load_raw_with_double_header(str(xlsx_path), sheet_name=sheet_name)
    print(f"  {data.shape[0]} linhas x {data.shape[1]} colunas lidas em {time.time()-t0:.1f}s")

    data = _coerce_dtypes(data)

    print("  classificando variáveis pra otimizar memória (MR -> booleano, SR/indicador -> categoria)...")
    meta = classify_columns(data, full_labels, short_names)
    mem_before = data.memory_usage(deep=True).sum()
    data = _optimize_memory(data, meta)
    mem_after = data.memory_usage(deep=True).sum()
    print(f"  memória em RAM: {mem_before/1e6:.1f}MB -> {mem_after/1e6:.1f}MB "
          f"({(1 - mem_after/mem_before):.0%} menor)")

    parquet_path = xlsx_path.with_suffix(".parquet")
    labels_path = xlsx_path.with_suffix("").with_suffix(".labels.json")

    t0 = time.time()
    data.to_parquet(parquet_path)
    t_write = time.time() - t0

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"short_names": short_names, "full_labels": full_labels}, f, ensure_ascii=False)

    print(f"  parquet escrito em {t_write:.1f}s -> {parquet_path.name} "
          f"({parquet_path.stat().st_size/1e6:.1f} MB, era {xlsx_path.stat().st_size/1e6:.1f} MB em xlsx)")
    print(f"  rótulos salvos em -> {labels_path.name}")
    print(f"\nA partir de agora, no app, use o caminho do .parquet -- não o do .xlsx.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convert_to_parquet.py /caminho/arquivo.xlsx [nome_da_aba]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Dados")
