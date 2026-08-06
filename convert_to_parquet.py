"""
Conversão única de xlsx (cabeçalho duplo) para parquet + rótulos.

Por que isso é um script separado, fora do app: ler um xlsx largo com
openpyxl custa minutos, não segundos -- medi 35s pra ler 14,5MB de teste
(3060 linhas x 1164 colunas); pra 405MB reais, projeção de 15-20min. Pagar
esse custo dentro do Streamlit, a cada sessão, era o que tornava o app
impraticável. Aqui você paga uma vez, offline, e o app passa a ler o
resultado -- parquet é colunar e ~76x mais rápido pra ler de volta.

Uso:
    python convert_to_parquet.py /caminho/df_completo.xlsx [nome_da_aba]

Gera, ao lado do xlsx:
    df_completo.parquet        -- os dados, colunas com nome curto
    df_completo.labels.json    -- {short_names, full_labels}

O sidecar de rótulos existe porque parquet só guarda nome de coluna, não as
duas linhas de cabeçalho do Excel -- sem ele, a linha 1 (pergunta completa)
se perderia na conversão, e o banner voltaria a mostrar código cru.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metadata import load_raw_with_double_header, _MEDIA_PATTERN, _WEIGHT_NAMES


def _coerce_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    pyarrow (o motor do parquet) exige tipo consistente por coluna. Uma
    planilha de pesquisa real mistura número e texto na mesma coluna com
    frequência (ex.: CRIANCAS com "0 - Nenhuma" numa linha e um número puro
    noutra) -- não é bug do arquivo, é como Excel deixa acontecer, e foi
    exatamente isso que quebrou minha primeira tentativa de benchmark.

    Regra: PESO e colunas "_media" viram numéricas de verdade, porque entram
    em conta matemática (soma ponderada, média). Todo o resto vira texto --
    é o que são, semanticamente: categoria, mesmo quando o rótulo parece
    número.
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


def convert(xlsx_path: str, sheet_name: str = "Dados") -> None:
    xlsx_path = Path(xlsx_path)

    t0 = time.time()
    print(f"Lendo {xlsx_path.name} -- essa é a parte lenta, só acontece agora...")
    data, full_labels, short_names = load_raw_with_double_header(str(xlsx_path), sheet_name=sheet_name)
    print(f"  {data.shape[0]} linhas x {data.shape[1]} colunas lidas em {time.time()-t0:.1f}s")

    data = _coerce_dtypes(data)

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
