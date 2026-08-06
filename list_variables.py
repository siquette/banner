"""
Auditoria do classificador: lista TODA variável que aparece na caixa de
cruzamento, com tipo e -- pra bloco MR -- quais colunas cruas foram
agrupadas ali dentro. Existe porque olhar a lista de longe (300+ entradas)
esconde problema que só aparece de perto, como dois grupos diferentes
mostrando o mesmo texto, ou uma variável sumindo por colisão de chave.

Uso:
    python list_variables.py /caminho/df.parquet
    python list_variables.py /caminho/df.xlsx [nome_da_aba]

Gera variaveis_cruzamento.csv ao lado do arquivo de entrada, e imprime no
terminal qualquer incoerência que achar (rótulo duplicado, colisão de
chave entre SR e grupo MR).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metadata import (
    VarType,
    _RM_TAG_PATTERN,
    classify_columns,
    crossable_variables,
    load_parquet_with_labels,
    load_raw_with_double_header,
    mr_groups,
)


def audit(path: str, sheet_name: str = "Dados") -> None:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".parquet":
        data, full_labels, short_names = load_parquet_with_labels(path)
    else:
        data, full_labels, short_names = load_raw_with_double_header(path, sheet_name=sheet_name)

    meta = classify_columns(data, full_labels, short_names)
    entries = crossable_variables(meta)
    groups = mr_groups(meta)

    print(f"{len(short_names)} colunas cruas -> {len(entries)} variáveis cruzáveis "
          f"({len(groups)} são blocos de múltipla resposta agrupados)\n")

    # incoerência 1: dois grupos/variáveis diferentes com o mesmo rótulo exibido
    label_counts = Counter(e["label"] for e in entries)
    label_dups = {lbl: c for lbl, c in label_counts.items() if c > 1}
    if label_dups:
        print(f"[ATENÇÃO] {len(label_dups)} rótulo(s) repetido(s) na caixa de seleção:")
        for lbl, c in label_dups.items():
            keys = [e["key"] for e in entries if e["label"] == lbl]
            print(f"  {lbl!r} aparece {c}x -> chaves distintas: {keys}")
        print()
    else:
        print("Nenhum rótulo duplicado na caixa de seleção.\n")

    # incoerência 2: chave de SR/indicador colidindo com código de grupo MR
    # (mais grave -- dict do app.py perde uma das duas silenciosamente)
    sr_keys = {e["key"] for e in entries if e["var_type"] in (VarType.SR, VarType.INDICATOR)}
    mr_keys = {e["key"] for e in entries if e["var_type"] == VarType.MR_OPTION}
    collision = sr_keys & mr_keys
    if collision:
        print(f"[ATENÇÃO] {len(collision)} colisão(ões) de chave entre variável SR/indicador "
              f"e grupo MR -- uma das duas está sumindo da caixa de seleção:")
        for k in collision:
            print(f"  {k!r}")
        print()
    else:
        print("Nenhuma colisão de chave entre SR/indicador e grupo MR.\n")

    # incoerência 3: coluna sem "-" no nome (nunca analisada como candidata a
    # MR) mas cujo rótulo completo traz a tag "(RM - ...)" -- desde a última
    # correção, classify_columns já trata isso como MR de 1 opção
    # automaticamente; esse check é uma rede de segurança pra confirmar que
    # nenhum caso escapou (só teria sentido reportar se ainda estivesse
    # classificado como SR apesar da tag).
    no_dash_rm = []
    for name in short_names:
        if "-" in name:
            continue
        m = meta[name]
        if m.var_type != VarType.SR:
            continue
        tag = _RM_TAG_PATTERN.search(m.label)
        if tag and tag.group(1).upper() == "RM":
            no_dash_rm.append(name)
    if no_dash_rm:
        print(f"[ATENÇÃO] {len(no_dash_rm)} variável(is) com tag (RM - ...) no rótulo, mas sem "
              f"'-' no nome, ainda classificada(s) como resposta única -- não deveria acontecer "
              f"mais, vale investigar:")
        for n in no_dash_rm:
            print(f"  {n!r} -> {meta[n].label[:100]}")
        print()
    else:
        print("Nenhuma variável com tag RM sem '-' no nome ficou classificada como SR.\n")

    # csv completo pra abrir no Excel e revisar com calma
    out_path = path_obj.with_name("variaveis_cruzamento.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["chave", "rotulo_exibido", "tipo", "n_colunas_no_grupo", "colunas_cruas"])
        for e in entries:
            if e["var_type"] == VarType.MR_OPTION:
                members = groups.get(e["key"], [])
                writer.writerow([e["key"], e["label"], e["var_type"].value, len(members),
                                  " | ".join(m.name for m in members)])
            else:
                writer.writerow([e["key"], e["label"], e["var_type"].value, 1, e["key"]])

    print(f"Lista completa salva em: {out_path}\n")

    print(f"--- todas as {len(entries)} variáveis ---")
    for e in entries:
        n = len(groups.get(e["key"], [])) if e["var_type"] == VarType.MR_OPTION else 1
        extra = f" ({n} colunas)" if n > 1 else ""
        print(f"[{e['var_type'].value}] {e['key']}{extra} -> {e['label']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python list_variables.py /caminho/arquivo.parquet-ou-xlsx [nome_da_aba]")
        sys.exit(1)
    audit(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Dados")
