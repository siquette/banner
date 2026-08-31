"""
list_variables.py — Auditoria do classificador de metadados.

PAPEL NO PROJETO
-----------------
Script standalone de diagnóstico, roda fora do Streamlit. Lista TODA
variável que apareceria na caixa de cruzamento de `app.py`, com tipo e --
pra bloco MR -- quais colunas cruas foram agrupadas ali dentro. Existe
porque olhar a lista de longe (300+ entradas) esconde problema que só
aparece de perto, como dois grupos diferentes mostrando o mesmo texto, ou
uma variável sumindo por colisão de chave -- os três tipos de
incoerência que este script já pegou de verdade no banco de produção
(ver `_check_duplicate_labels`, `_check_key_collision`,
`_check_orphan_rm_tag`).

Rodar isso depois de qualquer mudança em `metadata.py`, ou depois de
receber uma versão nova do banco, é a forma mais rápida de saber se a
classificação continua fazendo sentido antes de abrir o app de verdade.

USO
----
    python list_variables.py /caminho/df.parquet
    python list_variables.py /caminho/df.xlsx [nome_da_aba]

Gera `variaveis_cruzamento.csv` ao lado do arquivo de entrada, e imprime
no terminal qualquer incoerência encontrada.
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


# ══════════════════════════════════════════════════════════════════════
#  CHECAGENS DE INCOERÊNCIA
# ══════════════════════════════════════════════════════════════════════
# Cada função devolve uma lista de linhas de texto pra imprimir (vazia se
# nada de errado foi encontrado) -- separado em funções pra cada
# checagem ser testável/lida isoladamente, e pra facilitar adicionar uma
# quarta checagem no futuro sem inchar mais uma função só.

def _check_duplicate_labels(entries: list[dict]) -> list[str]:
    """
    Dois grupos/variáveis diferentes mostrando o mesmo texto na caixa de
    seleção. Aconteceu de verdade: grupos como P24.1/P24.1A/P24.2/P24.3
    colidiam no mesmo texto "P24" antes da correção em
    `metadata.crossable_variables` que passou a usar o código do grupo
    inteiro no rótulo, não só um pedaço cortado no primeiro ponto.
    """
    label_counts = Counter(e["label"] for e in entries)
    dups = {lbl: c for lbl, c in label_counts.items() if c > 1}
    if not dups:
        return ["Nenhum rótulo duplicado na caixa de seleção."]
    lines = [f"[ATENÇÃO] {len(dups)} rótulo(s) repetido(s) na caixa de seleção:"]
    for lbl, c in dups.items():
        keys = [e["key"] for e in entries if e["label"] == lbl]
        lines.append(f"  {lbl!r} aparece {c}x -> chaves distintas: {keys}")
    return lines


def _check_key_collision(entries: list[dict]) -> list[str]:
    """
    Chave de variável SR/indicador colidindo com código de grupo MR --
    mais grave que rótulo duplicado, porque o dict `options` que
    `app.py` monta (`{key: label}`) perde uma das duas entradas
    silenciosamente quando isso acontece (a última a ser inserida no
    dict vence, a outra some sem aviso nenhum). Nunca aconteceu de
    verdade no banco de produção até agora, mas vale checar sempre --
    é o tipo de bug que só aparece quando alguém percebe uma pergunta
    "sumida" da lista, difícil de notar de outra forma.
    """
    sr_keys = {e["key"] for e in entries if e["var_type"] in (VarType.SR, VarType.INDICATOR)}
    mr_keys = {e["key"] for e in entries if e["var_type"] == VarType.MR_OPTION}
    collision = sr_keys & mr_keys
    if not collision:
        return ["Nenhuma colisão de chave entre SR/indicador e grupo MR."]
    lines = [
        f"[ATENÇÃO] {len(collision)} colisão(ões) de chave entre variável SR/indicador "
        f"e grupo MR -- uma das duas está sumindo da caixa de seleção:"
    ]
    lines += [f"  {k!r}" for k in collision]
    return lines


def _check_orphan_rm_tag(meta: dict, short_names: list[str]) -> list[str]:
    """
    Coluna sem "-" no nome (nunca chega a ser candidata a MR pelo sinal
    estrutural) mas cujo rótulo completo traz a tag "(RM - ...)" --
    desde a correção em `classify_columns`, esse caso já é tratado como
    MR de 1 opção automaticamente. Esta checagem é uma rede de
    segurança pra confirmar que nenhum caso escapou (só teria sentido
    reportar se ainda estivesse classificado como SR apesar da tag,
    o que indicaria uma regressão na lógica de classificação).
    """
    orphans = []
    for name in short_names:
        if "-" in name:
            continue
        m = meta[name]
        if m.var_type != VarType.SR:
            continue
        tag = _RM_TAG_PATTERN.search(m.label)
        if tag and tag.group(1).upper() == "RM":
            orphans.append(name)

    if not orphans:
        return ["Nenhuma variável com tag RM sem '-' no nome ficou classificada como SR."]
    lines = [
        f"[ATENÇÃO] {len(orphans)} variável(is) com tag (RM - ...) no rótulo, mas sem "
        f"'-' no nome, ainda classificada(s) como resposta única -- não deveria acontecer "
        f"mais, vale investigar:"
    ]
    lines += [f"  {n!r} -> {meta[n].label[:100]}" for n in orphans]
    return lines


# ══════════════════════════════════════════════════════════════════════
#  EXPORTAÇÃO E RELATÓRIO
# ══════════════════════════════════════════════════════════════════════

def _write_csv(out_path: Path, entries: list[dict], groups: dict) -> None:
    """
    Uma linha por variável cruzável -- pra bloco MR, lista as colunas
    cruas que compõem o grupo, separadas por " | ", pra revisão manual no
    Excel de se a agregação faz sentido pergunta por pergunta.
    """
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


def audit(path: str, sheet_name: str = "Dados") -> None:
    """
    Ponto de entrada: carrega o banco (parquet ou xlsx, decidido pela
    extensão), classifica, roda as três checagens de incoerência, grava
    o CSV de auditoria, e imprime a lista completa de variáveis
    cruzáveis no terminal.
    """
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

    for check in (
        _check_duplicate_labels(entries),
        _check_key_collision(entries),
        _check_orphan_rm_tag(meta, short_names),
    ):
        print("\n".join(check) + "\n")

    out_path = path_obj.with_name("variaveis_cruzamento.csv")
    _write_csv(out_path, entries, groups)
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
