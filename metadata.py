"""
Classificador de metadados para dados tabulares de pesquisa.

Por que isso existe como módulo separado: um df de pesquisa real (vimos isso no
seu arquivo de exemplo, que tinha 1164 colunas) não pode ser cruzado como se
toda coluna fosse uma categórica simples. O SPSS/Quantum sabe a diferença entre
uma pergunta de resposta única e uma de múltipla escolha porque alguém digitou
essa informação manualmente na sintaxe. Aqui, em vez de exigir que você catalogue
1164 variáveis à mão, inferimos o tipo a partir de três sinais que já existem
nos seus dados de origem:

1. Nome curto com "-" (ex.: "P2.1a-Cisterna") -> bloco de múltipla resposta.
   Cada opção é uma coluna dummy: célula preenchida = opção marcada,
   célula vazia = opção não marcada. Confirmado no seu arquivo: dentro de um
   mesmo bloco, uma pessoa pode ter mais de uma coluna preenchida.
2. Nome curto contendo "_media" -> companion numérico de uma variável de
   escala/ordinal (ex.: "P1.3_media"). Serve para gerar a linha de "Média" do
   banner, no mesmo padrão que já aparece nas suas planilhas SPC.
3. Rótulo completo (linha 1 do cabeçalho) terminando em "_c" -> indicador
   pré-categorizado (ex.: "...(IACOM)_c"). Esses indicadores não deveriam ter
   célula vazia -- confirmamos isso no seu arquivo (13 de 14 com 0% de nulo).
   Quando aparecem, o "não aplicável" já vem como texto ("N/A - não entrou em
   contato"), não como célula vazia. A taxa de N/A varia muito de indicador
   para indicador (de 0% a 99% no seu exemplo), por isso a regra de base
   (incluir ou excluir o N/A) é sugerida por variável, não fixa globalmente.

Tudo que não cai em nenhum desses três padrões vira "SR" (resposta única
categórica) por padrão -- é o caso mais simples e mais comum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class VarType(str, Enum):
    SR = "single_response"          # categórica de resposta única
    MR_OPTION = "multi_response"    # uma coluna-opção dentro de um bloco de múltipla escolha
    SCALE_MEDIA = "scale_media"     # companion numérico (_media) de uma variável ordinal
    INDICATOR = "indicator"         # indicador pré-categorizado, com N/A textual possível
    IDENTIFIER = "identifier"       # ID, PESO -- não deve entrar como variável cruzável de conteúdo
    WEIGHT = "weight"               # coluna de peso amostral


@dataclass
class VariableMeta:
    name: str                      # nome curto -- é o nome real da coluna no df
    label: str                     # rótulo completo (linha 1 do Excel), para exibição no banner
    var_type: VarType
    mr_group: Optional[str] = None      # código do bloco, só para MR_OPTION (ex.: "P2.1a")
    mr_option_label: Optional[str] = None  # texto da opção, só para MR_OPTION (ex.: "Cisterna")
    scale_base: Optional[str] = None    # nome da coluna categórica irmã, só para SCALE_MEDIA
    na_rate: Optional[float] = None     # % de valores "N/A - ..." explícitos, só calculado para INDICATOR


# Nomes de colunas que identificam o respondente ou carregam peso amostral --
# nunca devem aparecer como opção de cruzamento de conteúdo no filtro da UI.
_IDENTIFIER_NAMES = {"ID"}
_WEIGHT_NAMES = {"PESO", "WEIGHT", "PESO_AMOSTRAL"}

_MR_PATTERN = re.compile(r"^(?P<code>.+?)-(?P<option>.+)$")
_MEDIA_PATTERN = re.compile(r"^(?P<base>.+?)_media(?:\s*:\s*.*)?$", re.IGNORECASE)
_INDICATOR_LABEL_PATTERN = re.compile(r"_c\s*$")
_NA_TEXT_PATTERN = re.compile(r"^N/A\b", re.IGNORECASE)
_RM_TAG_PATTERN = re.compile(r"\((RU|RM)\s*-\s*(ESPONT[ÂA]NEA|ESTIMULADA)\)", re.IGNORECASE)


def load_raw_with_double_header(path: str, sheet_name: str = 0) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Lê um Excel no padrão que você descreveu: linha 1 = pergunta completa,
    linha 2 = código curto, linha 3 em diante = dados.

    Devolve (df_com_nomes_curtos, rotulos_completos, nomes_curtos).

    Por que não descartamos a linha 1: ela vira o dicionário de rótulos que o
    banner usa para exibir "Avaliação da concessionária" em vez de "IAG" no
    cabeçalho final -- sem isso a tabela fica ilegível para quem não é da
    equipe técnica.

    Nomes curtos duplicados (aconteceu no seu arquivo de exemplo -- 4 casos)
    são desambiguados com um sufixo numérico, para não quebrar o pandas
    silenciosamente tratando a coluna como DataFrame em vez de Series.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    full_labels = raw.iloc[0].astype(str).tolist()
    short_names_orig = raw.iloc[1].astype(str).tolist()

    seen: dict[str, int] = {}
    short_names: list[str] = []
    for name in short_names_orig:
        if name in seen:
            seen[name] += 1
            short_names.append(f"{name}__dup{seen[name]}")
        else:
            seen[name] = 0
            short_names.append(name)

    data = raw.iloc[2:].reset_index(drop=True)
    data.columns = short_names
    return data, full_labels, short_names


def load_parquet_with_labels(parquet_path: str, labels_path: str | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Contraparte de load_raw_with_double_header, lendo o resultado de
    convert_to_parquet.py. Existe pra manter classify_columns (e todo o
    resto do pipeline) agnóstico a qual formato originou os dados -- nada
    depois desse ponto precisa saber se veio de xlsx ou parquet.
    """
    import json
    from pathlib import Path

    parquet_path_obj = Path(parquet_path)
    labels_path_obj = Path(labels_path) if labels_path else parquet_path_obj.with_suffix("").with_suffix(".labels.json")

    data = pd.read_parquet(parquet_path_obj)
    with open(labels_path_obj, "r", encoding="utf-8") as f:
        saved = json.load(f)
    return data, saved["full_labels"], saved["short_names"]


def _split_stem_option(code: str, label: str) -> tuple[str, str]:
    """
    Remove o prefixo do código (ex.: "P24.1. ") do começo do rótulo, depois
    separa em (enunciado_compartilhado, texto_da_opção) cortando no primeiro
    ":" ou ";" -- os dois separadores usados no dado real entre a pergunta e
    a opção específica. Sem separador, tudo vira enunciado e a opção fica
    vazia. Aspas soltas ao redor do texto da opção (artefato visto no dado
    real, ex. '"Porque não acho bom"') são removidas.
    """
    text = re.sub(rf"^{re.escape(code)}\.?\s*", "", label)
    cut = min((i for i in (text.find(":"), text.find(";")) if i != -1), default=-1)
    if cut == -1:
        return text.strip(), ""
    stem = text[:cut].strip()
    option = text[cut + 1:].strip().strip("\"'").strip()
    return stem, option


def classify_columns(
    data: pd.DataFrame,
    full_labels: list[str],
    short_names: list[str],
) -> dict[str, VariableMeta]:
    """
    Aplica os sinais de padrão descritos no docstring do módulo e devolve
    um dicionário nome_curto -> VariableMeta.

    A ordem de checagem importa: MR primeiro (sinal mais específico e mais
    barato de errar), depois _media, depois indicador via rótulo, sobra SR.

    Ter "-" no nome curto NÃO É SUFICIENTE pra ser múltipla resposta -- dado
    real mostrou item de bateria/grade (várias perguntas de resposta única
    em sequência, ex. "P27.3.1-Relacionamento com o cliente?",
    "P27.3.2-Recapeamento do asfalto?"...) usando o mesmo formato "código-
    descrição" que um bloco MR de verdade, só que cada código ali é ÚNICO --
    não se repete em nenhuma outra coluna. Um bloco MR de verdade sempre
    tem o mesmo código como prefixo de várias colunas (uma por opção
    marcável). Por isso: só vira MR_OPTION se (a) o rótulo completo trouxer
    a tag "(RM - ...)" explicitamente, OU, na ausência de tag, (b) o código
    antes do primeiro "-" se repetir em 2+ colunas. Testado contra 726
    colunas do dado real que tinham a tag pra conferir: essa regra bateu em
    724 (99,7%); as 2 exceções (ex.: "P36aa"/"P36ab", cada sufixo de letra
    grudado sem separador) foram resolvidas pela tag em si, que tem
    prioridade quando existe.

    Uma terceira situação: coluna SEM "-" no nome (então nunca passa pelo
    bloco acima), mas com a tag "(RM - ...)" no rótulo completo -- vira um
    grupo MR de UMA coluna só. Isso acontece quando, num estudo específico,
    só uma opção de uma bateria de múltipla resposta teve alguma menção (as
    demais simplesmente não entraram no export por incidência zero) -- é
    dado incompleto por baixa incidência, não erro, e tratar como MR de 1
    opção não quebra nada no motor de cruzamento.
    """
    label_by_name = dict(zip(short_names, full_labels))

    code_counts: dict[str, int] = {}
    for name in short_names:
        m = _MR_PATTERN.match(name)
        if m:
            code_counts[m.group("code")] = code_counts.get(m.group("code"), 0) + 1

    meta: dict[str, VariableMeta] = {}

    for name in short_names:
        label = label_by_name[name]

        if name in _IDENTIFIER_NAMES:
            meta[name] = VariableMeta(name, label, VarType.IDENTIFIER)
            continue
        if name.upper() in _WEIGHT_NAMES:
            meta[name] = VariableMeta(name, label, VarType.WEIGHT)
            continue

        mr_match = _MR_PATTERN.match(name)
        if mr_match:
            tag = _RM_TAG_PATTERN.search(label)
            if tag is not None:
                is_mr = tag.group(1).upper() == "RM"
            else:
                is_mr = code_counts[mr_match.group("code")] >= 2
            if is_mr:
                meta[name] = VariableMeta(
                    name, label, VarType.MR_OPTION,
                    mr_group=mr_match.group("code"),
                    mr_option_label=mr_match.group("option"),
                )
                continue
            # tinha "-" mas não é MR de verdade (item de bateria) -- cai pro
            # resto da função como um SR normal, tratando o "-" como parte
            # do texto do rótulo, não como separador código/opção.
        else:
            tag = _RM_TAG_PATTERN.search(label)
            if tag is not None and tag.group(1).upper() == "RM":
                stem, option = _split_stem_option(name, label)
                meta[name] = VariableMeta(
                    name, label, VarType.MR_OPTION,
                    mr_group=name,
                    mr_option_label=option or stem,
                )
                continue

        media_match = _MEDIA_PATTERN.match(name)
        if media_match:
            meta[name] = VariableMeta(
                name, label, VarType.SCALE_MEDIA,
                scale_base=media_match.group("base"),
            )
            continue

        if _INDICATOR_LABEL_PATTERN.search(label.strip()):
            col = data[name]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            # .str.match com na=False: célula vazia (nula de verdade, ex.: IDA
            # 100% nula) não é "N/A" textual -- são fenômenos diferentes, e
            # tratar os dois como a mesma coisa aqui inflaria a taxa de N/A
            # de indicadores que simplesmente não existem nesse estudo.
            na_rate = col.astype("string").str.match(_NA_TEXT_PATTERN, na=False).mean()
            meta[name] = VariableMeta(name, label, VarType.INDICATOR, na_rate=na_rate)
            continue

        meta[name] = VariableMeta(name, label, VarType.SR)

    return meta


def get_label(meta: dict[str, VariableMeta], key: str) -> str:
    """
    Rótulo de exibição para uma chave de variável -- que pode ser um nome de
    coluna comum ou o código de um bloco MR (que não é, ele mesmo, uma chave
    de `meta`; `meta` é indexado por coluna, não por bloco).
    """
    if key in meta:
        return meta[key].label
    members = [m for m in meta.values() if m.var_type == VarType.MR_OPTION and m.mr_group == key]
    if members:
        return f"{key} — {_mr_group_stem(members)}"
    return key


def mr_groups(meta: dict[str, VariableMeta]) -> dict[str, list[VariableMeta]]:
    """Agrupa as VariableMeta do tipo MR_OPTION pelo código do bloco (ex.: 'P2.1a')."""
    groups: dict[str, list[VariableMeta]] = {}
    for m in meta.values():
        if m.var_type == VarType.MR_OPTION:
            groups.setdefault(m.mr_group, []).append(m)
    return groups


def crossable_variables(meta: dict[str, VariableMeta]) -> list[dict]:
    """
    Lista "achatada" pronta para popular os selectboxes da UI: cada bloco MR
    vira uma única entrada (não uma por opção), cada SR/indicador vira uma
    entrada própria. IDENTIFIER e WEIGHT ficam de fora -- não fazem sentido
    como variável de conteúdo a cruzar.

    O rótulo de um grupo MR sempre começa com o próprio código do grupo
    (ex.: "P24.1") -- isso é o que garante que dois grupos nunca mostrem o
    mesmo texto na caixa de seleção, mesmo quando o rótulo completo original
    é parecido. Dado real mostrou grupos como P24.1/P24.1A/P24.2/P24.3
    colidindo no mesmo texto "P24" quando o rótulo tentava cortar no primeiro
    ponto -- "P24.1." tem ponto dentro do próprio código, não só no fim da
    frase, então cortar ali destrói justamente a parte que diferencia os
    grupos.
    """
    entries: list[dict] = []
    groups = mr_groups(meta)
    seen_mr: set[str] = set()

    for m in meta.values():
        if m.var_type in (VarType.IDENTIFIER, VarType.WEIGHT, VarType.SCALE_MEDIA):
            continue
        if m.var_type == VarType.MR_OPTION:
            if m.mr_group in seen_mr:
                continue
            seen_mr.add(m.mr_group)
            entries.append({
                "key": m.mr_group,
                "label": f"[Múltipla escolha] {m.mr_group} — {_mr_group_stem(groups[m.mr_group])}",
                "var_type": VarType.MR_OPTION,
            })
            continue
        tag = "[Indicador] " if m.var_type == VarType.INDICATOR else ""
        entries.append({"key": m.name, "label": f"{tag}{m.label}", "var_type": m.var_type})

    return entries


def _mr_group_stem(members: list[VariableMeta]) -> str:
    """
    Extrai um pedaço legível do enunciado da pergunta a partir do rótulo
    completo das opções de um grupo, pra complementar o código (que sozinho
    já garante unicidade, mas não diz nada sobre o conteúdo da pergunta).

    Quando o grupo tem UMA SÓ opção, cortar no separador apaga justamente a
    parte que diferencia esse grupo de outro parecido (dado real: "P36aa" e
    "P36ab" têm o mesmo enunciado compartilhado "Você tentou contato por
    telefone", e só o texto da opção -- "Ligação telefônica (voz)" vs.
    "Mensagem de texto" -- os diferencia). Nesse caso o texto da opção entra
    de volta no rótulo.
    """
    m = members[0]
    stem, _ = _split_stem_option(m.mr_group, m.label)
    if len(members) == 1 and m.mr_option_label:
        stem = f"{stem} ({m.mr_option_label})"
    return stem
