"""
crosstab_engine.py — Motor de cruzamento ponderado, com suporte nativo a
múltipla resposta.

PAPEL NO PROJETO
-----------------
Recebe o df já classificado por `metadata.py` e faz a conta de verdade:
cruza duas variáveis (stub x banner), pondera por peso amostral, monta a
tabela no formato NA/%LINHA/%COLUNA, e decide quando avisar que um número
merece leitura cautelosa (base pequena, cobertura baixa/concentrada).
`app.py` só chama `build_banner` e desenha o resultado -- toda a
estatística mora aqui

DECISÃO CENTRAL DE ARQUITETURA -- O FORMATO "LONGO"
------------------------------------------------------
Em vez de pivotar o df inteiro (caro -- com 1164 colunas seria bobagem
materializar tudo de uma vez), normalizamos cada variável -- SR, indicador
ou bloco MR -- para o MESMO formato "longo" só na hora em que ela é
selecionada no cruzamento:

    resp_id | category | weight

Uma SR vira uma linha por respondente. Um bloco MR vira várias linhas por
respondente (uma por opção marcada) -- esse é o "pivotar que se faz no
Power BI", só que automático e sob demanda (`to_long`, mais abaixo).
Depois disso, cruzar qualquer combinação SR x SR, SR x MR, MR x MR é o
mesmo merge + groupby, porque os dois lados já estão no mesmo formato. É
essa normalização que elimina o retrabalho manual por pergunta.

TRÊS CONVENÇÕES ESTATÍSTICAS FIXADAS AQUI, E O PORQUÊ
---------------------------------------------------------
1. Percentual usa peso (PESO), mas "Base Amostra"/NA reporta N não
   ponderado. É o padrão de mercado: o peso corrige a leitura do %, mas
   quem decide se uma célula é confiável estatisticamente é o tamanho
   real da amostra, não o tamanho ponderado (que pode parecer maior ou
   menor que a realidade).

2. A base de uma variável MR é "respondentes com pelo menos 1 opção
   marcada no bloco" -- não dá para distinguir com certeza, só pelos
   dados, quem foi filtrado por lógica de pulo de quem foi perguntado e
   marcou zero opções. Isso é uma limitação real, documentada aqui, não
   escondida. Se o bloco tiver uma opção explícita tipo "Nenhuma", ela
   funciona como o marcador de "perguntado e respondeu nada", e a base
   fica correta automaticamente.

3. A base do %LINHA é o "universo da pergunta de banner", não o Total
   do stub. O banco junta anos/operações e uma pergunta muitas vezes NÃO
   EXISTE pra um desses grupos (ex.: IACOM em PPPs). Esse vazio estrutural
   não é não-resposta: dividir por ele fazia o %LINHA não fechar 100% sem
   nenhuma pista do porquê. `infer_question_universe` tira da base os
   grupos onde a pergunta quase não tem resposta (< 5%), mas mantém quem
   foi perguntado e não respondeu -- esse vazio é real e continua
   visível. A regra é mecânica (não conhece o questionário), então a base
   usada vai escrita em `BannerBlock.universe_note` e `use_universe=False`
   restaura o Total inteiro.

ADVERTÊNCIA CONHECIDA -- STUB DE MÚLTIPLA RESPOSTA
------------------------------------------------------
Quando a variável de STUB (linha) é MR, uma pessoa com 2+ seleções conta
em 2+ linhas da tabela -- por isso %COLUNA pode somar mais de 100% nesse
caso, e "Base Amostra"/NA soma mais que o total de respondentes. Isso é
correto matematicamente (reflete múltipla seleção de verdade), não é bug,
mas é diferente do caso comum (stub de resposta única, onde tudo soma
100%) e vale ter em mente ao interpretar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metadata import VariableMeta, VarType, get_label, sort_categories

_NA_TEXT_PATTERN = r"^N/A\b"


# ══════════════════════════════════════════════════════════════════════
#  PESO E ACESSO SEGURO A COLUNA
# ══════════════════════════════════════════════════════════════════════

def get_weights(data: pd.DataFrame, meta: dict[str, VariableMeta], use_weighting: bool = True) -> pd.Series:
    """
    Devolve a série de pesos alinhada ao índice do df. Peso 1.0 pra toda
    linha se não houver coluna PESO, ou se `use_weighting=False`.

    `use_weighting=False` existe pra reproduzir leitura NÃO ponderada
    (contagem bruta) -- decisão de metodologia confirmada por Ro em
    15/09: "no geral costuma usar os dados não ponderado". Devolver
    peso uniforme em vez de bifurcar a lógica de cálculo em cada função
    consumidora (`to_long`, `_build_single_block`, `compute_index_trend`
    etc.) mantém o pipeline ponderado e não-ponderado idêntico em tudo
    MENOS o valor do peso -- reduz risco de os dois caminhos divergirem
    silenciosamente no futuro.
    """
    if not use_weighting:
        return pd.Series(1.0, index=data.index)
    weight_names = [m.name for m in meta.values() if m.var_type == VarType.WEIGHT]
    if not weight_names:
        return pd.Series(1.0, index=data.index)
    col = data[weight_names[0]]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce").fillna(1.0)


def _get_series(data: pd.DataFrame, name: str) -> pd.Series:
    """
    `data[name]` protegido contra nome curto duplicado -- quando o Excel de
    origem tem duas colunas com o mesmo nome curto (aconteceu 4x no banco
    real), `metadata.load_raw_with_double_header` desambigua com sufixo
    `__dupN`, mas se algum outro caminho de leitura não desambiguar,
    `data[name]` devolveria um DataFrame de 2 colunas em vez de uma
    Series. Pega sempre a primeira, pra nunca quebrar silenciosamente
    mais adiante num `.notna()`/`.astype()` que só faz sentido em Series.
    """
    col = data[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col


def get_column_series(data: pd.DataFrame, name: str) -> pd.Series:
    """Wrapper público de `_get_series` -- pra app.py/indices.py não precisarem importar nome privado do módulo."""
    return _get_series(data, name)


# ══════════════════════════════════════════════════════════════════════
#  NORMALIZAÇÃO PRO FORMATO LONGO
# ══════════════════════════════════════════════════════════════════════

def _mr_selected_mask(col: pd.Series) -> pd.Series:
    """
    "Essa opção foi marcada?" pra uma coluna-opção de bloco MR -- que
    aparece em dois formatos possíveis:

    - Original: texto da opção repetido quando marcado, célula vazia
      quando não.
    - Otimizado por `convert_to_parquet.py`: booleano puro, True =
      marcado (ver `_optimize_memory` lá -- criado porque guardar o texto
      da opção repetido em toda linha marcada gasta memória à toa, quando
      só a presença importa aqui).

    Os dois precisam funcionar aqui porque `list_variables.py` e testes
    locais às vezes leem xlsx cru direto, sem passar pela otimização. Um
    booleano puro nunca é "nulo" -- `.notna()` nele sempre devolveria
    True e marcaria todo mundo como selecionado, por isso o branch por
    dtype em vez de só chamar `.notna()` sempre.
    """
    if pd.api.types.is_bool_dtype(col):
        return col.fillna(False).astype(bool)
    return col.notna()


_ISDE_LABEL_FIX_VALUE = "Média próxima ou igual a 5 (>=4,5)"
"""
Bug de exportação conhecido na categórica `ISDE`: a faixa mais alta
(nota 5) chega do Excel/Quantum como o literal booleano "False" em vez
do rótulo esperado -- provavelmente uma fórmula tipo "nota>=4.5" que
nunca foi trocada pelo texto nas outras 4 faixas.

Confirmado célula a célula contra `P23_media` (o companion numérico
real do ISDE, ver `_MEDIA_ALIASES` em indices.py): das 8.890 linhas
com ISDE=="False", 8.768 (98,6%) têm P23_media==5.0 -- essas são
corrigidas aqui. As 122 restantes (1,4%) têm P23_media != 5.0 e são
100% concentradas na onda 2023 -- padrão diferente, possivelmente uma
causa distinta específica daquele lote. NÃO são corrigidas aqui por
falta de evidência de que é o mesmo bug; ficam como "False" na tela de
propósito, pra continuar visivelmente estranho até alguém investigar
a origem própria dessas 122 linhas.

Não afeta o ÍNDICE numérico (já usa P23_media, nunca lê ISDE direto) --
só a variável categórica ISDE quando usada como stub/banner na aba
Cruzamento, que é onde o texto errado aparecia na tela.
"""


def _fix_known_label_bugs(key: str, col: pd.Series, media_col: pd.Series | None = None) -> pd.Series:
    """
    Aplica correção de rótulo conhecida (hoje só ISDE) se `key` for uma
    das afetadas -- passa direto pras demais. `media_col`, quando
    fornecida, restringe a correção só às linhas onde o companion
    numérico confirma o valor esperado (ver docstring de
    `_ISDE_LABEL_FIX_VALUE`); sem ela, a correção não é aplicada -- é
    preferível manter "False" visível a arriscar rótulo errado sem a
    validação cruzada.
    """
    if key != "ISDE" or media_col is None:
        return col
    fixed = col.astype(str)
    confirmed = (fixed == "False") & (media_col == 5.0)
    fixed = fixed.where(~confirmed, _ISDE_LABEL_FIX_VALUE)
    return fixed


def to_long(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    key: str,
    weights: pd.Series,
    na_handling: str = "keep",
) -> pd.DataFrame:
    """
    Normaliza uma variável (SR, indicador ou bloco MR identificado por
    `key`) para o formato longo `resp_id | category | weight` -- ver
    docstring do módulo pro porquê desse formato ser o alicerce do motor.

    Uma SR/indicador vira no máximo uma linha por respondente. Um bloco MR
    vira zero, uma ou várias linhas por respondente (uma por opção
    marcada) -- é aqui, dentro dessa função, que o "pivotar" que se faria
    manualmente acontece, sob demanda.

    `na_handling`: 'keep' mantém a categoria "N/A - ..." de indicadores
    como categoria própria; 'exclude' remove essas linhas (a % passa a
    ser só de quem de fato respondeu). Não se aplica a SR nem a MR -- lá
    "não preenchido" já significa "não elegível", tratado em
    `eligible_respondents`, não é uma categoria "N/A" de texto.
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
    if key == "ISDE" and "P23_media" in data.columns:
        col = _fix_known_label_bugs(key, col, media_col=_get_series(data, "P23_media"))
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
    universo que deveria compor a base do banner para essa variável,
    antes de cruzar com qualquer outra coisa.

    Pra SR/indicador: quem não é nulo (e, se `na_handling='exclude'`,
    também não é "N/A - ..." textual). Pra MR: união de quem marcou
    QUALQUER opção do bloco -- ver a limitação documentada no módulo
    sobre não dar pra distinguir "pulou a pergunta" de "respondeu e não
    marcou nada".
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


def variables_with_data(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    keys: list[str],
    na_handling: str = "keep",
) -> set[str]:
    """
    Subconjunto de `keys` (variáveis cruzáveis: SR, indicador ou código de
    bloco MR) que têm PELO MENOS UM respondente elegível em `data`.

    Existe pra esconder, nas listas de stub/banner, perguntas que não têm
    nenhuma resposta no recorte atual (ex.: com OPERACAO = PPPs, os índices
    IACOM, IMC... somem -- não existem em PPPs). Usa a mesma definição de
    "elegível" de `eligible_respondents` (inclusive `na_handling='exclude'`
    para indicadores), de modo que "aparece na lista" e "o cruzamento tem
    base" nunca discordem. Vetorizado por tipo de coluna em vez de chamar
    `eligible_respondents` variável a variável -- essa função roda a cada
    mudança de filtro, com centenas de variáveis.
    """
    mr_cols: dict[str, list[str]] = {}
    for m in meta.values():
        if m.var_type == VarType.MR_OPTION and m.mr_group is not None:
            mr_cols.setdefault(m.mr_group, []).append(m.name)

    result: set[str] = set()
    plain_keys, plain_cols = [], []
    for key in keys:
        if key in mr_cols:
            continue
        m = meta[key]
        if m.var_type == VarType.INDICATOR and na_handling == "exclude":
            values = _get_series(data, m.name).dropna().unique()
            if any(re.match(_NA_TEXT_PATTERN, str(v), flags=re.IGNORECASE) is None for v in values):
                result.add(key)
        else:
            plain_keys.append(key)
            plain_cols.append(m.name)

    if plain_cols:
        has_any = data[plain_cols].notna().any(axis=0)
        # nomes duplicados no df (sufixo __dupN evita, mas não custa proteger): alinha por posição
        result.update(k for k, flag in zip(plain_keys, has_any.to_numpy()) if flag)

    for key in keys:
        if key in mr_cols:
            cols = mr_cols[key]
            block = data[cols]
            if all(pd.api.types.is_bool_dtype(block[c]) for c in cols):
                found = bool(block.to_numpy().any())
            else:
                found = any(bool(_mr_selected_mask(_get_series(data, c)).any()) for c in cols)
            if found:
                result.add(key)
    return result


# ══════════════════════════════════════════════════════════════════════
#  ESTRUTURA DE RESULTADO
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BannerBlock:
    """
    Resultado de cruzar UMA variável de banner contra o stub (mais o
    bloco especial "Total", sem cruzamento nenhum -- ver `build_banner`).
    Uma lista de `BannerBlock` (sempre com "Total" primeiro) é a moeda
    comum entre `build_banner` e as funções de formatação/exportação
    mais abaixo.
    """
    banner_key: str
    banner_label: str
    pct: pd.DataFrame              # linhas = categorias do stub, colunas = categorias do banner, valores = % coluna
    cell_n: pd.DataFrame           # N não ponderado por célula (mesma forma de pct) -- vira a linha "NA"
    cell_weighted: pd.DataFrame    # N ponderado por célula, pré-divisão -- usado pra calcular %LINHA
    base_n: pd.Series              # N não ponderado por categoria do banner
    small_n_flag: pd.Series        # True onde base_n < limiar
    coverage_warning: str | None = None  # ver _check_coverage / _check_base_coverage
    # --- base do %LINHA (ver `infer_question_universe`) ---------------------
    # Denominador do %LINHA DESTE bloco, por categoria do stub. É guardado
    # no próprio bloco (e não recalculado a partir do Total em cada lugar
    # que desenha tabela/gráfico/export) pra que tabela, gráfico e CSV
    # nunca possam divergir: todos leem `linha_pct_frame`.
    row_base_weighted: pd.Series | None = None   # N ponderado da base, por categoria do stub
    row_base_n: pd.Series | None = None          # N não ponderado (pessoas distintas), por categoria do stub
    universe_applied: bool = False               # True = base reduzida ao "universo da pergunta"
    universe_excluded_n: int = 0                 # pessoas fora da base por a pergunta não existir pro grupo delas
    universe_note: str | None = None             # texto pronto ("Base do %LINHA: ... exclui ...") ou None
    universe_excluded_groups: list[tuple[str, int]] | None = None  # lista completa (grupo, pessoas) pra quem quer detalhe
    base_people: int = 0                         # pessoas DISTINTAS na base do %LINHA (diferente de row_base_n.sum() quando o stub é MR)


# ══════════════════════════════════════════════════════════════════════
#  UNIVERSO DA PERGUNTA (BASE DO %LINHA)
# ══════════════════════════════════════════════════════════════════════
# O problema que isto resolve: o banco junta vários anos/operações, e uma
# pergunta frequentemente NÃO EXISTE pra um desses grupos (ex.: o índice
# IACOM não existe em PPPs). Quem não foi perguntado aparece como "vazio"
# no banco, igual a quem foi perguntado e não respondeu -- mas são coisas
# diferentes pra leitura: o primeiro não é resposta faltando, é pergunta
# que não existiu. Se o %LINHA divide pelo grupo inteiro, esses "vazios
# estruturais" entram no denominador e as barras/linhas não fecham 100%,
# sem nenhuma pista do porquê.
#
# A regra é mecânica de propósito (o app não conhece o desenho dos
# questionários -- ver `infer_question_universe`): um grupo (ex.:
# OPERACAO=PPPs) é "fora do universo" se a pergunta tem resposta em
# outros grupos do MESMO tipo de recorte mas quase nenhuma nesse.

UNIVERSE_KEYS_DEFAULT: tuple[str, ...] = ("ANO", "OPERACAO", "UF", "DIRETORIA", "REGIONAL", "PUBLICO")
"""
Colunas que definem "grupo" pra decidir onde uma pergunta existe. Cada
uma é testada sozinha e, quando `_UNIVERSE_PAIR_KEY` está entre elas, também
cruzada com ela (ANO x OPERACAO, ANO x UF...) -- o desenho do estudo costuma
mudar de uma onda pra outra (pergunta que existiu só em 2023 em uma
operação), e uma chave sozinha não enxerga isso. Colunas que não existem
no banco carregado são ignoradas.
"""
_UNIVERSE_PAIR_KEY = "ANO"
_UNIVERSE_MAX_COVERAGE = 0.05  # abaixo disso (% de pessoas com resposta no grupo) = pergunta não existe ali
_UNIVERSE_MIN_GROUP_N = 30     # grupo menor que isso não é julgado (pouca gente pra concluir qualquer coisa)


@dataclass
class _KeyGroups:
    """Um tipo de recorte (ex.: OPERACAO, ou ANO x UF): código de grupo por linha + como descrevê-lo."""
    names: tuple[str, ...]
    codes: np.ndarray            # int64 por linha; -1 = alguma chave do recorte é nula
    sizes: np.ndarray            # nº de pessoas de cada grupo (índice = código)
    categories: list[list]       # categorias de cada chave, na ordem de `names`

    def describe(self, code: int) -> str:
        """Texto legível do grupo -- ex.: "OPERACAO = PPPs" ou "ANO = 2022 e UF = BAHIA"."""
        parts, rest = [], int(code)
        for name, cats in zip(reversed(self.names), reversed(self.categories)):
            rest, idx = divmod(rest, len(cats))
            parts.append(f"{name} = {cats[idx]}")
        return " e ".join(reversed(parts))


def build_universe_key_groups(data: pd.DataFrame, universe_keys: tuple[str, ...] | None = None) -> list[_KeyGroups]:
    """
    Pré-calcula os códigos de grupo das chaves de universo. Separado de
    `infer_question_universe` pra ser feito UMA vez por `build_banner` e
    reaproveitado por todas as variáveis de banner (a parte cara é
    fatorizar as colunas-chave, não testar cada pergunta).
    """
    keys = [k for k in (universe_keys or UNIVERSE_KEYS_DEFAULT) if k in data.columns]
    coded: dict[str, tuple[np.ndarray, list]] = {}
    for k in keys:
        cat = _get_series(data, k).astype("category")
        coded[k] = (cat.cat.codes.to_numpy().astype(np.int64), list(cat.cat.categories))

    key_sets: list[tuple[str, ...]] = [(k,) for k in keys]
    if _UNIVERSE_PAIR_KEY in keys:
        key_sets += [(_UNIVERSE_PAIR_KEY, k) for k in keys if k != _UNIVERSE_PAIR_KEY]

    groups = []
    for names in key_sets:
        codes, cats_list, invalid = None, [], None
        for name in names:
            c, cats = coded[name]
            cats_list.append(cats)
            if codes is None:
                codes, invalid = c.copy(), c < 0
            else:
                codes = codes * max(len(cats), 1) + c
                invalid |= c < 0
        if codes is None:
            continue
        codes[invalid] = -1
        n_groups = int(np.prod([max(len(c), 1) for c in cats_list]))
        sizes = np.bincount(codes[codes >= 0], minlength=n_groups)
        groups.append(_KeyGroups(names=names, codes=codes, sizes=sizes, categories=cats_list))
    return groups


def infer_question_universe(
    key_groups: list[_KeyGroups],
    eligible_mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    "Onde essa pergunta existe?" -- devolve (máscara do universo, grupos
    excluídos com a contagem de pessoas que cada um tirou).

    Regra (mecânica, sem conhecer o questionário): em cada tipo de recorte
    (OPERACAO, ANO, ANO x UF...), um grupo com >= 30 pessoas e MENOS de 5%
    de resposta é considerado "a pergunta não existe aqui" -- MAS só se
    houver pelo menos um outro grupo do mesmo recorte onde ela tem
    cobertura >= 5%. Sem essa condição, uma pergunta condicional que só ~3%
    de TODO MUNDO responde (ex.: "motivo de não conectar") teria todos os
    grupos "excluídos" e o universo viraria "só quem respondeu", o que é
    uma base legítima mas não é o que este aviso diz.

    Quem respondeu a pergunta NUNCA sai do universo, mesmo num grupo
    excluído -- senão o numerador de uma célula poderia passar do
    denominador (%LINHA > 100%).

    Limite assumido (documentado, não escondido): o que distingue "pergunta
    não existe no grupo" de "pouca gente respondeu" é o corte de 5%. Uma
    pergunta que existe mas quase ninguém responde num grupo seria tratada
    como inexistente. O app mostra quem foi tirado da base e permite voltar
    ao Total inteiro, justamente por isso.
    """
    n = len(eligible_mask)
    excluded = np.zeros(n, dtype=bool)
    parts: list[tuple[str, int]] = []
    for g in key_groups:
        valid = g.codes >= 0
        n_groups = len(g.sizes)
        elig_counts = np.bincount(g.codes[valid & eligible_mask], minlength=n_groups)
        judged = g.sizes >= _UNIVERSE_MIN_GROUP_N
        has_data = judged & (elig_counts >= _UNIVERSE_MAX_COVERAGE * g.sizes)
        absent = judged & ~has_data
        if not absent.any() or not has_data.any():
            continue
        safe_codes = np.where(valid, g.codes, 0)
        newly = valid & absent[safe_codes] & ~excluded & ~eligible_mask
        if not newly.any():
            continue
        per_group = np.bincount(g.codes[newly], minlength=n_groups)
        for code in np.flatnonzero(per_group):
            parts.append((g.describe(int(code)), int(per_group[code])))
        excluded |= newly
    return ~excluded, parts


# ══════════════════════════════════════════════════════════════════════
#  AVISOS DE COBERTURA
# ══════════════════════════════════════════════════════════════════════
# Dois problemas diferentes de "esse número pode enganar", nenhum dos
# dois pego pelo alerta de N pequeno (que só olha tamanho de base, não a
# FORMA como essa base está distribuída ou de onde ela vem).

def _check_coverage(stub_long_filtered: pd.DataFrame, stub_label: str, threshold: float = 0.9) -> str | None:
    """
    Detecta a base elegível pro cruzamento inteiro concentrada numa única
    categoria do stub. Aconteceu de verdade no banco de produção -- um
    bloco de múltipla resposta (motivo de não conectar à rede) só tinha
    respondente do ano de 2024, porque a pergunta simplesmente não foi
    feita nas outras ondas do estudo consolidado. O resultado (100% em
    2024, 0% nos outros anos, em TODAS as opções do bloco) é
    aritmeticamente correto, mas não descreve diferença de comportamento
    entre anos -- descreve em que ano a pergunta existiu. N pequeno não
    pega isso porque a base total pode ser grande (577 pessoas, no caso
    real); o problema é a base inteira estar num só balde do stub, não o
    tamanho dela.

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


def _check_base_coverage(
    banner_n: int,
    base_n: int,
    threshold: float = 0.9,
    universe_applied: bool = False,
) -> str | None:
    """
    Segundo tipo de aviso, diferente de `_check_coverage`: não é a base
    estar concentrada num balde só do stub, é a base elegível pra essa
    variável de banner ser bem menor que a base contra a qual o %LINHA é
    calculado -- ou porque a pergunta não foi feita pra todo mundo, ou
    porque parte de quem foi perguntado não respondeu. Achado real: "utiliza
    água de poço" tinha base de 64.216 contra Total de 85.201 (75,4%) --
    sem esse aviso, só dava pra perceber somando a linha Base Amostra na
    mão, que foi exatamente como esse caso apareceu.

    O texto depende de `universe_applied` porque a MESMA cobertura baixa
    significa coisas diferentes: com o universo aplicado, os grupos onde a
    pergunta nem existe já saíram da base, então o que sobra é não-resposta
    de verdade; sem ele, a base é o Total inteiro e o aviso precisa dizer
    que o %LINHA também usa esse Total (a versão anterior afirmava que "os
    percentuais estão certos", o que era falso pro %LINHA nesse caso).

    Texto em linguagem simples de propósito -- "lógica de pulo" é jargão
    de quem constrói questionário, não de quem lê o banner.
    """
    if base_n <= 0:
        return None
    coverage = banner_n / base_n
    if coverage >= threshold:
        return None
    if universe_applied:
        return (
            f"Dentro dos grupos onde essa pergunta existe, só {_format_br_number(banner_n, 0)} de {_format_br_number(base_n, 0)} "
            f"pessoas responderam ({coverage:.0%}). Quem não respondeu continua na base: "
            f"no %LINHA aparece como a parte que falta pra 100%. O %COLUNA vale só pra "
            f"quem respondeu."
        )
    return (
        f"Só {_format_br_number(banner_n, 0)} de {_format_br_number(base_n, 0)} pessoas do Total responderam essa "
        f"pergunta ({coverage:.0%}) -- provavelmente ela não foi feita "
        f"pra todo mundo. O %LINHA usa o Total inteiro como base (por isso "
        f"não fecha 100% entre as colunas); o %COLUNA vale só pra quem respondeu."
    )


def _universe_note(excluded_parts: list[tuple[str, int]], base_n: int, max_items: int = 3) -> str | None:
    """
    Texto da base do %LINHA quando o universo tirou gente: quantas pessoas
    ficaram, quem saiu e por quê. `None` se ninguém saiu (nada a explicar).
    Os `max_items` maiores grupos aparecem pelo nome; o resto vira
    "e mais N recortes" -- a lista completa fica em
    `BannerBlock.universe_excluded_n` / não cabe numa legenda.
    """
    if not excluded_parts:
        return None
    total_out = sum(c for _, c in excluded_parts)
    ordered = sorted(excluded_parts, key=lambda x: -x[1])
    shown = ", ".join(f"{name} ({_format_br_number(c, 0)})" for name, c in ordered[:max_items])
    more = f" e mais {len(ordered) - max_items} recorte(s)" if len(ordered) > max_items else ""
    return (
        f"Base do %LINHA: {_format_br_number(base_n, 0)} pessoas. Fora da base "
        f"({_format_br_number(total_out, 0)}): {shown}{more} -- grupos onde essa "
        f"pergunta quase não tem resposta (menos de 5%), ou seja, provavelmente não foi feita."
    )


# ══════════════════════════════════════════════════════════════════════
#  MOTOR DE CRUZAMENTO
# ══════════════════════════════════════════════════════════════════════

def _build_single_block(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    stub_key: str,
    banner_key: str,
    weights: pd.Series,
    na_handling: str,
    small_n_threshold: int,
    total_n: int,
    key_groups: list[_KeyGroups] | None = None,
) -> BannerBlock:
    """
    Cruza UMA variável de banner contra o stub. Chamada uma vez por
    variável selecionada em `build_banner` -- nunca chamada diretamente
    de fora deste módulo (por isso o nome com `_`).

    Passo a passo:
    1. Normaliza os dois lados pro formato longo (`to_long`).
    2. Restringe aos respondentes elegíveis pros DOIS lados ao mesmo
       tempo (`both_elig`) -- interseção, não união, porque a base de um
       cruzamento é sempre "quem podia responder as duas perguntas".
    3. `merge` dos dois formatos longos por `resp_id`: se qualquer um dos
       lados for MR (várias linhas por pessoa), o merge naturalmente gera
       o produto cartesiano das seleções dessa pessoa -- é assim que uma
       pessoa com 2 respostas na variável A e 2 na B conta corretamente
       nas 4 combinações, sem código especial pra múltipla resposta na
       hora do cruzamento em si.
    4. Agrupa por par de categoria (stub, banner) pra somar peso (%) e
       contar distinto (NA).
    5. Define a BASE DO %LINHA (`row_base_weighted`): por padrão, o Total
       do stub; com `key_groups` (universo ligado), só quem está nos grupos
       onde a pergunta de banner existe -- ver `infer_question_universe`.
       Quem está no universo mas não respondeu o banner continua na base
       (não-resposta de verdade fica visível, não é escondida).
    """
    stub_long = to_long(data, meta, stub_key, weights, na_handling)
    banner_long = to_long(data, meta, banner_key, weights, na_handling)

    stub_elig = eligible_respondents(data, meta, stub_key, na_handling)
    banner_elig = eligible_respondents(data, meta, banner_key, na_handling)
    both_elig = stub_elig.intersection(banner_elig)

    stub_long = stub_long[stub_long.resp_id.isin(both_elig)]
    banner_long = banner_long[banner_long.resp_id.isin(both_elig)]

    # Base por categoria do banner: respondentes distintos elegíveis para
    # o stub, dentro de cada categoria do banner. drop_duplicates por
    # resp_id+categoria evita contar a mesma opção 2x (não deveria
    # acontecer, mas é barato garantir).
    base_pairs = banner_long.drop_duplicates(["resp_id", "category"])
    base_weighted = base_pairs.groupby("category")["weight"].sum()
    base_n = base_pairs.groupby("category")["resp_id"].nunique()
    # Ordem de leitura (escala conhecida do melhor pro pior, indicador
    # 1->5, residual por último) em vez da ordem alfabética que sai do
    # groupby -- tudo daqui pra baixo reindexa por base_weighted.index,
    # então reordenar só ela já propaga pra cell_table/cell_n_table/pct.
    base_weighted = base_weighted.reindex(sort_categories(list(base_weighted.index)))
    base_n = base_n.reindex(base_weighted.index)

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

    # --- base do %LINHA ---------------------------------------------------
    # stub_long_all: TODO elegível do stub (antes de exigir resposta no
    # banner) -- o %LINHA divide por ele, não por `stub_long` (que já foi
    # cortado em `both_elig` acima e faria toda linha somar 100% por
    # construção, escondendo a não-resposta).
    stub_long_all = to_long(data, meta, stub_key, weights, na_handling)
    stub_long_all = stub_long_all[stub_long_all.resp_id.isin(stub_elig)]
    universe_excluded_n, universe_note, excluded_groups = 0, None, None
    if key_groups is not None:
        elig_mask = data.index.isin(banner_elig)
        in_universe, excluded_parts = infer_question_universe(key_groups, elig_mask)
        universe_ids = data.index[in_universe]
        stub_long_all = stub_long_all[stub_long_all.resp_id.isin(universe_ids)]
        stub_ids_all = stub_elig.intersection(universe_ids)
        universe_excluded_n = len(stub_elig) - len(stub_ids_all)
        base_people = len(stub_ids_all)
        if universe_excluded_n > 0:
            excluded_groups = sorted(excluded_parts, key=lambda x: -x[1])
            universe_note = _universe_note(excluded_groups, base_people)
    else:
        base_people = total_n
    row_base_weighted = stub_long_all.groupby("category")["weight"].sum()
    row_base_n = stub_long_all.groupby("category")["resp_id"].nunique()

    coverage_warning = (
        _check_coverage(stub_long, get_label(meta, stub_key))
        or _check_base_coverage(len(both_elig), base_people, universe_applied=key_groups is not None)
    )

    return BannerBlock(
        banner_key=banner_key,
        banner_label=get_label(meta, banner_key),
        pct=pct,
        cell_n=cell_n_table.reindex(index=pct.index, fill_value=0),
        cell_weighted=cell_table.fillna(0.0),
        base_n=base_n.reindex(pct.columns).fillna(0).astype(int),
        small_n_flag=(base_n.reindex(pct.columns).fillna(0) < small_n_threshold),
        coverage_warning=coverage_warning,
        row_base_weighted=row_base_weighted,
        row_base_n=row_base_n,
        universe_applied=key_groups is not None,
        universe_excluded_n=universe_excluded_n,
        universe_note=universe_note,
        universe_excluded_groups=excluded_groups,
        base_people=base_people,
    )


def build_banner(
    data: pd.DataFrame,
    meta: dict[str, VariableMeta],
    stub_key: str,
    banner_keys: list[str],
    na_handling: str = "keep",
    small_n_threshold: int = 30,
    use_weighting: bool = False,
    use_universe: bool = True,
    universe_keys: tuple[str, ...] | None = None,
) -> list[BannerBlock]:
    """
    Ponto de entrada principal do motor -- é a única função deste módulo
    que `app.py` chama diretamente pra gerar um banner.

    Um `BannerBlock` por variável de banner selecionada -- é assim que
    uma tabela banner de verdade é composta: cada corte (Região, Gênero,
    Cliente x Não Cliente...) é cruzado independentemente contra o mesmo
    stub, não combinado entre si.

    O primeiro bloco da lista devolvida é sempre "Total" -- toda a base
    elegível para o stub, sem nenhum corte de banner -- porque é a
    referência que todo banner real tem antes das colunas de corte, e é
    o que `format_banner_table_full`/`format_table_for_export` usam como
    denominador do %LINHA. `banner_keys=[]` devolve só esse bloco Total
    (usado por `app.py` pra montar a coluna "Total geral (sem filtro)"
    quando um filtro de base está ativo).

    `use_weighting`: default False -- a leitura padrão é NÃO ponderada
    (contagem bruta), confirmado como padrão de referência (Ro, 15/09):
    aplicar o peso amostral nacional dentro de um recorte já filtrado
    infla/distorce a proporção, porque o peso foi calibrado pro
    universo total, não pro subgrupo filtrado -- foi assim que a
    divergência entre a tabela de referência (não ponderada, ex.:
    AMAZONAS/PROPRIA=72,9%) e o app (ponderado, 95,0%) apareceu, com o
    mesmo NA em ambos. `True` fica disponível como opção explícita do
    analista, não como padrão.

    `use_universe`: default True -- a base do %LINHA de cada bloco é o
    "universo da pergunta de banner" (grupos onde ela tem resposta), não o
    Total inteiro do stub. É o que impede uma pergunta que não existe pra
    um grupo (ex.: IACOM em PPPs) de entrar no denominador como se fosse
    não-resposta. `False` restaura o Total inteiro (comportamento antigo).
    Ver `infer_question_universe` pra regra e limites. `universe_keys`
    sobrescreve `UNIVERSE_KEYS_DEFAULT`.
    """
    weights = get_weights(data, meta, use_weighting=use_weighting)
    key_groups = build_universe_key_groups(data, universe_keys) if use_universe else None

    total_meta_key = "__TOTAL__"
    blocks = []

    stub_elig_all = eligible_respondents(data, meta, stub_key, na_handling)
    stub_long_all = to_long(data, meta, stub_key, weights, na_handling)
    stub_long_all = stub_long_all[stub_long_all.resp_id.isin(stub_elig_all)]
    total_base_pairs = stub_long_all.drop_duplicates(["resp_id"])
    total_weighted = total_base_pairs["weight"].sum()
    total_cell = stub_long_all.groupby("category")["weight"].sum()
    # Ordem de leitura das categorias do stub (linhas da tabela) -- mesma
    # lógica do banner acima, aplicada aqui porque é total_cell.index que
    # vira stub_categories em format_banner_table_full.
    total_cell = total_cell.reindex(sort_categories(list(total_cell.index)))
    # NUNCA usar total_base_pairs (deduplicado por resp_id) pra contar por
    # categoria -- quando o stub é MR, uma pessoa com 2 seleções precisa
    # contar nas 2 categorias. total_base_pairs serve só pra "quantas
    # pessoas distintas no total" (total_weighted, base_n, small_n_flag
    # abaixo), não pra "quantas por categoria". Bug real que existiu
    # aqui: usar total_base_pairs pra isso fazia pessoas com múltiplas
    # seleções perderem a segunda categoria -- com P2.1a, "Encanada"
    # mostrava 60 em vez de 69.
    total_cell_n = stub_long_all.groupby("category")["resp_id"].nunique()
    total_pct = (total_cell / total_weighted * 100).to_frame("Total")
    total_block = BannerBlock(
        banner_key=total_meta_key,
        banner_label="Total",
        pct=total_pct,
        cell_n=total_cell_n.reindex(total_pct.index).fillna(0).astype(int).to_frame("Total"),
        cell_weighted=total_cell.reindex(total_pct.index).fillna(0.0).to_frame("Total"),
        base_n=pd.Series({"Total": total_base_pairs["resp_id"].nunique()}),
        small_n_flag=pd.Series({"Total": total_base_pairs["resp_id"].nunique() < small_n_threshold}),
        row_base_weighted=total_cell.reindex(total_pct.index).fillna(0.0),
        row_base_n=total_cell_n.reindex(total_pct.index).fillna(0).astype(int),
        base_people=int(total_base_pairs["resp_id"].nunique()),
    )
    blocks.append(total_block)

    for bk in banner_keys:
        block = _build_single_block(
            data, meta, stub_key, bk, weights, na_handling, small_n_threshold,
            total_n=total_base_pairs["resp_id"].nunique(),
            key_groups=key_groups,
        )
        if block.pct.empty:
            # Acontece quando a variável de banner não tem nenhum
            # respondente elegível nesse recorte (ex.: pergunta de um
            # branch de rota que ninguém caiu nesse estudo/onda). Não é
            # erro -- é sinal de que essa variável não se aplica a esse
            # recorte de dados, então simplesmente não entra na lista.
            continue
        blocks.append(block)

    return blocks


# ══════════════════════════════════════════════════════════════════════
#  FORMATAÇÃO DE SAÍDA
# ══════════════════════════════════════════════════════════════════════

def linha_pct_frame(block: BannerBlock, total_block: BannerBlock) -> pd.DataFrame:
    """
    %LINHA de UM bloco -- ÚNICA fonte desse número pra tabela, gráfico e
    exportação (antes cada um refazia `cell_weighted / Total`, três cópias
    da mesma conta que podiam divergir). O denominador é a base do próprio
    bloco (`row_base_weighted`: universo da pergunta quando ligado, Total
    do stub quando não); blocos sem esse campo caem no Total.

    Células sem base viram NaN (não 0): quem chama decide como mostrar.
    """
    denom = block.row_base_weighted if block.row_base_weighted is not None else total_block.cell_weighted["Total"]
    frame = block.cell_weighted.divide(denom, axis=0) * 100
    return frame.replace([np.inf, -np.inf], np.nan)


def _base_sample_denominator(block: BannerBlock, total_block: BannerBlock) -> float:
    """Denominador (N não ponderado, somado nas categorias do stub, igual ao numerador `base_na`) da linha "Base Amostra"."""
    base = block.row_base_n if block.row_base_n is not None else total_block.cell_n["Total"]
    return float(base.sum())


def describe_block_base(block: BannerBlock) -> str:
    """
    Frase única sobre a base do %LINHA de um bloco de banner -- usada na
    tela (legenda acima da tabela) e nas notas das exportações, pra que
    quem abre o CSV/Excel depois, sem a tela, saiba contra quem o %LINHA
    foi calculado. Lista os grupos que ficaram de fora quando há (o
    detalhe completo está em `block.universe_excluded_groups`).
    """
    if not block.universe_applied:
        return f"Base do %LINHA: Total inteiro ({_format_br_number(block.base_people, 0)} pessoas)."
    if block.universe_note:
        return block.universe_note
    return (
        f"Base do %LINHA: {_format_br_number(block.base_people, 0)} pessoas "
        f"(a pergunta existe em todos os grupos; ninguém foi tirado da base)."
    )


def format_banner_table_full(blocks: list[BannerBlock]) -> pd.DataFrame:
    """
    Monta a tabela completa pra exibição -- NA (contagem não ponderada),
    %LINHA e %COLUNA por célula, o padrão clássico de banner do
    SPSS/Quantum, com três linhas por categoria do stub (mais três pra
    "Base Amostra" no final) em vez de uma linha só.

    %COLUNA é o que `_build_single_block` já calcula (`pct`): dentro da
    categoria do banner, qual a fatia de cada categoria do stub. %LINHA é
    o inverso: cada célula dividida pelo total ponderado da PRÓPRIA
    categoria do stub (vem do bloco "Total", que é sempre `blocks[0]` por
    construção de `build_banner`) -- não pela base da coluna, que é o
    denominador do %COLUNA.
    """
    total_block = blocks[0]
    stub_categories = list(total_block.pct.index)

    n_frames, linha_frames, coluna_frames = [], [], []
    for b in blocks:
        cols = pd.MultiIndex.from_product([[b.banner_label], b.pct.columns])

        n_frame = b.cell_n.copy()
        n_frame.columns = cols
        n_frames.append(n_frame)

        coluna_frame = b.pct.copy()
        coluna_frame.columns = cols
        coluna_frames.append(coluna_frame)

        linha_frame = linha_pct_frame(b, total_block).fillna(0.0)
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
    # Cada coluna é dividida pela base DO SEU bloco (mesma base do %LINHA
    # acima): com o universo ligado, "quanto da base respondeu" fecha em
    # 100% quando todo mundo do universo respondeu, em vez de mostrar a
    # parcela que nem foi perguntada como se fosse falta de resposta.
    denom = pd.Series(
        [_base_sample_denominator(b, total_block) for b in blocks for _ in b.pct.columns],
        index=base_na.index,
    )
    rows[("Base Amostra", "NA")] = base_na
    rows[("Base Amostra", "%LINHA")] = (base_na / denom.where(denom > 0) * 100).fillna(0.0).round(1)
    rows[("Base Amostra", "%COLUNA")] = pd.Series(100.0, index=base_na.index).round(1)

    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index)
    return table


def _format_br_number(value: float, decimals: int) -> str:
    """
    Formata número no padrão pt-BR (milhar com ".", decimal com ",").

    CSV é texto puro -- sem metadado de formato, então o separador tem
    que ser fabricado na mão aqui (diferente do .xlsx, onde o Excel lê o
    código de formato e escolhe o separador certo sozinho, ver
    `table_to_excel_bytes`). Faz a troca em duas etapas com um caractere
    de rascunho (`\\x00`, nunca aparece em número formatado) porque trocar
    "," por "." direto sobrescreveria o "." que "," vira em seguida.
    """
    text = f"{value:,.{decimals}f}"  # ex.: "1,234.5" (padrão EUA de saída do Python)
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def format_table_for_export(table: pd.DataFrame) -> pd.DataFrame:
    """
    Versão em texto de `format_banner_table_full`, pra exportar em CSV.

    A tabela original guarda tudo em float64 -- não é bug de conta, é
    limitação do pandas: linha NA (inteiro) e linhas %LINHA/%COLUNA
    (decimal) compartilham as mesmas colunas, e uma coluna só pode ter um
    tipo só, então tudo vira float. Na tela isso fica escondido porque o
    Streamlit formata na exibição (100.0 aparece como "100"); no CSV
    exportado não tem essa máscara, o valor cru aparece com ".0" -- foi
    assim que esse bug foi percebido pela primeira vez. Aqui a formatação
    (inteiro pra NA, 1 casa decimal pro resto) vira texto de verdade
    antes de gerar o CSV, pra o arquivo exportado bater com o que a tela
    mostra.

    Números no padrão pt-BR (milhar "." / decimal ",") -- o público daqui
    lê número em pt-BR, e usar vírgula-de-milhar estilo EUA
    (`f"{v:,.0f}"` puro) confundia quem abria o CSV rápido, achando que
    "1,234" era "1,234" decimal em vez de mil e duzentos e trinta e
    quatro.
    """
    out = table.copy().astype(object)
    na_rows = table.index.get_level_values(1) == "NA"
    out.loc[na_rows] = table.loc[na_rows].map(lambda v: _format_br_number(v, 0))
    out.loc[~na_rows] = table.loc[~na_rows].map(lambda v: _format_br_number(v, 1))
    return out


def table_to_excel_bytes(
    df: pd.DataFrame,
    *,
    sheet_name: str = "Dados",
    default_format: str = "#,##0.00",
    column_formats: dict[str, str] | None = None,
) -> bytes:
    """
    Serializa um DataFrame "achatado" (índice e colunas de 1 nível só,
    ex.: a tabela de "Visão geral" ou "Índice individual" de indices.py)
    pra bytes de .xlsx, com número de verdade em cada célula.

    Ao contrário do CSV (`format_table_for_export`), aqui NÃO se fabrica
    texto com separador pt-BR na mão: o código de formato do Excel
    ("#,##0.00") é sintaxe universal -- são placeholders de posição de
    milhar/decimal, não caracteres literais -- e o Excel escolhe "," ou
    "." de verdade a partir da configuração regional de quem abre o
    arquivo. Escrever texto pré-formatado seria pior aqui: a pessoa
    perderia a possibilidade de somar, filtrar ou fazer gráfico em cima
    da coluna dentro do próprio Excel.

    `column_formats`: código de formato por nome de coluna, pra tabelas
    com colunas de precisão diferente (ex.: "Média" com 2 casas, "N"
    como inteiro) -- coluna fora do dict cai em `default_format`.
    """
    import io

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    column_formats = column_formats or {}
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]  # limite de 31 caracteres do Excel pro nome da aba

    bold = Font(bold=True)
    ws.cell(row=1, column=1, value=df.index.name or "")
    for j, col_name in enumerate(df.columns):
        cell = ws.cell(row=1, column=j + 2, value=str(col_name))
        cell.font = bold
        cell.alignment = Alignment(horizontal="center")

    for i, (idx, row) in enumerate(df.iterrows()):
        ws.cell(row=i + 2, column=1, value=str(idx)).font = bold
        for j, col_name in enumerate(df.columns):
            val = row[col_name]
            if pd.notna(val):
                cell = ws.cell(row=i + 2, column=j + 2, value=float(val))
                cell.number_format = column_formats.get(col_name, default_format)

    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = 42
    for j in range(len(df.columns)):
        ws.column_dimensions[get_column_letter(j + 2)].width = 16

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def banner_table_to_excel_bytes(
    table: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> bytes:
    """
    Serializa o banner completo (índice de 2 níveis -- categoria do stub
    + métrica NA/%LINHA/%COLUNA; coluna de 2 níveis -- bloco de banner +
    categoria) pra bytes de .xlsx, reconstruindo o cabeçalho mesclado que
    o Streamlit mostra na tela.

    Número de verdade em cada célula (ver `table_to_excel_bytes` pro
    porquê disso importar mais aqui do que no CSV).

    `mask`: mesmo formato de `small_n_mask_full` -- pinta de amarelo a
    célula cuja base está abaixo do limiar, espelhando o destaque que já
    existe em `_highlight_small_n` na tela. Opcional: sem `mask`, exporta
    sem cor nenhuma.

    `notes`: linhas de texto gravadas abaixo da tabela (ex.: a base do
    %LINHA de cada banner, de `describe_block_base`) -- o arquivo exportado
    circula sem a tela, então a base precisa ir junto com os números.
    """
    import io

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Banner"

    n_index_cols = 2
    header_row1, header_row2, data_start = 1, 2, 3

    bold = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="DDDDDD")
    small_n_fill = PatternFill("solid", fgColor="FFF3CD")

    ws.cell(row=header_row2, column=1, value="Categoria").font = bold
    ws.cell(row=header_row2, column=2, value="Métrica").font = bold

    # linha 1 mescla o rótulo do bloco de banner (ex.: "Total", "ANO");
    # linha 2 tem a categoria dentro do bloco (ex.: "2023", "2024")
    current_label, span_start = None, n_index_cols + 1
    for i, (label, cat) in enumerate(table.columns):
        c = n_index_cols + 1 + i
        cell = ws.cell(row=header_row2, column=c, value=str(cat))
        cell.font = bold
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        if label != current_label:
            if current_label is not None:
                ws.merge_cells(start_row=header_row1, start_column=span_start, end_row=header_row1, end_column=c - 1)
            ws.cell(row=header_row1, column=span_start, value=str(label)).font = bold
            ws.cell(row=header_row1, column=span_start).alignment = Alignment(horizontal="center")
            current_label, span_start = label, c
    last_col = n_index_cols + len(table.columns)
    ws.merge_cells(start_row=header_row1, start_column=span_start, end_row=header_row1, end_column=last_col)

    int_fmt, dec_fmt = "#,##0", "#,##0.0"

    row_start_by_cat: dict = {}
    for r, ((cat, metric), row_values) in enumerate(table.iterrows()):
        excel_row = data_start + r
        row_start_by_cat.setdefault(cat, excel_row)
        ws.cell(row=excel_row, column=2, value=metric)
        fmt = int_fmt if metric == "NA" else dec_fmt
        for c, (col_key, val) in enumerate(zip(table.columns, row_values)):
            cell = ws.cell(row=excel_row, column=n_index_cols + 1 + c, value=float(val))
            cell.number_format = fmt
            if mask is not None and bool(mask.loc[(cat, metric), col_key]):
                cell.fill = small_n_fill

    # mescla a coluna "Categoria" nas 3 linhas (NA/%LINHA/%COLUNA) de
    # cada categoria do stub, igual ao efeito visual da tela
    for cat, first_row in row_start_by_cat.items():
        ws.cell(row=first_row, column=1, value=str(cat)).font = bold
        ws.merge_cells(start_row=first_row, start_column=1, end_row=first_row + 2, end_column=1)

    if notes:
        note_row = data_start + len(table) + 1
        ws.cell(row=note_row, column=1, value="Notas").font = bold
        for i, text in enumerate(notes, start=1):
            ws.cell(row=note_row + i, column=1, value=text)

    ws.freeze_panes = ws.cell(row=data_start, column=n_index_cols + 1).coordinate
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 10
    for c in range(n_index_cols + 1, last_col + 1):
        ws.column_dimensions[get_column_letter(c)].width = 14

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def small_n_mask_full(blocks: list[BannerBlock]) -> pd.DataFrame:
    """
    Máscara booleana do mesmo formato (linhas/colunas) de
    `format_banner_table_full`, pra `app.py` colorir de amarelo toda
    célula cuja base está abaixo do limiar.

    A marcação de N pequeno vale pra COLUNA inteira (é a base daquela
    coluna que é pequena, não uma métrica específica dela) -- por isso o
    mesmo `small_n_flag` de cada bloco se repete nas 3 linhas de cada
    categoria do stub, e também nas 3 linhas de "Base Amostra".
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
