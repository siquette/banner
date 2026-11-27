"""
diagnostico.py — Procura de problemas no banco (aba "🔎 Dados: filtro e diagnóstico").

PAPEL NO PROJETO
-----------------
Cada função aqui recebe o banco e devolve uma TABELA DE ACHADOS: o que
parece errado, quantas linhas, em que anos. O app só mostra. Nada aqui
conserta dado -- decisão do Ro (26/11): a aba acha e documenta, o
conserto vai pro ETL (`aplicar_mapa.py`, `CORRECOES_DE_VALOR`), com
auditoria. Se o app consertasse, o mesmo banco daria um número no app e
outro no ETL.

Sem Streamlit aqui, pelo mesmo motivo de `crosstab_engine.py` e
`stats_engine.py`: testável sozinho (`tests/test_diagnostico.py`).

CADA VERIFICAÇÃO SAIU DE UM PROBLEMA QUE JÁ APARECEU NESTE BANCO
------------------------------------------------------------------
- `preenchimento_por_ano`: a P23_media veio vazia em 2026 com a P23 cheia;
  a P1.x_media vazia em 2023. Ano vazio no MEIO da série (ex.: 2022 e 2025
  cheios, 2023 e 2024 vazios) é sinal de pergunta perdida na junção.
- `texto_vs_media`: na P23 de 2022-2023, ~1.700 entrevistas com
  "SATISFEITO" e média 2.
- `colunas_identicas` / `colunas_complementares` / `codigos_duplicados`:
  pergunta partida em duas colunas entre anos (P37.1.1 x P48), colunas com
  o mesmo conteúdo (N26_013 x N26_091), código "__dup1" na P2.2a.
- `duplicadas_na_chave`: a pessoa é ID + ANO + CONCESSIONARIA (e ID_UNICO).
- Regras de consistência: ficam em `regras_diagnostico.py` (SQL), rodadas
  pelo `sql_filter`.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

LIMITE_ESCALA = 11  # até 11 valores distintos, a média é de escala (1-5, 0-10): dá pra comparar texto x número


def _anos(data: pd.DataFrame, coluna_ano: str) -> pd.Series:
    return data[coluna_ano].astype("string").fillna("(sem ano)")


# ── 1. preenchimento por ano ──────────────────────────────────────────

def preenchimento_por_ano(data: pd.DataFrame, coluna_ano: str = "ANO") -> pd.DataFrame:
    """
    % de linhas com resposta, por coluna (linhas) e ano (colunas), mais:
    - `anos_vazios_no_meio`: anos com 0% entre o primeiro e o último ano
      com dado (o sinal mais forte de problema);
    - `anos_vazios`: todos os anos com 0%;
    - `sinal`: texto curto pra ordenar e filtrar ('' = nada a ver).
    Pergunta que só existe em parte dos anos (ex.: nova em 2026) NÃO é
    sinal sozinha; só o buraco no meio é.
    """
    anos = _anos(data, coluna_ano)
    pct = data.notna().groupby(anos.to_numpy()).mean().T * 100
    pct = pct[sorted(pct.columns)]
    ordem = list(pct.columns)

    def _buraco(linha: pd.Series) -> list[str]:
        com = [i for i, a in enumerate(ordem) if linha[a] > 0]
        if not com:
            return []
        return [ordem[i] for i in range(min(com) + 1, max(com)) if linha[ordem[i]] == 0]

    pct["anos_vazios_no_meio"] = pct[ordem].apply(_buraco, axis=1).map(lambda l: ", ".join(l))
    pct["anos_vazios"] = pct[ordem].apply(lambda r: ", ".join(a for a in ordem if r[a] == 0), axis=1)
    pct["sinal"] = np.where(pct["anos_vazios_no_meio"] != "", "ano vazio no meio da série", "")
    vazia = pct[ordem].max(axis=1) == 0
    pct.loc[vazia, "sinal"] = "coluna vazia em todos os anos"
    return pct


# ── 2. texto x média ──────────────────────────────────────────────────

def pares_texto_media(data: pd.DataFrame) -> list[tuple[str, str]]:
    """(coluna de texto, coluna `_media`) com o mesmo código -- ex.: ('P23', 'P23_media')."""
    cols = set(data.columns)
    return [(c[: -len("_media")], c) for c in data.columns if c.endswith("_media") and c[: -len("_media")] in cols]


def _mapa_texto_numero(texto: pd.Series, numero: pd.Series) -> pd.Series:
    """Nota de cada resposta escrita = a nota mais comum que ela tem no banco (aprendida das linhas com os dois)."""
    ambos = texto.notna() & numero.notna()
    df = pd.DataFrame({"t": texto[ambos].astype("object"), "m": numero[ambos]})
    return df.groupby("t")["m"].agg(lambda s: s.mode().iloc[0])


def texto_vs_media(data: pd.DataFrame, coluna_ano: str = "ANO") -> pd.DataFrame:
    """
    Uma linha por par (texto, `_media`) com problema:
    - `divergem`: a resposta escrita tem uma nota diferente da que ela
      costuma ter (ex.: "SATISFEITO" com nota 2, quando quase sempre é 4);
    - `texto_sem_media`: tem resposta escrita que normalmente tem nota, mas
      a `_media` está vazia (ex.: P23 em 2026);
    - `media_sem_texto`: tem nota mas não tem resposta escrita;
    - `anos`: em que anos estão essas linhas.

    A nota "certa" de cada resposta é aprendida do próprio banco (a mais
    comum). Só vale pra média de ESCALA (até 11 valores): em faixa (idade x
    faixa etária, índice x "média próxima de 3") uma resposta escrita cobre
    muitos números, e a comparação não faz sentido -- esses pares saem.
    Respostas que nunca têm nota (ex.: "Não sabe") não contam como
    "texto sem média".
    """
    anos = _anos(data, coluna_ano)
    linhas = []
    for texto_col, media_col in pares_texto_media(data):
        numero = pd.to_numeric(data[media_col], errors="coerce")
        if numero.nunique() > LIMITE_ESCALA or not (data[texto_col].notna() & numero.notna()).any():
            continue
        texto = data[texto_col].astype("object")
        esperado = texto.map(_mapa_texto_numero(texto, numero))
        divergem = texto.notna() & numero.notna() & (numero != esperado)
        sem_media = texto.notna() & numero.isna() & esperado.notna()
        sem_texto = texto.isna() & numero.notna()
        qualquer = divergem | sem_media | sem_texto
        if not qualquer.any():
            continue
        linhas.append({
            "texto": texto_col, "media": media_col,
            "divergem": int(divergem.sum()), "texto_sem_media": int(sem_media.sum()),
            "media_sem_texto": int(sem_texto.sum()),
            "anos": ", ".join(f"{a}: {n}" for a, n in anos[qualquer].value_counts().sort_index().items()),
        })
    out = pd.DataFrame(linhas, columns=["texto", "media", "divergem", "texto_sem_media", "media_sem_texto", "anos"])
    return out.sort_values(["divergem", "texto_sem_media", "media_sem_texto"], ascending=False, ignore_index=True)


def linhas_texto_vs_media(data: pd.DataFrame, texto_col: str, media_col: str) -> pd.Series:
    """Máscara das linhas com problema num par (pro 'Quais são' da aba)."""
    numero = pd.to_numeric(data[media_col], errors="coerce")
    texto = data[texto_col].astype("object")
    esperado = texto.map(_mapa_texto_numero(texto, numero))
    return ((texto.notna() & numero.notna() & (numero != esperado))
            | (texto.notna() & numero.isna() & esperado.notna())
            | (texto.isna() & numero.notna()))


# ── 3. colunas parecidas ──────────────────────────────────────────────

def codigos_duplicados(colunas: list[str]) -> list[str]:
    """Códigos com '__dup' -- o ETL achou duas colunas com o mesmo nome e numerou a segunda."""
    return [c for c in colunas if "__dup" in c]


def colunas_identicas(data: pd.DataFrame) -> list[list[str]]:
    """
    Grupos de colunas com exatamente o mesmo conteúdo, linha a linha.
    Coluna constante (vazia, ou um valor só -- ex.: opção de múltipla
    escolha que ninguém marcou) fica de fora: seria "igual" a todas as
    outras constantes sem dizer nada.

    Pra não comparar 1.600 x 1.600 colunas: agrupa por uma assinatura
    barata (quantas respostas, quantos valores diferentes) e só calcula o
    hash do conteúdo dentro dos grupos com mais de uma coluna.
    """
    por_assinatura: dict[tuple[int, int], list[str]] = defaultdict(list)
    for c in data.columns:
        s = data[c]
        n, k = int(s.notna().sum()), int(s.nunique())
        if n == 0 or k <= 1:
            continue
        por_assinatura[(n, k)].append(c)
    grupos: list[list[str]] = []
    for cols in por_assinatura.values():
        if len(cols) < 2:
            continue
        # impressão digital do PADRÃO linha a linha (factorize: 1º valor que aparece = 0, 2º = 1...),
        # rápida pra qualquer tipo; colunas iguais têm o mesmo padrão. Padrão igual com valores
        # diferentes (ex.: Sim/Não x True/False) é descartado na conferência abaixo.
        por_hash: dict[bytes, list[str]] = defaultdict(list)
        for c in cols:
            codigos = pd.factorize(data[c], use_na_sentinel=True)[0].astype(np.int32)
            por_hash[hashlib.blake2b(codigos.tobytes(), digest_size=16).digest()].append(c)
        for mesmos in por_hash.values():
            restantes = list(mesmos)
            while len(restantes) > 1:  # confirma pelo conteúdo, valor a valor (como texto, pra bool e categoria compararem)
                base = data[restantes[0]].astype("string")
                iguais = [restantes[0]] + [c for c in restantes[1:] if base.equals(data[c].astype("string"))]
                if len(iguais) > 1:
                    grupos.append(iguais)
                restantes = [c for c in restantes if c not in iguais]
    return grupos


def _palavras(rotulo: str) -> frozenset[str]:
    """Palavras do texto da pergunta, sem o código do começo, sem acento, sem '(média)'."""
    r = re.sub(r"^\s*[A-Za-z0-9_.\-]+\s*[.)-]?\s*", "", str(rotulo))
    r = re.sub(r"\(m[ée]dia\)|\(R[UM][^)]*\)|_c\s*$", "", r, flags=re.IGNORECASE)
    r = "".join(ch for ch in unicodedata.normalize("NFKD", r) if not unicodedata.combining(ch)).lower()
    return frozenset(w for w in re.sub(r"[^a-z0-9 ]", " ", r).split() if len(w) > 2)


def colunas_complementares(data: pd.DataFrame, rotulos: dict[str, str], coluna_ano: str = "ANO",
                           semelhanca_min: float = 0.7) -> pd.DataFrame:
    """
    Pares de colunas que parecem a MESMA pergunta partida entre anos: o
    texto da pergunta é parecido (≥ 70% das palavras em comum) e uma só
    tem dado nos anos em que a outra não tem. Ex.: P37.1.1 (2025-2026) x
    P48 (2023-2024). Só compara colunas que não têm todos os anos.
    """
    anos = _anos(data, coluna_ano)
    todos = set(anos.unique())
    anos_de = {c: frozenset(anos[data[c].notna()].unique()) for c in data.columns}
    palavras = {c: _palavras(rotulos.get(c, c)) for c in data.columns}
    candidatas = [c for c in data.columns if anos_de[c] and anos_de[c] != todos and palavras[c]]
    indice: dict[str, set[str]] = defaultdict(set)
    for c in candidatas:
        for w in palavras[c]:
            indice[w].add(c)
    pares = set()
    for a in candidatas:
        vizinhas: Counter = Counter()
        for w in palavras[a]:
            if len(indice[w]) < 300:  # palavra comum demais ("voce", "empresa") não ajuda a achar par
                vizinhas.update(indice[w])
        for b in vizinhas:
            if b <= a or anos_de[a] & anos_de[b]:
                continue
            sem = len(palavras[a] & palavras[b]) / len(palavras[a] | palavras[b])
            if sem >= semelhanca_min:
                pares.add((a, b, round(sem, 2)))
    out = pd.DataFrame(
        [{"coluna_a": a, "anos_a": ", ".join(sorted(anos_de[a])), "coluna_b": b, "anos_b": ", ".join(sorted(anos_de[b])),
          "semelhanca_texto": s, "pergunta_a": rotulos.get(a, a), "pergunta_b": rotulos.get(b, b)} for a, b, s in pares],
        columns=["coluna_a", "anos_a", "coluna_b", "anos_b", "semelhanca_texto", "pergunta_a", "pergunta_b"],
    )
    return out.sort_values("semelhanca_texto", ascending=False, ignore_index=True)


# ── 4. duplicadas na chave ────────────────────────────────────────────

CHAVES = [["ID_UNICO"], ["ID", "ANO", "CONCESSIONARIA"]]


def duplicadas_na_chave(data: pd.DataFrame) -> pd.DataFrame:
    """Pra cada chave de pessoa que existe no banco: quantas linhas repetem a chave (todas as cópias contam)."""
    linhas = []
    for chave in CHAVES:
        if not all(c in data.columns for c in chave):
            continue
        rep = data.duplicated(subset=chave, keep=False)
        linhas.append({"chave": " + ".join(chave), "linhas_repetidas": int(rep.sum()), "de": len(data)})
    return pd.DataFrame(linhas, columns=["chave", "linhas_repetidas", "de"])


def mascara_duplicadas(data: pd.DataFrame, chave: list[str]) -> pd.Series:
    return data.duplicated(subset=chave, keep=False)


# ── onde está ─────────────────────────────────────────────────────────

def onde_esta(data: pd.DataFrame, mascara, por: list[str], top: int = 15) -> dict[str, pd.Series]:
    """Contagem das linhas marcadas por cada variável de `por` (as `top` maiores) -- diz se o problema é de um ano, uma cidade..."""
    m = np.asarray(mascara, dtype=bool)
    out = {}
    for c in por:
        if c in data.columns:
            out[c] = data.loc[m, c].astype("string").fillna("(vazio)").value_counts().head(top)
    return out
