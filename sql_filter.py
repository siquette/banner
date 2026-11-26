"""
sql_filter.py — Filtro de base escrito em SQL (aba "Filtro SQL" do app).

PAPEL NO PROJETO
-----------------
Recebe o banco (DataFrame) e uma CONDIÇÃO SQL -- só o que viria depois do
WHERE -- e devolve uma máscara booleana: uma posição por entrevista,
True = passou. `app.py` usa essa máscara como o primeiro corte da base,
antes dos filtros em cascata da barra lateral. Nada aqui conhece o
Streamlit: dá pra testar o módulo inteiro sem abrir o app, pelo mesmo
motivo de `crosstab_engine.py` ser separado.

POR QUE SQL, E POR QUE DUCKDB
------------------------------
O filtro da barra lateral só faz "manter estes valores nesta variável",
com AND entre variáveis. Não faz OR, número (PESO > 1,5), múltipla
escolha, nem condição que depende de outras linhas ("cidades presentes
nos 5 anos"). SQL faz tudo isso numa linha. O DuckDB roda SQL direto em
cima do DataFrame que já está na memória, sem copiar o banco: ele só lê
as colunas que a condição cita.

POR QUE SÓ A CONDIÇÃO
----------------------
A caixa aceita só a expressão do WHERE; a consulta em volta é montada
aqui. Assim o resultado é sempre "um subconjunto das entrevistas do
banco" -- nunca uma tabela nova, nunca uma linha inventada. A checagem
de parênteses existe porque, sem ela, `1=1) UNION SELECT ... WHERE (1=1`
escapa do WHERE e cola linhas falsas no resultado (testado).

POR QUE ROLA SOBRE O BANCO INTEIRO
-----------------------------------
A condição enxerga o banco inteiro, não a base já filtrada pela barra
lateral. Uma subconsulta como "cidades presentes em todos os anos"
precisa ver todos os anos; se rodasse depois de ANO = 2025 na barra
lateral, a resposta sairia errada sem aviso.

PROTEÇÕES
----------
- Uma condição só: sem `;` e sem comentário.
- Sem acesso a arquivo, internet, extensão ou mudança de configuração
  (`enable_external_access = false`, configuração travada no fim).
- Sem acesso a variável do Python pelo nome (`python_enable_replacements
  = false`): a única tabela visível é `dados`.
- Limite de memória e de tempo (`LIMITE_SEGUNDOS`); passou disso, a
  consulta é interrompida.
- Colunas retiradas do app (`VarType.EXCLUDED`, ex.: IMCE/IMCT) não vão
  pro DuckDB: a tabela `dados` é o banco sem elas.

POSIÇÃO DA LINHA, NÃO ORDEM DA RESPOSTA
-----------------------------------------
O banco vai pro DuckDB com uma coluna a mais, `__linha__` (0, 1, 2...
= posição no DataFrame), e a consulta devolve as posições que passaram:
`SELECT __linha__ FROM dados WHERE (condição)`. A máscara é montada a
partir dessas posições. Não dá pra confiar na ORDEM das linhas que o
DuckDB devolve: com subconsulta (IN, junção) ele embaralha a saída --
testado: a contagem batia, mas as linhas eram outras. Com a posição
explícita, ordem não importa, e não depende de existir um ID único no
banco. E como é um WHERE de verdade, valem as regras de WHERE (COUNT/SUM
fora de subconsulta dá erro claro).
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

TABELA = "dados"
_COLUNA_LINHA = "__linha__"
LIMITE_SEGUNDOS = 10
LIMITE_MEMORIA = "1GB"


class ErroFiltroSQL(ValueError):
    """
    Erro com a mensagem já em português, pronta pra mostrar na tela.
    `detalhe` guarda a mensagem original do DuckDB (em inglês), pra quem
    sabe SQL conferir quando a tradução não bastar.
    """

    def __init__(self, mensagem: str, detalhe: str | None = None):
        super().__init__(mensagem)
        self.detalhe = detalhe


# ── checagem do texto, antes de chegar no DuckDB ──────────────────────

def limpar_condicao(texto: str | None) -> str | None:
    """Tira espaço das pontas. Vazio (ou só espaço) = sem filtro = None."""
    if texto is None:
        return None
    texto = texto.strip()
    return texto or None


def _fim_das_aspas(texto: str, inicio: int, aspa: str) -> int | None:
    """Posição da aspa que fecha a que abre em `inicio`; aspa dobrada ('' ou "") é aspa escrita, não fecha."""
    i = inicio + 1
    while i < len(texto):
        if texto[i] == aspa:
            if i + 1 < len(texto) and texto[i + 1] == aspa:
                i += 2
                continue
            return i
        i += 1
    return None


def _trecho(texto: str, pos: int) -> str:
    """Pedaço do texto em volta de `pos`, pra mensagem de erro dizer onde."""
    ini = max(0, pos - 20)
    return ("…" if ini > 0 else "") + texto[ini:pos + 1]


def validar_condicao(texto: str) -> str:
    """
    Confere o que dá pra conferir sem o DuckDB e devolve a condição limpa.
    Levanta `ErroFiltroSQL` com a mensagem em português.

    Lê o texto uma vez, pulando o que está entre aspas (dentro de
    'texto' ou "coluna", `;` e parêntese não contam).
    """
    cond = limpar_condicao(texto)
    if cond is None:
        raise ErroFiltroSQL("A caixa está vazia: escreva uma condição ou use Limpar filtro.")
    profundidade = 0
    i = 0
    while i < len(cond):
        ch = cond[i]
        if ch in ("'", '"'):
            fim = _fim_das_aspas(cond, i, ch)
            if fim is None:
                tipo = "simples (')" if ch == "'" else 'duplas (")'
                raise ErroFiltroSQL(f"Tem aspas {tipo} abertas sem fechar, a partir de: {cond[i:i + 30]}")
            i = fim + 1
            continue
        if ch == ";":
            raise ErroFiltroSQL("Só uma condição por vez: tire o ponto e vírgula (;).")
        if cond.startswith("--", i) or cond.startswith("/*", i):
            raise ErroFiltroSQL("Comentário (-- ou /* */) não é aceito na caixa: apague e aplique de novo.")
        if ch == "(":
            profundidade += 1
        elif ch == ")":
            profundidade -= 1
            if profundidade < 0:
                raise ErroFiltroSQL(f"Tem um ')' sem '(' correspondente, perto de: {_trecho(cond, i)}")
        i += 1
    if profundidade > 0:
        raise ErroFiltroSQL(f"Falta fechar {profundidade} parêntese(s).")
    return cond


_IDENT_SOLTO = re.compile(r"[A-Za-z_À-ɏ][\w$]*")


def colunas_citadas(texto: str | None, colunas: list[str]) -> list[str]:
    """
    Colunas do banco que aparecem na condição, na ordem em que aparecem
    -- entre aspas duplas ou soltas (sem diferenciar maiúscula, como o
    DuckDB faz). Texto entre aspas simples é valor, não coluna. Usado pra
    já marcar essas colunas na tabela da aba.
    """
    cond = limpar_condicao(texto)
    if cond is None:
        return []
    por_maiuscula = {c.upper(): c for c in colunas}
    achadas: list[str] = []
    i = 0
    while i < len(cond):
        ch = cond[i]
        if ch in ("'", '"'):
            fim = _fim_das_aspas(cond, i, ch)
            if fim is None:
                break
            if ch == '"':
                nome = cond[i + 1:fim].replace('""', '"')
                c = por_maiuscula.get(nome.upper())
                if c and c not in achadas:
                    achadas.append(c)
            i = fim + 1
            continue
        m = _IDENT_SOLTO.match(cond, i)
        if m and (i == 0 or not (cond[i - 1].isalnum() or cond[i - 1] == "_")):
            c = por_maiuscula.get(m.group(0).upper())
            if c and c not in achadas:
                achadas.append(c)
            i = m.end()
            continue
        i += 1
    return achadas


@lru_cache(maxsize=1)
def _palavras_reservadas() -> frozenset[str]:
    try:
        import duckdb
    except ImportError:  # sem duckdb a aba ainda abre (e avisa no Aplicar); aqui basta o essencial
        return frozenset({"SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL", "ORDER", "GROUP", "AS", "CASE", "END"})
    with duckdb.connect(":memory:") as con:
        rows = con.execute("SELECT keyword_name FROM duckdb_keywords() WHERE keyword_category = 'reserved'").fetchall()
    return frozenset(r[0].upper() for r in rows)


def como_escrever(coluna: str) -> str:
    """
    Como a coluna tem que ser escrita na condição. Nome simples (letra,
    número, _) vai solto; com ponto, hífen, espaço, acento ou palavra
    reservada do SQL, vai entre aspas duplas: "P29.1", "P46.1-Outros".
    """
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", coluna) and coluna.upper() not in _palavras_reservadas():
        return coluna
    return '"' + coluna.replace('"', '""') + '"'


# ── execução ──────────────────────────────────────────────────────────

def calcular_mascara(
    data: pd.DataFrame,
    condicao: str,
    colunas_ocultas: tuple[str, ...] = (),
    limite_segundos: float = LIMITE_SEGUNDOS,
) -> np.ndarray:
    """
    Roda a condição sobre o banco inteiro e devolve a máscara (np.ndarray
    de bool, len == len(data)). Levanta `ErroFiltroSQL` com a mensagem em
    português -- inclusive quando o DuckDB recusa a condição.

    Uma conexão nova por chamada: o DuckDB não deve ser compartilhado
    entre as sessões do Streamlit (cada aba do navegador é uma thread), e
    abrir + registrar o DataFrame custa ~0,2 s. Quem chama guarda o
    resultado em cache.
    """
    import duckdb  # import aqui: o app abre normal mesmo se o duckdb faltar, só a aba avisa

    cond = validar_condicao(condicao)
    ocultas = tuple(c for c in colunas_ocultas if c in data.columns)
    # drop/assign não copiam o banco no pandas 3 (copy-on-write); no 2.x copiam (~0,2 s)
    base = data.drop(columns=list(ocultas)).assign(**{_COLUNA_LINHA: np.arange(len(data), dtype=np.int64)})
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET python_enable_replacements = false")
        con.execute("SET enable_external_access = false")
        con.execute(f"SET memory_limit = '{LIMITE_MEMORIA}'")
        con.execute(f"SET threads = {max(1, min(4, os.cpu_count() or 1))}")
        con.register(TABELA, base)
        con.execute("SET lock_configuration = true")

        cronometro = threading.Timer(limite_segundos, con.interrupt)
        cronometro.start()
        try:
            posicoes = con.execute(f'SELECT "{_COLUNA_LINHA}" FROM {TABELA} WHERE ({cond})').fetchnumpy()[_COLUNA_LINHA]
        except duckdb.Error as exc:
            raise ErroFiltroSQL(traduzir_erro(exc, cond, ocultas, limite_segundos), _primeira_linha(exc)) from exc
        finally:
            cronometro.cancel()
    finally:
        con.close()

    mascara = np.zeros(len(data), dtype=bool)
    mascara[np.asarray(posicoes, dtype=np.int64)] = True
    return mascara


# ── mensagens ─────────────────────────────────────────────────────────

def _primeira_linha(exc: Exception, limite: int = 300) -> str:
    """Primeira linha da mensagem original, cortada: com coluna categórica o DuckDB lista todas as categorias."""
    linha = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return linha if len(linha) <= limite else linha[:limite] + " …"


def traduzir_erro(exc: Exception, cond: str, colunas_ocultas: tuple[str, ...] = (), limite_segundos: float = LIMITE_SEGUNDOS) -> str:
    """
    Mensagem do DuckDB (inglês, técnica) -> o que está errado e como
    arrumar, em português. Casa pelos trechos fixos que o DuckDB 2.x
    escreve; o que não casar cai na última linha, com a mensagem original.
    """
    tipo = type(exc).__name__
    msg = str(exc)
    linha = _primeira_linha(exc)

    if tipo == "InterruptException":
        return (f"A condição passou de {limite_segundos:g} s e foi interrompida. Costuma ser junção da "
                f"tabela com ela mesma; quase sempre dá pra trocar por uma subconsulta com IN.")
    if tipo == "PermissionException" or "disabled by configuration" in msg or "lock_configuration" in msg:
        return "Isso não é permitido aqui: ler arquivo, acessar internet ou mudar configuração. A única tabela é dados."
    if tipo == "OutOfMemoryException" or "Out of Memory" in msg:
        return "A condição precisou de memória demais e foi parada. Tente uma subconsulta mais simples."

    m = re.search(r'Referenced column "(.+?)" not found', msg)
    if m:
        nome = m.group(1)
        if nome in colunas_ocultas:
            return f"A coluna {nome} foi retirada do app de propósito e não pode ser usada no filtro."
        cands = re.findall(r'"(.+?)"', msg.split("Candidate bindings:", 1)[1].splitlines()[0]) if "Candidate bindings:" in msg else []
        sem_acento = sem_acento_minusculo(nome).upper()
        if sem_acento != nome.upper() and sem_acento in (c.upper() for c in cands):
            return f"A coluna {nome} não existe. Você quis dizer {sem_acento}? Os códigos não têm acento."
        texto = f"A coluna {nome} não existe na tabela dados."
        if cands:
            texto += " Parecidas: " + ", ".join(cands[:5]) + "."
        return texto + (f" Se {nome} é um valor (texto), escreva entre aspas simples: '{nome}'. "
                        f"Aspas duplas são só pra nome de coluna.")
    m = re.search(r'Table with name (.+?) does not exist', msg)
    if m:
        return f"A tabela {m.group(1)} não existe. Aqui só tem uma tabela: dados."
    m = re.search(r'(?:Scalar |Aggregate |Table )?Function with name (.+?) does not exist', msg)
    if m:
        sug = re.search(r'Did you mean "(.+?)"', msg)
        return f"A função {m.group(1)} não existe no DuckDB." + (f" Você quis dizer {sug.group(1)}?" if sug else "")
    if "cannot contain aggregates" in msg:
        return ("COUNT, SUM, AVG... não podem ficar soltos na condição, só dentro de uma subconsulta, ex.: "
                "CIDADES IN (SELECT CIDADES FROM dados GROUP BY CIDADES HAVING COUNT(*) > 100).")
    if "window functions" in msg:
        return "Função de janela (OVER) não pode ficar solta na condição, só dentro de uma subconsulta."
    if tipo == "ParserException" or "syntax error" in msg:
        fim = re.search(r"\b(AND|OR|NOT|IN|LIKE|BETWEEN|IS)\s*$|([=<>!]+)\s*$", cond, re.IGNORECASE)
        if fim:
            return f"A condição terminou no meio: sobrou um {(fim.group(1) or fim.group(2)).upper()} no fim."
        perto = re.search(r'syntax error at or near "(.+?)"', msg)
        if perto:
            return f"Erro de escrita perto de {perto.group(1)}. Confira vírgulas, aspas e parênteses."
        return "Erro de escrita na condição. Confira vírgulas, aspas e parênteses."
    m = re.search(r"Could not convert string '(.*?)' to (\w+)", msg)
    if m:
        if m.group(2).upper().startswith("BOOL"):
            return ("A condição precisa dar verdadeiro ou falso: compare a coluna com um valor, ex.: "
                    "CIDADES = 'Rio de Janeiro'.")
        return (f"Tipos não combinam: '{m.group(1)}' é texto e foi comparado com número. Coluna de texto se "
                f"compara com texto entre aspas simples (P25 = 'Melhorou'); PESO e as colunas _media são número.")
    if "Cannot compare values of type" in msg or "No function matches" in msg or "Conversion Error" in msg:
        return ("Tipos não combinam (texto com número, ou verdadeiro/falso com texto?). Texto vai entre aspas "
                "simples; PESO e as colunas _media são número; múltipla escolha é verdadeiro/falso.")
    return f"O DuckDB recusou a condição: {linha}"


def resumo_valores(serie: pd.Series, n: int = 8) -> str:
    """Os valores que uma coluna tem, em uma linha -- pra busca de código e pra mensagem de 'nenhuma entrevista'."""
    if pd.api.types.is_bool_dtype(serie):
        return "verdadeiro / falso (True / False)"
    if pd.api.types.is_numeric_dtype(serie):
        s = serie.dropna()
        return f"número de {s.min():g} a {s.max():g}" if len(s) else "(sem valor)"
    contagem = serie.dropna().astype("string").value_counts()
    if contagem.empty:
        return "(sem valor)"
    # cabem todos: em ordem alfabética (ANO sai 2022, 2023...); não cabem: os mais frequentes
    valores = sorted(str(v) for v in contagem.index) if len(contagem) <= n else [str(v) for v in contagem.index[:n]]
    resto = len(contagem) - len(valores)
    return ", ".join(valores) + (f" … (+{resto})" if resto > 0 else "")


def mensagem_sem_linhas(data: pd.DataFrame, condicao: str) -> str:
    """Mensagem de 'nenhuma entrevista passou', com os valores das colunas de texto que a condição cita."""
    texto = "Nenhuma entrevista passou na condição, então o filtro NÃO foi aplicado. Maiúscula e acento contam."
    for c in colunas_citadas(condicao, list(data.columns))[:3]:
        s = data[c]
        if not (pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s)):
            texto += f" {c} tem: {resumo_valores(s)}."
    return texto


def sem_acento_minusculo(texto: str) -> str:
    """Pra busca de código: 'Concessionária' acha 'CONCESSIONARIA'."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(texto)) if not unicodedata.combining(c)).lower()
