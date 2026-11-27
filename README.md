# Gerador de Banner (AEGEA/H2R)

Ferramenta em Python/Streamlit que substitui a montagem manual de tabelas
banner no SPSS/Quantum para pesquisas de satisfação. Cruza qualquer
variável de conteúdo (stub) contra qualquer variável de perfil (banner),
com tratamento automático de múltipla resposta, ponderação por amostra, e
uma aba dedicada a acompanhar os índices de tracking (IACOM, IMC, IM...)
ao longo do tempo.

Testado contra o banco de produção real: banco 2026, consolidado das
ondas de 2022 a 2026 (`ANO`), com 1629 colunas.

---

## Sumário

- [Estrutura de pastas](#estrutura-de-pastas)
- [Como rodar localmente](#como-rodar-localmente)
- [Como atualizar o banco](#como-atualizar-o-banco)
- [Como fazer deploy](#como-fazer-deploy)
- [Estatística](#estatística)
- [Índices: duas variáveis](#índices-duas-variáveis)
- [Banco: filtro SQL e visão da base](#banco-filtro-sql-e-visão-da-base)
- [Diagnóstico do banco (fora do app)](#diagnóstico-do-banco-fora-do-app)
- [Arquitetura](#arquitetura)
- [Convenções de dado descobertas](#convenções-de-dado-descobertas)
- [Convenções estatísticas](#convenções-estatísticas)
- [Limitações conhecidas](#limitações-conhecidas)
- [Backlog](#backlog-fases-futuras)

---

## Estrutura de pastas

```
app.py, metadata.py,        código do app -- fica na raiz porque o Streamlit
crosstab_engine.py,         Cloud exige `app.py` na raiz (Main file path) e
indices.py, sql_filter.py,  os módulos se importam entre si pelo nome
stats_engine.py,
diagnostico.py,             (diagnóstico: fora do app por enquanto)
regras_diagnostico.py
requirements.txt            o que o Streamlit Cloud lê pra montar o ambiente
pixi.toml / pixi.lock       ambiente local
data/
  processed/                banco que o app lê: .parquet + .labels.json
  sample/                   amostra pequena (~300 linhas) pra testes e conversas
tools/                      scripts offline, rodados à mão (nunca pelo app)
  convert_to_parquet.py       xlsx -> parquet otimizado
  list_variables.py           audita a classificação de variáveis
  make_sample.py              gera a amostra de data/sample
docs/                       documentação e relatórios gerados
tests/                      testes automáticos (pytest) -- `pixi run test`
```

Arquivos `.xlsx` brutos **não ficam no repositório** (`.gitignore`): são
entrada do ETL do banco, que vive separado deste projeto.

## Como rodar localmente

Ambiente gerenciado por [pixi](https://pixi.sh/). Requer os arquivos
`.parquet` + `.labels.json` já gerados (ver [Como atualizar o
banco](#como-atualizar-o-banco)) em `data/processed/`.

```bash
pixi install
pixi run streamlit run app.py
```

Testes automáticos: `pixi run test` (ou `python -m pytest tests`, com
`pip install pytest`). Rodar antes de subir mudança em conta estatística.

**Windows:** o `pixi.toml` declara só `platforms = ["linux-64"]`. Pra
usar pixi no Windows, adicione `"win-64"` a essa lista; ou use um
ambiente virtual com `pip install -r requirements.txt`.

Se não tiver `pixi.toml` ainda, criar do zero:

```bash
pixi init .
pixi add python=3.12
pixi add streamlit pandas plotly openpyxl pyarrow
```

**Cuidado com pasta sincronizada por nuvem (OneDrive/Google Drive/
Dropbox).** O `pixi install` cria hardlinks entre o cache de pacotes e o
ambiente do projeto -- isso falha silenciosamente ou trava com erro tipo
`failed to link ... No such file or directory` quando um dos dois lados
está dentro de uma pasta sincronizada (o cliente de sincronização
interfere na operação de link). Duas soluções, mais simples primeiro:

1. Mover a pasta do projeto pra fora da sincronização (`~/Documentos/...`
   em vez de `~/OneDrive/...`) -- git continua funcionando normal depois
   do `mv`, o `.git/config` vai junto.
2. Se precisar manter o projeto dentro da pasta sincronizada, usar
   ambientes desacoplados (`pixi config set --local detached-environments
   true`), que grava o ambiente físico fora da pasta e symlinka de volta.

## Como atualizar o banco

O app nunca lê `.xlsx` diretamente (ver [por quê](#por-que-parquet)).
Sempre que o banco original mudar:

```bash
python tools/convert_to_parquet.py /caminho/df_completo.xlsx [nome_da_aba]
```

Isso gera, ao lado do xlsx: `df_completo.parquet` e
`df_completo.labels.json`. Copiar os dois pra `data/processed/`
(sobrescrevendo os antigos), renomeados pra baterem com `DATA_PATH` em
`app.py` (por padrão, `data/processed/df_completo_2026.parquet`), e
commitar os dois junto no mesmo commit do código, se algum código mudou
também. **O `.parquet` e o `.labels.json` andam sempre juntos, com o mesmo
nome base** -- o app acha o segundo a partir do nome do primeiro.

A conversão demora minutos pro banco de produção inteiro (~7min medidos
no banco anterior, de 156MB/106mil linhas) -- é a parte lenta do
processo, mas só acontece nessa hora, offline, nunca dentro do Streamlit.

Depois de gerar um `.parquet` novo, vale rodar `list_variables.py` pra
conferir que a classificação de variável continua fazendo sentido:

```bash
python tools/list_variables.py data/processed/df_completo_2026.parquet
```

Ele grava a lista completa em `docs/variaveis_cruzamento.csv`.

Pra gerar a amostra pequena (até 60 respondentes por onda, embaralhados
com semente fixa, pra não vir só da primeira onda):

```bash
python tools/make_sample.py
```

## Como fazer deploy

**Streamlit Community Cloud**, a partir de um **repositório privado** no
GitHub -- o banco tem dado de respondente real, nunca deixar público. O
plano gratuito permite 1 repositório privado; confirmar a visibilidade em
Settings → General → Danger Zone antes do primeiro push de dado.

**Nem temporariamente.** O histórico do git guarda as versões antigas do
parquet, então abrir o repositório "só um instante" expõe também elas. Pra
dar acesso de leitura a uma pessoa ou ferramenta, conceda o acesso à
conta, sem mudar a visibilidade.

**Mudou o caminho ou o nome do parquet?** Altere o `DATA_PATH` do
`app.py` **no mesmo commit** que move o arquivo. Se forem commits
separados, o commit do meio sobe pro Cloud sem banco.

`requirements.txt` (não o `pixi.toml`) é o que o Streamlit Cloud lê pra
montar o ambiente remoto -- os dois arquivos convivem, cada um serve seu
propósito (local vs. deploy).

Depois de `git push`, o Cloud às vezes não detecta a mudança sozinho --
force pelo botão "Reboot app" no painel se o app não atualizar.

**Limite de memória**: o plano gratuito tem teto de ~1GB de RAM pro
processo inteiro. Foi por isso que `convert_to_parquet.py` otimiza tipo
de coluna antes de escrever o parquet (ver `_optimize_memory` no
código) -- sem isso, o banco de produção projetava ~1,3GB só pra existir
carregado, estourando o limite antes de qualquer conta rodar.

## Estatística

Aba **📐 Estatística**, parte **Descritivas** (os testes de hipótese são a
próxima fase). Serve pras variáveis **numéricas** do banco -- as colunas
`_media` (índices, notas, idade, nº de moradores...). O Cruzamento e o
Comparar continuam sendo o lugar das categóricas (frequência e %).

**Como usar**

1. Escolha uma ou mais variáveis numéricas (busca pelo código ou pelo texto
   da pergunta).
2. Se quiser, **quebre por** uma categórica (ANO, UF, OPERACAO...): a tabela
   ganha uma linha por grupo e o gráfico compara os grupos.
3. Marque os **cálculos** e escolha o **gráfico**.

**Escala ou contínua.** O app separa sozinho: até 11 valores diferentes é
**escala** (nota de 1 a 5, de 0 a 10); o resto é **contínua** (idade,
média de vários itens, valores). Isso decide o gráfico recomendado:

| Tipo | Recomendado | Por quê |
|---|---|---|
| escala | barras com o % de cada nota | numa escala de 1 a 5 a caixa do box plot quase sempre vai de 3 a 4 e esconde a diferença (dos 15 índices, 10 têm a caixa igual, com médias de 3,27 a 4,21) |
| contínua, sem quebra | histograma | mostra o formato da distribuição |
| contínua, com quebra | box plot | compara os grupos lado a lado |

Os outros gráficos continuam disponíveis; quando a escolha não serve pro
tipo, o app avisa e explica. A **dispersão** compara as duas primeiras
variáveis escolhidas, com uma amostra de 5.000 pontos (com 100 mil vira
uma mancha); variável de escala é tremida um pouco pra os pontos não
caírem todos no mesmo lugar.

**Os cálculos** (fórmulas e porquês em `stats_engine.py`):

- Tudo **ponderado pelo PESO** quando o peso está ligado na barra lateral;
  com o peso desligado, viram as fórmulas de sempre.
- **n** = entrevistas com valor; **n efetivo** = quanto a amostra "vale"
  depois do peso (no banco de 2026 inteiro, ~49% do n).
- **Erro padrão** considera o peso; **IC 95%** = média ± t × erro padrão,
  com n efetivo − 1 graus de liberdade.
- **Mediana e quartis** são sempre um valor que existe no dado (numa escala
  de 1 a 5, a mediana é 3 ou 4, nunca 3,5).
- **Amostra por cota:** erro padrão e IC são a convenção de mercado (como se
  a amostra fosse sorteada). Servem pra comparar a incerteza entre grupos,
  não como probabilidade de erro de verdade.

Grupo sem nenhuma resposta (ex.: ano em que a pergunta não foi feita) sai
da tabela, com aviso. Respeita o filtro SQL e os da barra lateral.

**Índices na Estatística:** as colunas de índice são lidas com a mesma regra
da aba Índices (no IACOM, quem não teve contato com a comunicação entra
como nota 3), pra o mesmo índice dar o mesmo número nas duas abas.

**Índice de multiplicidade** (no Cruzamento): quando a variável de linha é
de múltipla escolha, aparece embaixo do banner a média de opções marcadas
por respondente, por coluna, e ela vai pras Notas da exportação. É a soma
do %COLUNA ÷ 100, então bate com a tabela. A base é quem marcou pelo menos
uma opção (o banco não separa quem pulou a pergunta), por isso o índice
nunca fica abaixo de 1.

## Índices: duas variáveis

Na aba **📈 Índices**, a visão **Duas variáveis** cruza UM índice por duas
variáveis ao mesmo tempo -- ex.: cidade nas linhas e ano nas colunas. As
colunas são a variável de "Segmentar por"; as linhas, a que você escolher
(padrão CIDADES).

- **Tabela:** média do índice por célula, com Total por linha e por coluna;
  célula com menos entrevistas que o limiar da barra lateral fica em
  amarelo (como no banner). Dá pra ver o n de cada célula.
- **Gráfico:** uma linha por categoria ao longo das colunas, Total
  tracejado. Com mais de 8 linhas (ex.: 130 cidades), todas ficam em cinza
  e você destaca até 3. Ponto vazado = célula com n pequeno.
- **Exportação:** a grade em Excel e uma tabela longa em CSV com média, n e
  cobertura de cada combinação, com os filtros nas Notas.
- O Total de cada ano muda conforme as cidades que entraram naquele ano.
  Pra comparar ano contra ano sem esse efeito, use antes o filtro SQL das
  cidades presentes em todos os anos (exemplo pronto na aba 🔎 Banco).

## Banco: filtro SQL e visão da base

Aba **🔎 Banco: filtro e visão** (a última). É o lugar pra filtro complexo
e pra **olhar o banco por dentro** -- ver quem está na base que o app está
usando:

1. **Filtro SQL** (em cima): recorte que a barra lateral não faz.
2. **A base agora** (embaixo): as entrevistas que sobraram depois do SQL e
   dos filtros da barra lateral -- quantas, onde estão, quais são.

### Filtro SQL

Serve pra recorte que a barra lateral não
faz: OR, número (`PESO > 1.5`), múltipla escolha, e condição que depende
de outras linhas ("cidades presentes nos 5 anos"). É pra quem já sabe SQL.

**Como usar**

1. Escreva só a **condição** -- o que viria depois do `WHERE`. A tabela se
   chama `dados`.
2. Clique **Aplicar**. A partir daí o filtro vale pro app inteiro.
3. **Limpar filtro** (na aba, no aviso do topo ou na barra lateral) tira o
   filtro. Com a caixa vazia, o app funciona como antes.

**Regras de escrita**

- Coluna = **código** da pergunta (`P25`, `OPERACAO`), não o texto. O
  código não tem acento. Pra achar o código, use "Procurar o código da
  pergunta", dentro da aba (busca pelo código ou por palavras da pergunta).
- Código com ponto, hífen ou espaço vai entre **aspas duplas**: `"P29.1"`,
  `"P46.1-Outros"`. A coluna "como escrever" da busca já mostra pronto.
- Valor de texto vai entre **aspas simples**: `OPERACAO = 'PLENA'`.
  Maiúscula e acento contam (`'Plena'` não acha nada).
- `ANO` é texto: `ANO = '2025'`.
- Múltipla escolha é verdadeiro/falso: `"P46.1-Outros"` sozinho = quem
  marcou. Cuidado: `NOT "P46.1-Outros"` inclui quem nunca recebeu a
  pergunta.
- Uma condição só: sem `;` e sem comentário.

**Exemplos**

```sql
OPERACAO = 'PLENA' AND ANO IN ('2024', '2025')
"P21.7_media" >= 4 AND PESO > 0.5
lower(CIDADES) LIKE '%rio%'
CIDADES IN (SELECT CIDADES FROM dados WHERE OPERACAO = 'PLENA'
            GROUP BY CIDADES HAVING COUNT(DISTINCT ANO) = 5)
AND OPERACAO = 'PLENA'
```

**Como ele se combina com o resto**

- O SQL é o **primeiro** filtro e roda sobre o **banco inteiro**. Os
  filtros da barra lateral vêm depois, em cascata, já só com os valores
  que sobraram. (Se o SQL rodasse depois, uma subconsulta como "presente
  em todos os anos" daria errado com ANO filtrado na barra lateral.)
- Vale pra todas as abas. A coluna "Total geral (sem filtro)" do banner
  aparece quando há filtro SQL.
- Enquanto houver filtro SQL, um **aviso fixo** fica no topo de todas as
  abas e na barra lateral, com a condição e o número de entrevistas.
- A condição e os filtros da barra lateral vão escritos **embaixo do
  banner** e nas **Notas** do CSV/Excel (banner e índices).

**Proteções** (detalhe em `sql_filter.py`): sem acesso a arquivo,
internet ou configuração; a única tabela visível é `dados`; limite de
memória e de 10 s por condição; condição que não deixa ninguém na base
não é aplicada. As colunas retiradas do app (IMCE/IMCT) não existem na
tabela `dados`. Erro aparece em português; a mensagem original do DuckDB
fica num campo que abre e fecha, pra quem quiser conferir.

**Dependência:** `duckdb` (testado com 1.5.3 e 2.0.1).

### A base agora

- **Quantas**: entrevistas na base, em **contagem sem peso** (é pra
  enxergar o banco, não pra estimar -- não bate com o % do banner, e não
  deve). Embaixo, a lista de filtros que está valendo.
- **Só quem entrou no cruzamento**: chave opcional. Ligada, fica só quem
  respondeu o stub escolhido na barra lateral -- o número bate com o
  "Base Amostra NA" da coluna Total do banner (conferido: P25, 70.953).
  Em stub de múltipla escolha não bate de propósito: lá o NA soma menções,
  aqui são pessoas.
- **Contar por** (onde estão): contagem por ANO, OPERACAO e CIDADES; dá pra
  trocar ou somar colunas (até 6). Mostra os 15 maiores valores de cada.
- **Colunas**: já vêm ID_UNICO, ANO, OPERACAO, CIDADES, as colunas que a
  condição SQL usa, as dos filtros da barra lateral, o stub e o banner
  (stub de múltipla escolha = uma coluna por opção). Dá pra adicionar
  qualquer coluna do banco ou tirar. Ao trocar stub, banner ou filtro, as
  colunas deles entram sozinhas; o que você adicionou ou tirou à mão fica.
- **Ver as linhas**: até 2.000 na tela; o CSV leva todas.
- **Resumo das colunas**: nulos, distribuição (contagem bruta) e
  `describe()` das colunas escolhidas.

## Diagnóstico do banco (fora do app)

`diagnostico.py` e `regras_diagnostico.py` varrem o banco atrás de
inconsistência (regras que deveriam dar zero linhas, texto × média,
preenchimento por ano, colunas idênticas ou partidas entre anos,
duplicadas na chave). Ficaram **fora do app** desde 27/11: achar erro no
banco é trabalho do ETL, e o app é pra analisar. O plano é rodar como
ferramenta separada (script ou app próprio) -- ainda não existe. As
funções seguem testadas em `tests/test_diagnostico.py`.

**Como criar uma regra nova**: acrescente um item em
`regras_diagnostico.py`, com `nome`, `condicao` (SQL, mesma escrita do
filtro) e `explicacao`. A condição descreve o **erro** (o que deveria dar
zero linhas).

## Arquitetura

```
metadata.py            classifica cada coluna (SR/MR/indicador/peso/...)
        │
        ├──> crosstab_engine.py     motor de cruzamento ponderado
        ├──> indices.py             tendência/quadrante dos índices
sql_filter.py          filtro SQL (DuckDB): condição -> máscara de linhas
stats_engine.py        estatística descritiva ponderada (aba Estatística)
diagnostico.py         diagnóstico do banco (fora do app); regras em regras_diagnostico.py
        │
tools/convert_to_parquet.py   script offline: xlsx -> parquet otimizado
tools/list_variables.py       script offline: audita a classificação
        │
        v
app.py                  interface Streamlit (só orquestra, não calcula)
```

| Arquivo | Papel |
|---|---|
| `metadata.py` | Lê o cabeçalho duplo do Excel e classifica cada coluna: resposta única (SR), múltipla resposta (MR), indicador pré-categorizado, ou companion numérico (`_media`). Todo o resto do projeto depende do dicionário `dict[str, VariableMeta]` que esse módulo produz. |
| `crosstab_engine.py` | Motor de cruzamento ponderado. Normaliza qualquer variável (SR ou MR) pro mesmo formato longo (`resp_id \| category \| weight`) antes de cruzar -- é isso que elimina a necessidade de pivotar múltipla resposta manualmente. Calcula NA, %LINHA, %COLUNA, e os avisos de cobertura. |
| `indices.py` | Média ponderada, tendência entre ondas e correlação ponderada entre índices (pro quadrante). Separado do motor de cruzamento porque índice não tem categoria pra cruzar, tem um número. |
| `sql_filter.py` | Filtro SQL da aba 🔎 Banco: confere a condição, roda no DuckDB com as travas de segurança e devolve quais entrevistas passaram; traduz o erro pro português. Não conhece o Streamlit. |
| `stats_engine.py` | Estatística descritiva ponderada das numéricas (média, erro padrão, IC, quantis, moda...) e os números dos gráficos (distribuição, histograma, box plot). Não conhece o Streamlit; testado em `tests/`. |
| `diagnostico.py` | Funções do diagnóstico (preenchimento por ano, texto × média, colunas idênticas/complementares, duplicadas na chave, onde está). **Não é chamado pelo app** (ver Diagnóstico do banco). Recebe DataFrame, devolve DataFrame; não conhece o Streamlit; testado em `tests/test_diagnostico.py`. |
| `regras_diagnostico.py` | Lista das regras de consistência (condições SQL que deveriam dar zero linhas). É aqui que se cria regra nova. |
| `app.py` | Interface Streamlit -- cinco abas (Cruzamento / Comparar / Índices / Estatística / Banco: filtro e visão). Banco fixo: lê sempre o `.parquet` de caminho relativo ao próprio arquivo, sem upload nem input de caminho na tela. |
| `tools/convert_to_parquet.py` | Script que roda fora do Streamlit, converte o `.xlsx` bruto pra `.parquet` otimizado. Único jeito de atualizar o banco. |
| `tools/list_variables.py` | Audita a classificação de variáveis, gera `docs/variaveis_cruzamento.csv` pra revisão manual. |
| `tools/make_sample.py` | Gera a amostra de `data/sample/`, estratificada por onda. |

Cada módulo tem um docstring no topo explicando seu papel com mais
detalhe -- esta tabela é só o mapa; o "porquê" de cada decisão de design
está comentado direto no código, perto de onde a decisão foi tomada.

### Por que parquet

Ler um xlsx largo com openpyxl custa minutos, não segundos -- medido
~35s pra 14,5MB de teste, ~7min pros ~156MB/106mil linhas do banco de
produção. Parquet é colunar e ~76x mais rápido pra ler de volta, além de
~16x menor em disco (compressão colunar aproveita a esparsidade natural
de blocos de múltipla resposta, onde a maioria das células está vazia).

## Convenções de dado descobertas

Padrões que não são óbvios e custaram investigação contra o banco real
-- documentados aqui e, com mais detalhe, nos comentários de
`metadata.py`:

- **`-` no nome curto não é suficiente pra indicar múltipla resposta.**
  Só vira MR se a tag `(RM - ...)` estiver no rótulo completo, OU se o
  código antes do `-` se repetir em 2+ colunas. Item de bateria/grade
  (ex.: "P27.3.1-Relacionamento com o cliente?") usa o mesmo formato
  sem ser múltipla resposta de verdade -- cada código ali é único.
- **Todo indicador (`IACOM`, `IMC`...) tem um companion numérico**
  (`IACOM_media`, via `scale_base`) -- é a média real, mais útil que a
  faixa categórica pra tracking, e é o que alimenta a aba Índices. Só
  vira indicador se tiver o rótulo `_c` **e** o companion no banco.
  Duas exceções documentadas em `indices._MEDIA_ALIASES`: `IAG` usa
  `P46_media` e `ISDE` usa `P23_media` (as colunas "óbvias" não servem).
- **ISDE foi descontinuado em 2026.** Sua fonte (`P23_media`) tem dado de
  2022 a 2025 e 0% em 2026: a série termina em 2025 e o ISDE some do
  quadrante da onda mais recente, com aviso na tela. Não é bug.
- **IMCE e IMCT não são usados** pelo app: não têm companion `_media` no
  banco, então ficam só como variáveis comuns de cruzamento.
- **Taxa de N/A varia de 0% a 99% entre indicadores** -- não existe
  regra única de tratamento que sirva pra todos (é por isso que
  "manter/excluir N/A" é uma escolha na tela, não fixa no código).
- **`Total` é sempre a base do stub**, nunca da variável de banner
  específica -- confusão recorrente na prática, resolvida com uma
  legenda de cobertura ("resposta de X de Y") acima da tabela.
- **Banco é consolidado de várias ondas (`ANO`)**; perguntas específicas
  podem só existir numa onda -- gera aviso automático de cobertura
  concentrada quando isso acontece.

## Convenções estatísticas

- **Peso amostral é uma escolha na tela** ("Aplicar peso amostral
  (PESO)", em "3. Regras"), **ligado por padrão**, com um aviso fixo no
  topo do app dizendo se o peso está ligado ou desligado -- pra ninguém
  analisar dado ponderado sem saber. Ligado, o % usa a coluna `PESO` (ou
  1.0 uniforme se não houver); desligado, usa contagem bruta. Vale pro
  app inteiro, inclusive Índices e Estatística (até 25/11 a aba Índices
  era sempre ponderada; desde 26/11 segue a chave). **NA/Base
  Amostra é sempre não ponderado** -- é o N real que diz se uma célula é
  estatisticamente confiável, não o N ponderado (que pode parecer maior
  ou menor que a realidade).
- **Base do %LINHA = universo da pergunta** (toggle, ligado por padrão):
  um grupo sai do denominador se menos de 5% dele respondeu a pergunta e
  ela tem resposta em outros grupos. A regra é automática e não conhece
  o desenho do questionário -- a legenda acima da tabela mostra quem
  saiu, e vale conferir.
- **%LINHA** = dentro da categoria do stub, qual a fatia de cada opção do
  banner ("como esse grupo se comporta"). **%COLUNA** = dentro da opção
  do banner, qual a fatia de cada categoria do stub ("quem escolheu essa
  opção" -- perfil, não comportamento).
- **Base de uma variável MR** = respondentes com pelo menos 1 opção
  marcada no bloco. Limitação conhecida: não dá pra distinguir com
  certeza "pulou a pergunta" de "respondeu e marcou zero opções", a menos
  que o bloco tenha uma opção explícita tipo "Nenhuma".
- **Stub de múltipla resposta é suportado, mas soma passa de 100%** --
  uma pessoa com 2+ seleções conta em 2+ linhas da tabela. Correto
  matematicamente, só diferente do caso comum (stub de resposta única).
- **Importância de índice** (quadrante) = correlação de Pearson
  ponderada, pessoa por pessoa, contra um índice de referência
  escolhível. Não é causa -- é "importância derivada", aproximação padrão
  de mercado quando não se pergunta importância diretamente.

## Limitações conhecidas

- Regra de N/A em indicador é global (manter/excluir) na tela, não
  configurável por variável.
- Filtro de base por variável de múltipla resposta não suportado na barra
  lateral; dá pra fazer pelo Filtro SQL da aba 🔎 Banco.
- Quadrante "Tendência x Nível" precisa de 2+ ondas na base filtrada pra
  gerar ponto -- com filtro restringindo a uma onda só, o quadrante fica
  vazio pra esse eixo (degrada graciosamente, não quebra).
- Correlação de importância é par a par, não regressão múltipla -- não
  controla a influência de outros índices ao calcular um par específico.

## Backlog (fases futuras)

- Box plot das variáveis que compõem um índice.
- Mais testes automatizados: `tests/` já cobre `stats_engine.py`, o índice
  de multiplicidade e `diagnostico.py`; falta o resto do `crosstab_engine.py` e o `metadata.py`
  (começando pelas perguntas de múltipla resposta).
- Clusterizar assuntos/variáveis e visualizar força de ligação com
  grafos -- usar correlação (Pearson pra contínua, Spearman pra ordinal,
  V de Cramér pra categórica sem ordem), considerar Análise Fatorial
  antes de ir pra grafo (é a ferramenta clássica pra achar dimensão
  latente em itens de escala de concordância, que é a maior parte do
  questionário). Inspirado no Metroverse do Harvard Growth Lab, sem
  necessidade de replicar o visual exato (Voronoi/embedding).
- Migração pra Neon (Postgres), se o projeto crescer pra multi-banco:
  desenho recomendado é uma tabela `respondentes` (SR/indicador, uma
  linha por pessoa) + uma tabela `selecoes_rm` (`respondente_id`,
  `codigo_pergunta`, `opcao`), ligadas por FK -- o schema relacional
  padrão pra dado de pesquisa com múltipla resposta.
