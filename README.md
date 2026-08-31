# Gerador de Banner (AEGEA/H2R)

Ferramenta em Python/Streamlit que substitui a montagem manual de tabelas
banner no SPSS/Quantum para pesquisas de satisfação. Cruza qualquer
variável de conteúdo (stub) contra qualquer variável de perfil (banner),
com tratamento automático de múltipla resposta, ponderação por amostra, e
uma aba dedicada a acompanhar os índices de tracking (IACOM, IMC, IM...)
ao longo do tempo.

Testado contra o banco de produção real: 106.217 respondentes, 1164
colunas, consolidado de várias ondas (`ANO`).

---

## Sumário

- [Como rodar localmente](#como-rodar-localmente)
- [Como atualizar o banco](#como-atualizar-o-banco)
- [Como fazer deploy](#como-fazer-deploy)
- [Arquitetura](#arquitetura)
- [Convenções de dado descobertas](#convenções-de-dado-descobertas)
- [Convenções estatísticas](#convenções-estatísticas)
- [Limitações conhecidas](#limitações-conhecidas)
- [Backlog](#backlog-fases-futuras)

---

## Como rodar localmente

Ambiente gerenciado por [pixi](https://pixi.sh/). Requer os arquivos
`.parquet` + `.labels.json` já gerados (ver [Como atualizar o
banco](#como-atualizar-o-banco)) na mesma pasta do `app.py`.

```bash
pixi install
pixi run streamlit run app.py
```

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
python convert_to_parquet.py /caminho/df_completo.xlsx [nome_da_aba]
```

Isso gera, ao lado do xlsx: `df_completo.parquet` e
`df_completo.labels.json`. Copiar os dois pra pasta do projeto
(sobrescrevendo os antigos), renomeados pra baterem com `DATA_PATH` em
`app.py` (por padrão, `df_completo_v2_corrigido.parquet`), e commitar os
dois junto no mesmo commit do código, se algum código mudou também.

A conversão demora minutos pro banco de produção inteiro (~7min pros
156MB/106mil linhas medidos) -- é a parte lenta do processo, mas só
acontece nessa hora, offline, nunca dentro do Streamlit.

Depois de gerar um `.parquet` novo, vale rodar `list_variables.py` pra
conferir que a classificação de variável continua fazendo sentido:

```bash
python list_variables.py df_completo.parquet
```

## Como fazer deploy

**Streamlit Community Cloud**, a partir de um **repositório privado** no
GitHub -- o banco tem dado de respondente real, nunca deixar público. O
plano gratuito permite 1 repositório privado; confirmar a visibilidade em
Settings → General → Danger Zone antes do primeiro push de dado.

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

## Arquitetura

```
metadata.py            classifica cada coluna (SR/MR/indicador/peso/...)
        │
        ├──> crosstab_engine.py     motor de cruzamento ponderado
        ├──> indices.py             tendência/quadrante dos índices
        │
convert_to_parquet.py   script offline: xlsx -> parquet otimizado
list_variables.py       script offline: audita a classificação
        │
        v
app.py                  interface Streamlit (só orquestra, não calcula)
```

| Arquivo | Papel |
|---|---|
| `metadata.py` | Lê o cabeçalho duplo do Excel e classifica cada coluna: resposta única (SR), múltipla resposta (MR), indicador pré-categorizado, ou companion numérico (`_media`). Todo o resto do projeto depende do dicionário `dict[str, VariableMeta]` que esse módulo produz. |
| `crosstab_engine.py` | Motor de cruzamento ponderado. Normaliza qualquer variável (SR ou MR) pro mesmo formato longo (`resp_id \| category \| weight`) antes de cruzar -- é isso que elimina a necessidade de pivotar múltipla resposta manualmente. Calcula NA, %LINHA, %COLUNA, e os avisos de cobertura. |
| `indices.py` | Média ponderada, tendência entre ondas e correlação ponderada entre índices (pro quadrante). Separado do motor de cruzamento porque índice não tem categoria pra cruzar, tem um número. |
| `app.py` | Interface Streamlit -- duas abas (Cruzamento / Índices). Banco fixo: lê sempre o `.parquet` de caminho relativo ao próprio arquivo, sem upload nem input de caminho na tela. |
| `convert_to_parquet.py` | Script que roda fora do Streamlit, converte o `.xlsx` bruto pra `.parquet` otimizado. Único jeito de atualizar o banco. |
| `list_variables.py` | Audita a classificação de variáveis, gera `variaveis_cruzamento.csv` pra revisão manual. |

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
  faixa categórica pra tracking, e é o que alimenta a aba Índices.
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

- **% sempre ponderado** (usa a coluna `PESO`, ou 1.0 uniforme se não
  houver); **NA/Base Amostra sempre não ponderado** -- é o N real que diz
  se uma célula é estatisticamente confiável, não o N ponderado (que pode
  parecer maior ou menor que a realidade).
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
- Filtro de base por variável de múltipla resposta não suportado.
- Quadrante "Tendência x Nível" precisa de 2+ ondas na base filtrada pra
  gerar ponto -- com filtro restringindo a uma onda só, o quadrante fica
  vazio pra esse eixo (degrada graciosamente, não quebra).
- Correlação de importância é par a par, não regressão múltipla -- não
  controla a influência de outros índices ao calcular um par específico.

## Backlog (fases futuras)

- Box plot das variáveis que compõem um índice.
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
