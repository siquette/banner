# Contexto: Gerador de Banner (AEGEA/H2R)

Ferramenta em Python/Streamlit que substitui a montagem manual de tabelas
banner no SPSS/Quantum pra pesquisas de satisfação. Cruza qualquer
variável de conteúdo (stub) contra qualquer variável de perfil (banner),
com tratamento automático de múltipla resposta, ponderação por amostra, e
uma aba dedicada a acompanhar os índices de tracking (IACOM, IMC, IM...)
ao longo do tempo. Testado contra o banco de produção real: 106.217
respondentes, 1164 colunas, consolidado de várias ondas (`ANO`).

**A documentação completa (arquitetura, convenções de dado, convenções
estatísticas, limitações conhecidas, backlog) está no `README.md` do
projeto -- se ele estiver anexado no Project Knowledge, é a referência
principal, mais completa que este resumo. Este texto é só o "atalho".**

## Estado atual (resumo)

- Seis módulos: `metadata.py` (classificação de variável + ordem/cor de
  categoria), `crosstab_engine.py` (motor de cruzamento ponderado),
  `indices.py` (tendência/quadrante/importância dos índices), `app.py`
  (interface, duas abas), `convert_to_parquet.py` e `list_variables.py`
  (scripts offline).
- Código já passou por uma revisão de manutenção: docstring completo em
  toda função pública, seções demarcadas dentro de cada arquivo, funções
  grandes quebradas em menores (ex.: a aba Índices tem uma função por
  visão -- geral/individual/quadrante -- em vez de um `if/elif` só).
- Aba Índices tem três visões: Visão geral (com alternador Gráfico/
  Tabela), Individual, e Quadrante (dois eixos possíveis: Tendência x
  Nível, ou Importância x Desempenho via correlação ponderada).
- Gráficos: três tipos (Barras, Barra horizontal, Linha), com controle de
  espaçamento de barra na tela, legenda horizontal embaixo (não à
  direita), e cor semântica (verde->vermelho) pra categoria de escala
  reconhecida (Ótimo/Bom/Regular/Ruim/Péssimo e afins) -- ver
  `metadata.sort_categories`/`category_color`.

## Como eu (Claude) devo trabalhar nesse projeto

- **Mandar só o trecho de código que muda na conversa, nunca o arquivo
  inteiro** -- o arquivo completo sempre vai como artefato pra download.
  Isso é economia de token deliberada, pedida explicitamente.
- Calibrar o tanto de teste pelo risco: mudança em `crosstab_engine.py`/
  `metadata.py` (onde mora a estatística) merece bateria de teste
  pesada (comparação campo a campo contra o comportamento anterior,
  não só "roda sem erro"). Mudança de UI/estética (cor, slider, texto)
  não precisa do mesmo tanto.
- Testar rodando de verdade antes de entregar, não só `py_compile` --
  já escapou mais de um bug (erro de formatação de string, parâmetro
  obsoleto do Streamlit) que só aparecia com o código executando.
- Ao editar código, mandar os arquivos relacionados juntos, no mesmo
  commit -- já aconteceu import quebrar por causa de arquivos de levas
  diferentes misturados no repo local.

## Pendências conhecidas

- Regra de N/A em indicador é global (manter/excluir), não configurável
  por variável.
- Filtro de base por variável de múltipla resposta não suportado.
- Correlação de importância (quadrante) é par a par, não regressão
  múltipla.

## Backlog (fases futuras)

- Box plot das variáveis que compõem um índice.
- Clusterizar assuntos/variáveis e visualizar força de ligação com
  grafos -- usar correlação (Pearson pra contínua, Spearman pra ordinal,
  V de Cramér pra categórica sem ordem), considerar Análise Fatorial
  antes de ir pra grafo. Inspirado no Metroverse do Harvard Growth Lab.
- Migração pra Neon (Postgres) se o projeto crescer pra multi-banco --
  desenho recomendado: tabela `respondentes` (uma linha por pessoa) +
  tabela `selecoes_rm` (`respondente_id`, `codigo_pergunta`, `opcao`).
