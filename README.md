# Gerador de banner

MVP funcional, testado contra o seu arquivo real (`df_completo_v2_corrigido_1.xlsx`,
1164 colunas, 102 respondentes). Roda local:

```bash
pixi add streamlit plotly pandas openpyxl   # ou pip install -r requirements.txt
pixi run streamlit run app.py
```

## O que já funciona

- **Leitura do padrão de cabeçalho duplo**: linha 1 = pergunta completa, linha 2 =
  código curto. A linha 1 vira rótulo de exibição — não é descartada.
- **Classificação automática de variável** por padrão de nome (`metadata.py`):
  resposta única, bloco de múltipla resposta, indicador pré-categorizado,
  companion numérico (`_media`), identificador e peso.
- **Cruzamento ponderado** (`crosstab_engine.py`): usa a coluna de peso
  (`PESO` por padrão) para o percentual, mas reporta N *não ponderado* na
  linha "Base Amostra" — é o N não ponderado que diz se uma célula é confiável.
- **Múltipla resposta tratada nativamente**, nos dois lados do cruzamento (stub
  ou banner), incluindo MR x MR. Sem pivotar nada manualmente antes — é
  exatamente o passo que hoje você faz no Power BI, só que automático e sob
  demanda, só para a variável que você seleciona.
- **N/A de indicador configurável** (manter como categoria vs. excluir da
  base) — testado no indicador `IAA` do seu arquivo: 97% N/A se mantido,
  base cai para 3 respondentes se excluído. A validação bateu com o número
  que eu tinha calculado à parte (97.1%) antes de existir qualquer código.
- **Alerta de N pequeno**: células com base abaixo do limiar (padrão 30,
  ajustável na UI) ficam destacadas, não escondidas.
- **Filtro de base** (ex.: só homem) aplicado antes de qualquer cruzamento —
  é um recorte de linhas do df, nada em `crosstab_engine.py` precisa saber
  que ele existe.
- **Leitura via parquet** (`convert_to_parquet.py`) para arquivo grande.
  Medido: 35s pra ler 14,5MB de xlsx via openpyxl, 0,46s pro mesmo dado em
  parquet — 76x. Rode `python convert_to_parquet.py seu_arquivo.xlsx` uma
  vez (é essa parte que demora, minutos pra arquivo de centenas de MB), e
  aponte o app pro `.parquet` gerado dali em diante, não pro `.xlsx`.
- Gráfico de barras (Plotly) e exportação CSV do banner gerado.

## Limitações conhecidas, deixadas explícitas de propósito

- **Base de um bloco MR = respondentes com ao menos 1 opção marcada.** Não dá
  para distinguir com certeza "pulou a pergunta por lógica de rota" de
  "respondeu e não marcou nada", a menos que o bloco tenha uma opção
  explícita tipo "Nenhuma". Se algum dos seus blocos tem uma coluna de
  screener/elegibilidade separada, me avisa que eu incluo isso na base.
- **N/A de indicador é uma regra global (keep/exclude) na UI atual, não por
  variável.** Os dados mostram que a taxa de N/A varia de 0% a 99% entre
  indicadores — uma regra por variável (com sugestão automática baseada na
  taxa) é o próximo incremento natural, não implementado ainda para manter
  esse primeiro entregável pequeno e testável.
- **Sem teste de significância estatística entre colunas do banner** (as
  letras que o SPSS/Quantum mostra). Isso ainda está em aberto — depende de
  decidir se o teste deve considerar o peso (Kish effective sample size,
  não N ponderado bruto) para não ficar overconfident. Vale uma conversa à
  parte antes de implementar, porque a escolha errada aqui é pior que não ter
  o teste.
- **Um banner por vez na tela** — múltiplas variáveis de banner aparecem
  lado a lado (igual ao SPSS), mas ainda não há um modo de "gerar todos os
  cruzamentos de uma vez e exportar". Dado o volume (1164 colunas → mais de
  600 mil pares possíveis), isso é intencional: cruzar sob demanda evita
  gerar volume de saída que ninguém vai ler.
