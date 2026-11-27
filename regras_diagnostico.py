"""
regras_diagnostico.py — Regras de consistência da aba "🔎 Dados".

Cada regra é uma condição SQL (a mesma linguagem do Filtro SQL) que
deveria dar ZERO linhas. A aba roda todas e mostra quantas linhas
falharam em cada uma, onde estão (ano, operação, cidade) e quais são.

PRA ACRESCENTAR UMA REGRA: copie um bloco da lista e troque o nome, a
condição e a explicação. A condição usa os códigos das colunas (entre
aspas duplas quando têm ponto ou hífen). Regra que cita coluna que não
existe no banco é pulada, com aviso -- o banco pode mudar.

Decisão do Ro (26/11): as regras ficam no código (e não criadas pela
tela) pra não sumirem quando o app reinicia e pra cada regra nova ficar
registrada no git.
"""

# Índices com média de 1 a 5 (os `_media` da aba Índices)
_INDICES_1_A_5 = ["P46_media", "IQPA_media", "ICA_media", "IFA_media", "IAC_media", "IACOM_media", "IAA_media",
                  "IMC_media", "IMRU_media", "IMRS_media", "IM_media", "P23_media"]

REGRAS: list[dict] = [
    {
        "nome": "Mais crianças do que moradores",
        "condicao": '"P50.1_media" > "P50_media"',
        "explicacao": "P50.1 (crianças na casa) não pode ser maior que P50 (total de moradores). Achado em 26/11: 22 casos, todos de 2022.",
    },
    {
        "nome": "Idade fora de 18 a 100",
        "condicao": "F1_media < 18 OR F1_media > 100",
        "explicacao": "A pesquisa é com maiores de idade; acima de 100 é quase sempre erro de digitação.",
    },
    {
        "nome": "Índice fora de 1 a 5",
        "condicao": " OR ".join(f'"{c}" < 1 OR "{c}" > 5' for c in _INDICES_1_A_5),
        "explicacao": "Média de escala de 1 a 5 fora desse intervalo é erro de conta ou de digitação.",
    },
    {
        "nome": "Mesma cidade em mais de uma UF",
        "condicao": "CIDADES IN (SELECT CIDADES FROM dados GROUP BY CIDADES HAVING COUNT(DISTINCT UF) > 1)",
        "explicacao": "Mostra todas as entrevistas dessas cidades. Pode ser erro (como Novo Progresso em 2022, já corrigido) "
                      "ou cidades diferentes com o mesmo nome em estados diferentes -- confira no 'Onde está'.",
    },
]
