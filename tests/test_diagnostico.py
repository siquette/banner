"""
Testes de diagnostico.py com um banco pequeno montado com os problemas
que já apareceram no banco real (cada teste planta um e confere que a
verificação acha -- e que não acusa o que está certo).
"""

import numpy as np
import pandas as pd

import diagnostico as dg

NOTA = {"INSATISFEITO": 2, "SATISFEITO": 4}


def _banco() -> pd.DataFrame:
    anos = ["2022"] * 4 + ["2023"] * 4 + ["2024"] * 4
    texto = ["SATISFEITO", "INSATISFEITO", "SATISFEITO", "NÃO SABE"] * 3
    media = [NOTA.get(t, np.nan) for t in texto]
    media[2] = 2.0                     # divergência plantada: SATISFEITO com nota 2 (2022)
    media[8], media[9] = np.nan, np.nan  # 2024: texto sem média (como a P23 em 2026)
    return pd.DataFrame({
        "ID": list(range(12)),
        "ANO": anos,
        "CONCESSIONARIA": ["X"] * 12,
        "P23": texto,
        "P23_media": media,
        "P2": ["a", "b", "c", "d", None, None, None, None, "e", "f", "g", "h"],  # vazia só em 2023 (no meio)
        "NOVA": [None] * 8 + ["x"] * 4,                                            # só começou em 2024: NÃO é sinal
        "COPIA_A": [1, 2, 3, 4] * 3,
        "COPIA_B": [1, 2, 3, 4] * 3,                                               # idêntica à A
        "CONSTANTE": [0] * 12,                                                     # constante: não entra
    })


def test_preenchimento_acha_ano_vazio_no_meio_e_nao_acusa_pergunta_nova():
    p = dg.preenchimento_por_ano(_banco())
    assert p.loc["P2", "anos_vazios_no_meio"] == "2023"
    assert p.loc["P2", "sinal"] == "ano vazio no meio da série"
    assert p.loc["NOVA", "sinal"] == ""
    assert p.loc["P2", "2022"] == 100 and p.loc["P2", "2023"] == 0


def test_texto_vs_media():
    t = dg.texto_vs_media(_banco()).set_index("media")
    assert t.loc["P23_media", "divergem"] == 1
    assert t.loc["P23_media", "texto_sem_media"] == 2       # "NÃO SABE" sem nota não conta
    assert t.loc["P23_media", "media_sem_texto"] == 0
    assert "2024: 2" in t.loc["P23_media", "anos"] and "2022: 1" in t.loc["P23_media", "anos"]
    m = dg.linhas_texto_vs_media(_banco(), "P23", "P23_media")
    assert list(np.flatnonzero(m)) == [2, 8, 9]


def test_texto_vs_media_ignora_media_continua():
    d = pd.DataFrame({"F1": ["18 a 24"] * 20, "F1_media": np.arange(20, dtype=float)})  # faixa x idade: não compara
    assert dg.texto_vs_media(d.assign(ANO="2022")).empty


def test_colunas_identicas_sem_constantes():
    grupos = dg.colunas_identicas(_banco())
    assert ["COPIA_A", "COPIA_B"] in grupos
    assert not any("CONSTANTE" in g for g in grupos)


def test_colunas_identicas_padrao_igual_valor_diferente_nao_e_igual():
    d = pd.DataFrame({"a": ["Sim", "Não", "Sim"], "b": ["x", "y", "x"]})
    assert dg.colunas_identicas(d) == []


def test_complementares():
    d = pd.DataFrame({
        "ANO": ["2022", "2023", "2024"],
        "P48_media": [7.0, 8.0, np.nan],
        "P37_media": [np.nan, np.nan, 9.0],
    })
    rotulos = {"P48_media": "P48. (MÉDIA) Numa escala de 0 a 10 o quanto recomendaria o atendimento",
               "P37_media": "P37. (média) Numa escala de 0 a 10 o quanto recomendaria o atendimento"}
    c = dg.colunas_complementares(d, rotulos)
    assert len(c) == 1 and {c.loc[0, "coluna_a"], c.loc[0, "coluna_b"]} == {"P48_media", "P37_media"}


def test_duplicadas_e_codigos_dup():
    d = _banco()
    assert dg.duplicadas_na_chave(d).set_index("chave").loc["ID + ANO + CONCESSIONARIA", "linhas_repetidas"] == 0
    d2 = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    assert dg.duplicadas_na_chave(d2).set_index("chave").loc["ID + ANO + CONCESSIONARIA", "linhas_repetidas"] == 2
    assert dg.codigos_duplicados(["P28", "P28__dup1", "X"]) == ["P28__dup1"]


def test_onde_esta():
    d = _banco()
    o = dg.onde_esta(d, d["ANO"] == "2023", ["ANO"])
    assert o["ANO"].to_dict() == {"2023": 4}
