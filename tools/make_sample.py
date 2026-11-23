"""Gera amostra estratificada por ANO (até 60 por onda) + copia o labels.json."""
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "df_completo_2026.parquet"
DST = ROOT / "data" / "sample" / "df_exemplo_pequeno.parquet"

df = pd.read_parquet(SRC)
shuffled = df.sample(frac=1, random_state=42)  # embaralha com semente fixa (reprodutível)
sample = shuffled[shuffled.groupby("ANO", observed=True).cumcount() < 60]  # até 60 por onda
sample.to_parquet(DST)
shutil.copy(SRC.with_suffix("").with_suffix(".labels.json"),
            DST.with_suffix("").with_suffix(".labels.json"))
print(f"{len(sample)} linhas, {sample['ANO'].nunique()} ondas -> {DST}")