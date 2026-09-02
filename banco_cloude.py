import pandas as pd
import shutil

df = pd.read_parquet("df_completo_v2_corrigido.parquet")
df.head(300).to_parquet("df_exemplo_pequeno.parquet")
shutil.copy("df_completo_v2_corrigido.labels.json", "df_exemplo_pequeno.labels.json")