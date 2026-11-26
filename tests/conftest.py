"""Deixa os módulos da raiz (stats_engine, crosstab_engine...) importáveis nos testes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
