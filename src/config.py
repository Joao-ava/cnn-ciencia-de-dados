"""Configurações compartilhadas do projeto (use sempre daqui, não copie os valores)."""

from pathlib import Path

DATASET_ID = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SIZE = 0.15
TEST_SIZE = 0.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_PATH = DATA_DIR / "splits.csv"  # versionado no git: todos usam a mesma divisão
CACHE_DIR = DATA_DIR / "cache"         # ignorado no git: imagens pré-processadas
