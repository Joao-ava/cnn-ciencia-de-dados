"""Carregamento de dados para a CNN.

Dataset: Fruit and Vegetable Disease (Healthy vs Rotten) - Kaggle
https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

Uso rápido:
    from src.data_loader import load_datasets
    train_ds, val_ds, test_ds, class_names = load_datasets()
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf

from src import config

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
SPLITS = ("train", "val", "test")
AUTOTUNE = tf.data.AUTOTUNE


def download_dataset() -> Path:
    """Baixa o dataset (ou reutiliza o cache do kagglehub) e retorna o caminho."""
    path = Path(kagglehub.dataset_download(config.DATASET_ID))
    print("Path to dataset files:", path)
    return path


def find_image_files(root: Path) -> tuple[list[str], list[str]]:
    """Percorre `root` e retorna (caminhos, rótulos).

    O rótulo é o nome da pasta onde a imagem está (ex.: 'Apple__Healthy').
    Funciona mesmo se o dataset tiver uma pasta raiz extra dentro do zip.
    """
    paths, labels = [], []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if Path(name).suffix.lower() in IMG_EXTENSIONS:
                paths.append(os.path.join(dirpath, name))
                labels.append(Path(dirpath).name)
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {root}")
    return paths, labels


def stratified_split(labels: np.ndarray, val_size: float, test_size: float, seed: int):
    """Divide os índices em treino/validação/teste mantendo a proporção de cada classe."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_val = int(round(len(idx) * val_size))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    return (rng.permutation(train_idx), rng.permutation(val_idx), rng.permutation(test_idx))


# ---------------------------------------------------------------------------
# Splits fixos (data/splits.csv)
# ---------------------------------------------------------------------------

def read_excluded(excluded_path: Path | None = None) -> set[str]:
    """Caminhos (relativos à raiz do dataset) listados em data/excluidas.csv."""
    excluded_path = Path(excluded_path or config.EXCLUDED_PATH)
    if not excluded_path.exists():
        return set()
    with open(excluded_path, newline="", encoding="utf-8") as f:
        return {r["path"] for r in csv.DictReader(f)}


def create_splits(data_dir: Path, splits_path: Path | None = None) -> list[dict]:
    """Gera a divisão treino/val/teste e salva em CSV.

    Os caminhos ficam relativos à raiz do dataset, então o mesmo CSV funciona
    em qualquer máquina (inclusive Colab). A estratificação usa a classe
    completa (ex.: 'Apple__Healthy'), então o modo binário também fica estratificado.
    Imagens listadas em data/excluidas.csv ficam de fora.
    """
    data_dir = Path(data_dir)
    splits_path = Path(splits_path or config.SPLITS_PATH)
    paths, labels = find_image_files(data_dir)
    excluded = read_excluded()
    keep = [i for i, p in enumerate(paths) if Path(p).relative_to(data_dir).as_posix() not in excluded]
    paths, labels = [paths[i] for i in keep], [labels[i] for i in keep]
    tr, va, te = stratified_split(np.array(labels), config.VAL_SIZE, config.TEST_SIZE, config.SEED)

    split_of = {}
    for name, idx in zip(SPLITS, (tr, va, te)):
        for i in idx:
            split_of[int(i)] = name

    rows = [
        {
            "path": Path(p).relative_to(data_dir).as_posix(),
            "class_name": labels[i],
            "split": split_of[i],
        }
        for i, p in enumerate(paths)
    ]
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Splits salvos em {splits_path} ({len(rows)} imagens)")
    return rows


def read_splits(splits_path: Path | None = None) -> list[dict]:
    """Lê data/splits.csv, já sem as imagens de data/excluidas.csv."""
    excluded = read_excluded()
    with open(splits_path or config.SPLITS_PATH, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["path"] not in excluded]


def get_splits(data_dir: Path, regenerate: bool = False) -> list[dict]:
    """Lê data/splits.csv; cria se não existir (ou se regenerate=True)."""
    if regenerate or not config.SPLITS_PATH.exists():
        return create_splits(data_dir)
    return read_splits()


# ---------------------------------------------------------------------------
# tf.data
# ---------------------------------------------------------------------------

def _decode_image(path, label, img_size):
    img = tf.io.read_file(path)
    # expand_animations=False garante tensor 3D mesmo para .gif
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    # uint8 ocupa 4x menos no cache do que float32
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img, label


def _cache_name(split, paths, labels, img_size) -> str:
    """Nome do cache muda se mudar a lista de imagens, os rótulos ou o tamanho."""
    h = hashlib.md5()
    for p, y in zip(paths, labels):
        h.update(f"{p}|{y}\n".encode())
    return f"{split}_{img_size[0]}x{img_size[1]}_{h.hexdigest()[:10]}"


def _make_dataset(paths, labels, split, img_size, batch_size, shuffle, seed, cache):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _decode_image(p, y, img_size), num_parallel_calls=AUTOTUNE)
    # Ignora arquivos corrompidos em vez de quebrar o treino
    ds = ds.ignore_errors()

    if cache == "disk":
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(config.CACHE_DIR / _cache_name(split, paths, labels, img_size)))
    elif cache == "memory":
        ds = ds.cache()
    elif cache is not None:
        raise ValueError("cache deve ser 'disk', 'memory' ou None")

    # shuffle DEPOIS do cache, senão a ordem fica congelada entre épocas
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def load_datasets(
    data_dir: str | Path | None = None,
    img_size: tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    cache: str | None = "disk",
    regenerate_splits: bool = False,
    seed: int = config.SEED,
):
    """Retorna (train_ds, val_ds, test_ds, class_names).

    - data_dir: pasta do dataset; se None, baixa via kagglehub.
    - cache: 'disk' (padrão, em data/cache/), 'memory' ou None. A 1ª época
      grava o cache; as seguintes ficam bem mais rápidas.
    - Os splits vêm de data/splits.csv (iguais para todo o grupo).
    - Rótulos inteiros (use SparseCategoricalCrossentropy).
    - Imagens em float32 [0, 255]; normalize dentro do modelo
      (ex.: layers.Rescaling(1/255)) ou com o preprocess_input do backbone.
    """
    root = Path(data_dir) if data_dir else download_dataset()
    rows = get_splits(root, regenerate=regenerate_splits)

    labels = [r["class_name"] for r in rows]
    class_names = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths = np.array([str(root / r["path"]) for r in rows])
    y = np.array([class_to_idx[c] for c in labels], dtype=np.int32)
    split = np.array([r["split"] for r in rows])

    datasets = {}
    for name in SPLITS:
        idx = np.where(split == name)[0]
        datasets[name] = _make_dataset(
            paths[idx], y[idx], name, img_size, batch_size,
            shuffle=(name == "train"), seed=seed, cache=cache,
        )
        print(f"{name:5s}: {len(idx):6d} imagens")
    print(f"{len(class_names)} classes | cache={cache}")

    return datasets["train"], datasets["val"], datasets["test"], class_names


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("Classes:", class_names)
    images, labels = next(iter(train_ds))
    print("Batch:", images.shape, labels.shape, images.dtype)
