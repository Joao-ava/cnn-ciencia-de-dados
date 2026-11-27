"""Pré-processamento das imagens para a CNN.

O `data_loader` já entrega imagens float32 em [0, 255], 224x224. Aqui fica o
que vem depois disso:

1. **Data augmentation**: só no treino. Como são camadas Keras, ficam
   desligadas automaticamente em `model.evaluate` / `model.predict`.
2. **Normalização**: leva os pixels para a escala que o modelo espera.
3. **Pesos das classes**: compensam o desbalanceamento (até 14,6x).

Há duas formas de usar (escolha UMA, nunca as duas juntas):

A) No tf.data.Dataset (pré-processamento roda na CPU, em paralelo com a GPU):
    from src.preprocessing import load_preprocessed_datasets, compute_class_weights

    train_ds, val_ds, test_ds, class_names = load_preprocessed_datasets(normalization="0_1")
    model.fit(train_ds, validation_data=val_ds,
              class_weight=compute_class_weights(class_names))

B) Como camadas dentro do modelo (o modelo salvo já recebe imagens 0-255):
    from src.data_loader import load_datasets
    from src.preprocessing import build_preprocessing, compute_class_weights

    train_ds, val_ds, test_ds, class_names = load_datasets()
    class_weight = compute_class_weights(class_names)

    inputs = keras.Input(shape=(*config.IMG_SIZE, 3))
    x = build_preprocessing(normalization="0_1")(inputs)
    ...  # resto da CNN
    model.fit(train_ds, validation_data=val_ds, class_weight=class_weight)

Tudo fica dentro do modelo, então o modelo salvo já recebe imagens "cruas"
(0-255) e quem usar depois não precisa repetir o pré-processamento.
"""

from __future__ import annotations

from collections import Counter

import keras
import tensorflow as tf
from keras import layers

from src import config
from src.data_loader import AUTOTUNE, load_datasets, read_splits

# Escalas de normalização. A escolha depende do modelo:
# - "0_1":    CNN do zero          -> pixels / 255
# - "-1_1":   MobileNetV2/V3, Xception, Inception -> (pixels / 127.5) - 1
# - "imagenet": ResNet50/VGG estilo "torch" -> (pixels/255 - média) / desvio
# - None:     EfficientNet/ConvNeXt (keras.applications): já normalizam
#             por dentro e esperam [0, 255]
NORMALIZATIONS = ("0_1", "-1_1", "imagenet", None)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_augmentation(
    flip: str = "horizontal_and_vertical",
    rotation: float = 0.1,
    zoom: float = 0.1,
    translation: float = 0.1,
    contrast: float = 0.1,
    brightness: float = 0.1,
    seed: int = config.SEED,
) -> keras.Sequential:
    """Camadas de augmentation, ativas apenas no treino.

    Os valores são frações. Ex.: rotation=0.1 é ±10% de 360°, ou seja ±36°.
    Use 0 para desligar uma transformação.

    Não há corte de cor (hue/saturação) de propósito: a cor é justamente
    o que separa fruta saudável de podre.
    """
    aug = []
    if flip:
        aug.append(layers.RandomFlip(flip, seed=seed))
    if rotation:
        aug.append(layers.RandomRotation(rotation, fill_mode="reflect", seed=seed))
    if zoom:
        aug.append(layers.RandomZoom(zoom, fill_mode="reflect", seed=seed))
    if translation:
        aug.append(layers.RandomTranslation(translation, translation, fill_mode="reflect", seed=seed))
    if contrast:
        aug.append(layers.RandomContrast(contrast, value_range=(0, 255), seed=seed))
    if brightness:
        # value_range=(0, 255): a augmentation roda ANTES da normalização
        aug.append(layers.RandomBrightness(brightness, value_range=(0, 255), seed=seed))
    return keras.Sequential(aug, name="augmentation")


def build_normalization(normalization: str | None = "0_1") -> layers.Layer:
    """Camada que normaliza imagens [0, 255] para a escala do modelo."""
    if normalization == "0_1":
        return layers.Rescaling(1.0 / 255, name="normalization")
    if normalization == "-1_1":
        return layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalization")
    if normalization == "imagenet":
        # Normalization faz (x - mean) / sqrt(variance); a escala 0-255 vai na média/variância
        return layers.Normalization(
            mean=[m * 255 for m in IMAGENET_MEAN],
            variance=[(s * 255) ** 2 for s in IMAGENET_STD],
            name="normalization",
        )
    if normalization is None:
        return layers.Identity(name="normalization")
    raise ValueError(f"normalization deve ser um de {NORMALIZATIONS}")


def build_preprocessing(
    normalization: str | None = "0_1",
    augment: bool = True,
    **augmentation_kwargs,
) -> keras.Sequential:
    """Bloco completo (augmentation + normalização) para colocar no início do modelo.

    Ex.:
        inputs = keras.Input(shape=(224, 224, 3))
        x = build_preprocessing(normalization=None)(inputs)   # EfficientNet
        x = keras.applications.EfficientNetB0(include_top=False, pooling="avg")(x)
    """
    blocks = []
    if augment:
        blocks.append(build_augmentation(**augmentation_kwargs))
    blocks.append(build_normalization(normalization))
    return keras.Sequential(blocks, name="preprocessing")


def preprocess_dataset(
    ds: tf.data.Dataset,
    normalization: str | None = "0_1",
    augment: bool = False,
    **augmentation_kwargs,
) -> tf.data.Dataset:
    """Aplica augmentation (opcional) e normalização num tf.data.Dataset já em batches.

    Roda na CPU, em paralelo com o treino na GPU (num_parallel_calls + prefetch).
    A augmentation é sorteada de novo a cada época.
    """
    augmentation = build_augmentation(**augmentation_kwargs) if augment else None
    norm = build_normalization(normalization)

    def _apply(x, y):
        if augmentation is not None:
            x = augmentation(x, training=True)
        return norm(x), y

    return ds.map(_apply, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def load_preprocessed_datasets(
    normalization: str | None = "0_1",
    augment: bool = True,
    augmentation: dict | None = None,
    **load_kwargs,
):
    """Carrega os dados e já devolve os tf.data.Dataset pré-processados.

    Retorna (train_ds, val_ds, test_ds, class_names), como `load_datasets`:
    - train_ds: augmentation (se augment=True) + normalização
    - val_ds / test_ds: só normalização (nunca augmentation)

    - augmentation: argumentos de `build_augmentation`, ex.: {"rotation": 0.05, "zoom": 0}
    - load_kwargs: repassados ao `load_datasets` (data_dir, batch_size, cache, ...)

    Ex.:
        train_ds, val_ds, test_ds, class_names = load_preprocessed_datasets()
        model.fit(train_ds, validation_data=val_ds,
                  class_weight=compute_class_weights(class_names))
    """
    train_ds, val_ds, test_ds, class_names = load_datasets(**load_kwargs)
    train_ds = preprocess_dataset(train_ds, normalization, augment, **(augmentation or {}))
    val_ds = preprocess_dataset(val_ds, normalization)
    test_ds = preprocess_dataset(test_ds, normalization)
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(class_names: list[str], splits_path=None) -> dict[int, float]:
    """Pesos por classe para `model.fit(class_weight=...)`.

    Fórmula "balanced" (igual à do scikit-learn): n_total / (n_classes * n_classe).
    Classes pequenas pesam mais; com classes balanceadas, todos os pesos dariam 1.
    Conta só as imagens de treino de data/splits.csv, sem abrir nenhuma imagem.

    `class_names` deve ser o retornado por `load_datasets` (mesma ordem dos rótulos).
    """
    counts = Counter(r["class_name"] for r in read_splits(splits_path) if r["split"] == "train")
    missing = set(class_names) - set(counts)
    if missing:
        raise ValueError(f"Classes sem imagens de treino: {sorted(missing)}")
    total = sum(counts[c] for c in class_names)
    return {i: total / (len(class_names) * counts[c]) for i, c in enumerate(class_names)}


def save_augmentation_preview(dataset, path="augmentation_preview.png", n_images=4, n_variations=5):
    """Salva uma grade com algumas imagens e variações da augmentation.

    Serve para conferir que as transformações não estão fortes demais.
    """
    import matplotlib.pyplot as plt  # só salva arquivo; não troca o backend (quebraria o notebook)

    augmentation = build_augmentation()
    images, _ = next(iter(dataset))
    images = images[:n_images]

    fig, axes = plt.subplots(n_images, n_variations + 1, figsize=(2 * (n_variations + 1), 2 * n_images))
    for i in range(n_images):
        axes[i, 0].imshow(images[i].numpy().astype("uint8"))
        axes[i, 0].set_title("original" if i == 0 else "")
        for j in range(1, n_variations + 1):
            aug = augmentation(images[i : i + 1], training=True)[0]
            axes[i, j].imshow(keras.ops.convert_to_numpy(aug).clip(0, 255).astype("uint8"))
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Prévia salva em {path}")


if __name__ == "__main__":
    from src.data_loader import load_datasets

    train_ds, _, _, class_names = load_datasets()

    weights = compute_class_weights(class_names)
    print("\nPesos das classes (treino):")
    for i, c in enumerate(class_names):
        print(f"  {c:28s} {weights[i]:.2f}")

    save_augmentation_preview(train_ds)
