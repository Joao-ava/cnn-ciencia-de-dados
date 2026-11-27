"""Construção da CNN (do zero) para classificação saudável x podre.

A arquitetura é um empilhado de blocos convolucionais (Conv2D + BatchNorm
opcional + MaxPooling + Dropout opcional) seguido por camadas densas. Os
hiperparâmetros da arquitetura e do treino ficam parametrizados para que
`src/tuning.py` possa buscá-los com o Keras Tuner.

Uso direto (sem busca, com os valores default):
    from src.model import build_model
    from src import config

    model = build_model(num_classes=28)
    model.summary()

Uso com o Keras Tuner (feito em src/tuning.py):
    from src.model import make_build_model

    build_fn = make_build_model(num_classes=28)
    model = build_fn(hp)   # hp: keras_tuner.HyperParameters
"""

from __future__ import annotations

from typing import Callable

import keras
from keras import layers

from src import config

# Espaço de busca dos hiperparâmetros (usado por make_build_model e documentado
# aqui para ficar junto da arquitetura que eles controlam).
#   - conv_blocks:        quantos blocos convolucionais (cada um termina em MaxPooling)
#   - filters_inicial:    nº de filtros do 1º bloco; dobra (ou não) a cada bloco
#   - filters_growth:     1 (mesmo nº em todos os blocos) ou 2 (dobra a cada bloco)
#   - kernel_size:        tamanho do kernel das convoluções
#   - conv_por_bloco:     quantas camadas Conv2D em sequência antes do pooling
#   - batchnorm:          liga/desliga BatchNormalization após cada Conv2D
#   - dropout_conv:       dropout após o pooling de cada bloco (0 desliga)
#   - dense_units:        nº de neurônios da camada densa antes da saída
#   - dropout_dense:      dropout antes da camada de saída (0 desliga)
#   - learning_rate:      taxa de aprendizado do otimizador (escala log)
#   - optimizer:          "adam" ou "rmsprop"
HYPERPARAMETER_SPACE = {
    "conv_blocks": {"min": 2, "max": 5, "default": 3},
    "filters_inicial": {"values": [16, 32, 64], "default": 32},
    "filters_growth": {"values": [1, 2], "default": 2},
    "kernel_size": {"values": [3, 5], "default": 3},
    "conv_por_bloco": {"values": [1, 2], "default": 1},
    "batchnorm": {"default": True},
    "dropout_conv": {"min": 0.0, "max": 0.3, "step": 0.1, "default": 0.0},
    "dense_units": {"values": [128, 256, 512], "default": 256},
    "dropout_dense": {"min": 0.2, "max": 0.5, "step": 0.1, "default": 0.3},
    "learning_rate": {"min": 1e-4, "max": 1e-2, "sampling": "log", "default": 1e-3},
    "optimizer": {"values": ["adam", "rmsprop"], "default": "adam"},
}


def _build_optimizer(name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    if name == "adam":
        return keras.optimizers.Adam(learning_rate)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate)
    raise ValueError(f"optimizer deve ser 'adam' ou 'rmsprop', recebi {name!r}")


def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (*config.IMG_SIZE, 3),
    conv_blocks: int = 3,
    filters_inicial: int = 32,
    filters_growth: int = 2,
    kernel_size: int = 3,
    conv_por_bloco: int = 1,
    batchnorm: bool = True,
    dropout_conv: float = 0.0,
    dense_units: int = 256,
    dropout_dense: float = 0.3,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    name: str = "cnn_frutas",
) -> keras.Model:
    """Monta e compila a CNN com os hiperparâmetros passados (ou os default).

    Espera imagens já normalizadas (ex.: `load_preprocessed_datasets`, ver
    `src/preprocessing.py`) — este módulo cuida só da arquitetura, não do
    pré-processamento. Rótulos inteiros: a perda é
    `sparse_categorical_crossentropy`.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    filters = filters_inicial
    for _ in range(conv_blocks):
        for _ in range(conv_por_bloco):
            x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
            if batchnorm:
                x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        if dropout_conv:
            x = layers.Dropout(dropout_conv)(x)
        filters *= filters_growth

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    if dropout_dense:
        x = layers.Dropout(dropout_dense)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=_build_optimizer(optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_build_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (*config.IMG_SIZE, 3),
) -> Callable[["keras_tuner.HyperParameters"], keras.Model]:  # noqa: F821
    """Fábrica da função `build_model(hp)` que o Keras Tuner espera.

    `num_classes` e `input_shape` ficam fixos (vêm dos dados, não fazem parte
    da busca); os outros hiperparâmetros são sorteados a partir de
    `HYPERPARAMETER_SPACE` a cada trial.
    """

    def build(hp) -> keras.Model:
        s = HYPERPARAMETER_SPACE
        return build_model(
            num_classes=num_classes,
            input_shape=input_shape,
            conv_blocks=hp.Int("conv_blocks", s["conv_blocks"]["min"], s["conv_blocks"]["max"],
                                default=s["conv_blocks"]["default"]),
            filters_inicial=hp.Choice("filters_inicial", s["filters_inicial"]["values"],
                                       default=s["filters_inicial"]["default"]),
            filters_growth=hp.Choice("filters_growth", s["filters_growth"]["values"],
                                      default=s["filters_growth"]["default"]),
            kernel_size=hp.Choice("kernel_size", s["kernel_size"]["values"],
                                   default=s["kernel_size"]["default"]),
            conv_por_bloco=hp.Choice("conv_por_bloco", s["conv_por_bloco"]["values"],
                                      default=s["conv_por_bloco"]["default"]),
            batchnorm=hp.Boolean("batchnorm", default=s["batchnorm"]["default"]),
            dropout_conv=hp.Float("dropout_conv", s["dropout_conv"]["min"], s["dropout_conv"]["max"],
                                   step=s["dropout_conv"]["step"], default=s["dropout_conv"]["default"]),
            dense_units=hp.Choice("dense_units", s["dense_units"]["values"],
                                   default=s["dense_units"]["default"]),
            dropout_dense=hp.Float("dropout_dense", s["dropout_dense"]["min"], s["dropout_dense"]["max"],
                                    step=s["dropout_dense"]["step"], default=s["dropout_dense"]["default"]),
            learning_rate=hp.Float("learning_rate", s["learning_rate"]["min"], s["learning_rate"]["max"],
                                    sampling=s["learning_rate"]["sampling"],
                                    default=s["learning_rate"]["default"]),
            optimizer=hp.Choice("optimizer", s["optimizer"]["values"], default=s["optimizer"]["default"]),
        )

    return build


if __name__ == "__main__":
    model = build_model(num_classes=28)
    model.summary()
