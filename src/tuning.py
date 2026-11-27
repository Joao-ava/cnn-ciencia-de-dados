"""Busca de hiperparâmetros da CNN (src/model.py) com o Keras Tuner.

A arquitetura e o espaço de busca ficam em `src/model.py`
(`HYPERPARAMETER_SPACE` e `make_build_model`); aqui só montamos o `Tuner` e
rodamos a busca sobre os dados já pré-processados (`src/preprocessing.py`).

Uso rápido:
    from src.tuning import run_search

    tuner, best_hp, best_model = run_search(max_epochs=15, max_trials=20)

Ou direto do terminal (usa os parâmetros default abaixo):
    python -m src.tuning
"""

from __future__ import annotations

import keras
import keras_tuner as kt

from src import config
from src.model import make_build_model
from src.preprocessing import compute_class_weights, load_preprocessed_datasets


def run_search(
    max_epochs: int = 15,
    max_trials: int = 20,
    factor: int = 3,
    patience: int = 3,
    project_name: str = "cnn_frutas",
    overwrite: bool = False,
    **load_kwargs,
):
    """Roda a busca de hiperparâmetros com Hyperband e devolve o melhor modelo.

    - `max_epochs`: nº máximo de épocas de um trial (o Hyperband descarta
      trials ruins bem antes disso, então o custo real é bem menor).
    - `max_trials` (`hyperband_iterations` no Keras Tuner): quantas rodadas
      completas de Hyperband rodar; aumente se quiser explorar mais o espaço.
    - `factor`: fração de trials que sobrevive a cada rodada de eliminação.
    - `patience`: paciência do early stopping (monitora `val_loss`).
    - `overwrite`: True descarta trials salvos de uma busca anterior com o
      mesmo `project_name` e começa do zero.
    - `load_kwargs`: repassados a `load_preprocessed_datasets` (ex.:
      `batch_size`, `normalization`, `augmentation`).

    Resultados intermediários ficam em `config.TUNING_DIR` (fora do git), o
    que permite retomar uma busca interrompida (com `overwrite=False`).
    """
    train_ds, val_ds, _test_ds, class_names = load_preprocessed_datasets(**load_kwargs)
    class_weight = compute_class_weights(class_names)
    build_fn = make_build_model(num_classes=len(class_names))

    tuner = kt.Hyperband(
        build_fn,
        objective="val_accuracy",
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=max_trials,
        directory=str(config.TUNING_DIR),
        project_name=project_name,
        overwrite=overwrite,
        seed=config.SEED,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[early_stop],
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    print("\nMelhores hiperparâmetros encontrados:")
    for nome, valor in best_hp.values.items():
        print(f"  {nome}: {valor}")

    return tuner, best_hp, best_model


if __name__ == "__main__":
    run_search()
