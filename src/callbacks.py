"""Callbacks do treino final da CNN.

`build_callbacks` monta a lista padrão: F1 macro na validação a cada época,
checkpoint do melhor modelo, early stopping, redução da taxa de aprendizado,
histórico em CSV e (opcional) TensorBoard. Todos monitoram a mesma métrica
(`val_f1_macro` por padrão).

Uso:
    from src.callbacks import build_callbacks
    from src.preprocessing import compute_class_weights

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=50,
        class_weight=compute_class_weights(class_names),
        callbacks=build_callbacks(val_ds, run_name="final"),
    )

Por que F1 macro e não val_loss/val_accuracy: as classes são desbalanceadas e o
`class_weight` só vale no treino, então a loss e a acurácia de validação são
dominadas pelas classes grandes. O F1 macro dá o mesmo peso a todas as classes
e é a métrica usada na avaliação final (ver `src/metrics.py`).
"""

from __future__ import annotations

from pathlib import Path

import keras

from src import config
from src.metrics import macro_f1, predict_dataset


class MacroF1Callback(keras.callbacks.Callback):
    """Calcula o F1 macro na validação ao fim de cada época e grava em `logs`.

    O Keras não calcula F1 com rótulos inteiros (`keras.metrics.F1Score` exige
    one-hot), por isso o cálculo é feito aqui com o sklearn.

    Precisa vir ANTES dos callbacks que monitoram a métrica (checkpoint, early
    stopping, ReduceLROnPlateau, CSVLogger): todos recebem o mesmo dict de
    `logs`, na ordem da lista. `build_callbacks` já coloca na ordem certa.
    O `History` e a barra de progresso do Keras vêm depois, então
    `val_f1_macro` aparece em `history.history` e no log de cada época.
    """

    def __init__(self, val_ds, name: str = "val_f1_macro"):
        super().__init__()
        self.val_ds = val_ds
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_prob = predict_dataset(self.model, self.val_ds)
        f1 = macro_f1(y_true, y_prob.argmax(axis=1))
        if logs is not None:
            logs[self.name] = f1


def _mode_for(monitor: str) -> str:
    """'min' para métricas de erro (loss), 'max' para as demais (F1, acurácia)."""
    return "min" if "loss" in monitor else "max"


def build_callbacks(
    val_ds=None,
    run_name: str = "final",
    monitor: str = "val_f1_macro",
    patience: int = 8,
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    tensorboard: bool = True,
    models_dir: str | Path = config.MODELS_DIR,
    results_dir: str | Path = config.RESULTS_DIR,
) -> list[keras.callbacks.Callback]:
    """Lista de callbacks para o `model.fit` do treino final.

    - `val_ds`: dataset de validação. Obrigatório quando `monitor="val_f1_macro"`.
    - `run_name`: nome do treino. Os arquivos vão para `models/{run_name}/` e
      `results/{run_name}/`.
    - `monitor`: métrica que decide o melhor modelo, a parada e a redução de LR
      (`"val_f1_macro"`, `"val_loss"` ou `"val_accuracy"`).
    - `patience`: épocas sem melhora até parar (early stopping).
    - `lr_patience`, `lr_factor`, `min_lr`: épocas sem melhora até multiplicar o
      LR por `lr_factor`, sem descer de `min_lr`. Mantenha `lr_patience` menor
      que `patience`, para o LR cair algumas vezes antes de o treino parar.
    - `tensorboard`: grava logs em `results/{run_name}/tensorboard`
      (`tensorboard --logdir results`).
    """
    if lr_patience >= patience:
        raise ValueError(
            "lr_patience deve ser menor que patience, senão o treino para "
            "antes de a taxa de aprendizado ser reduzida."
        )

    mode = _mode_for(monitor)
    model_dir = Path(models_dir) / run_name
    result_dir = Path(results_dir) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[keras.callbacks.Callback] = []
    if monitor == "val_f1_macro":
        if val_ds is None:
            raise ValueError("monitor='val_f1_macro' precisa do val_ds.")
        callbacks.append(MacroF1Callback(val_ds))

    callbacks += [
        keras.callbacks.ModelCheckpoint(
            str(model_dir / "best.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(result_dir / "history.csv")),
    ]
    if tensorboard:
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir=str(result_dir / "tensorboard"))
        )
    return callbacks
