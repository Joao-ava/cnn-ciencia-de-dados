"""Treinamento final da CNN e salvamento do melhor modelo.

Pega os melhores hiperparâmetros da busca (src/tuning.py), treina o modelo
por mais épocas e guarda em `models/`:
    - melhor_modelo.keras      : pesos da época com menor val_loss
    - historico_treino.json    : loss/accuracy por época (treino e validação)
    - historico_treino.csv     : o mesmo, gravado época a época (CSVLogger)
    - curvas_treino.png        : gráficos de loss e acurácia

Uso (no notebook):
    from src.train import train, plot_history

    model, history = train(hp_values=best_hp.values, epochs=40)
    plot_history(history)

Depois, para reutilizar o modelo salvo (aplicação, avaliação):
    import keras
    model = keras.models.load_model("models/melhor_modelo.keras")
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt

from src import config
from src.model import build_model

MODELS_DIR = config.PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "melhor_modelo.keras"
HISTORY_JSON_PATH = MODELS_DIR / "historico_treino.json"
HISTORY_CSV_PATH = MODELS_DIR / "historico_treino.csv"
CURVES_PATH = MODELS_DIR / "curvas_treino.png"

# Argumentos que build_model aceita como hiperparâmetros. O Hyperband também
# devolve chaves como "tuner/epochs" em best_hp.values; elas são descartadas.
_BUILD_ARGS = set(inspect.signature(build_model).parameters) - {"num_classes", "name"}


def default_callbacks(model_path: Path = BEST_MODEL_PATH, patience: int = 8) -> list:
    """Callbacks básicos. Troque pelos do grupo se houver `src/callbacks.py`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        # salva só quando val_loss melhora: no fim, o arquivo é o melhor modelo
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 3), verbose=1
        ),
        keras.callbacks.CSVLogger(str(HISTORY_CSV_PATH)),
    ]


def save_hyperparameters(hp_values: dict, path: Path = MODELS_DIR / "melhores_hiperparametros.json"):
    """Guarda os hiperparâmetros vencedores da busca (evita rodar o Hyperband de novo)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    limpos = {k: v for k, v in hp_values.items() if k in _BUILD_ARGS}
    Path(path).write_text(json.dumps(limpos, indent=2, ensure_ascii=False))
    return limpos


def load_hyperparameters(path: Path = MODELS_DIR / "melhores_hiperparametros.json") -> dict:
    return json.loads(Path(path).read_text())


def train(
    datasets=None,
    hp_values: dict | None = None,
    model: keras.Model | None = None,
    epochs: int = 40,
    patience: int = 8,
    callbacks: list | None = None,
    model_path: Path = BEST_MODEL_PATH,
    **load_kwargs,
):
    """Treina a CNN e salva o melhor modelo em `model_path`.

    - `datasets`: tupla (train_ds, val_ds, class_names). Se None, usa
      `load_preprocessed_datasets(**load_kwargs)`.
    - `hp_values`: hiperparâmetros (ex.: `best_hp.values` da busca). Se None,
      usa os valores default de `build_model`.
    - `model`: modelo já compilado (ex.: transfer learning). Se passado,
      `hp_values` é ignorado.
    - `callbacks`: lista própria; se None, usa `default_callbacks`.

    Retorna `(model, history)`, com o modelo já nos pesos da melhor época.
    """
    if datasets is None:
        from src.preprocessing import load_preprocessed_datasets

        train_ds, val_ds, _test_ds, class_names = load_preprocessed_datasets(**load_kwargs)
    else:
        train_ds, val_ds, class_names = datasets

    from src.preprocessing import compute_class_weights

    class_weight = compute_class_weights(class_names)

    if model is None:
        params = {k: v for k, v in (hp_values or {}).items() if k in _BUILD_ARGS}
        model = build_model(num_classes=len(class_names), **params)
    model.summary()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if callbacks is None:
        callbacks = default_callbacks(model_path, patience)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    HISTORY_JSON_PATH.write_text(json.dumps(history.history, indent=2))
    melhor = min(range(len(history.history["val_loss"])), key=history.history["val_loss"].__getitem__)
    print(
        f"\nMelhor época: {melhor + 1} | val_loss={history.history['val_loss'][melhor]:.4f} "
        f"| val_accuracy={history.history['val_accuracy'][melhor]:.4f}"
    )
    print(f"Modelo salvo em: {model_path}")
    return model, history


def plot_history(history, path: Path = CURVES_PATH, show: bool = True):
    """Curvas de loss e acurácia (treino x validação) e marca a melhor época."""
    h = history.history if hasattr(history, "history") else history
    epocas = range(1, len(h["loss"]) + 1)
    melhor = min(range(len(h["val_loss"])), key=h["val_loss"].__getitem__) + 1

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    for ax, chave, titulo in [(eixos[0], "loss", "Loss"), (eixos[1], "accuracy", "Acurácia")]:
        ax.plot(epocas, h[chave], label="Treino")
        ax.plot(epocas, h[f"val_{chave}"], label="Validação")
        ax.axvline(melhor, color="gray", ls="--", label=f"Melhor época ({melhor})")
        ax.set_xlabel("Época")
        ax.set_title(titulo)
        ax.legend()
    fig.tight_layout()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig
