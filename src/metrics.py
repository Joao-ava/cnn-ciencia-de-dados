"""Coleta de métricas da CNN: predições, F1 macro, métricas por classe e gráficos.

As classes são desbalanceadas (até 14,6x), então a acurácia sozinha engana: um
modelo que erra todas as classes pequenas ainda passa de 90%. Por isso a
avaliação usa F1 macro (média simples do F1 de cada classe, todas pesando
igual) e o `classification_report` por classe.

Uso:
    from src.metrics import evaluate_model, plot_history

    metrics = evaluate_model(model, test_ds, class_names, output_dir="results/final")
    print(metrics["f1_macro"])
    plot_history(history, path="results/final/history.png")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def predict_dataset(model, ds) -> tuple[np.ndarray, np.ndarray]:
    """Roda o modelo num tf.data.Dataset e devolve (y_true, y_prob).

    Rótulos e predições saem da MESMA passada pelos batches. Não use
    `model.predict(ds)` e leia os rótulos em outra passada: o dataset usa
    `ignore_errors()`, e se um arquivo falhar as duas listas ficam desalinhadas.
    """
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(np.asarray(model.predict_on_batch(x)))
        y_true.append(np.asarray(y))
    return np.concatenate(y_true), np.concatenate(y_prob)


def macro_f1(y_true, y_pred) -> float:
    """F1 macro: média simples do F1 de cada classe (classes pequenas pesam igual)."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    """Métricas gerais + por classe a partir dos rótulos e das classes previstas.

    `per_class` é o `classification_report` do sklearn em dict:
    {classe: {"precision", "recall", "f1-score", "support"}}.
    Classes sem nenhuma imagem no conjunto aparecem com support 0 (não quebram).
    """
    labels = list(range(len(class_names)))
    avg = {"labels": labels, "zero_division": 0}
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_amostras": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **avg)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **avg)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", **avg)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **avg)),
        "per_class": {c: report[c] for c in class_names},
    }


def evaluate_model(
    model,
    ds,
    class_names: list[str],
    output_dir: str | Path | None = None,
    prefix: str = "test",
    verbose: bool = True,
) -> dict:
    """Avalia o modelo num dataset (normalmente o `test_ds`) e devolve as métricas.

    Com `output_dir`, salva:
      - `{prefix}_metrics.json`: métricas gerais + por classe
      - `{prefix}_per_class.csv`: precisão, recall, F1 e suporte de cada classe
      - `{prefix}_confusion_matrix.png`: matriz de confusão normalizada por linha
    """
    y_true, y_prob = predict_dataset(model, ds)
    y_pred = y_prob.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred, class_names)

    if verbose:
        labels = list(range(len(class_names)))
        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=class_names,
                digits=3,
                zero_division=0,
            )
        )
        print(
            f"F1 macro: {metrics['f1_macro']:.4f} | acurácia: {metrics['accuracy']:.4f} | "
            f"acurácia balanceada: {metrics['balanced_accuracy']:.4f}"
        )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        _save_per_class_csv(metrics["per_class"], out / f"{prefix}_per_class.csv")
        plot_confusion_matrix(
            y_true, y_pred, class_names, path=out / f"{prefix}_confusion_matrix.png"
        )
        if verbose:
            print(f"Métricas salvas em {out}/")

    return metrics


def _save_per_class_csv(per_class: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["classe", "precision", "recall", "f1", "support"])
        for nome, m in per_class.items():
            writer.writerow(
                [
                    nome,
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1-score']:.4f}",
                    int(m["support"]),
                ]
            )


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    normalize: str | None = "true",
    path: str | Path | None = None,
):
    """Matriz de confusão (linhas = classe real, colunas = classe prevista).

    `normalize="true"` divide cada linha pelo total da classe: a diagonal vira o
    recall de cada classe e dá para comparar classes grandes e pequenas.
    `normalize=None` mostra as contagens.
    """
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)), normalize=normalize)

    size = max(6, 0.45 * n)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n), class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(n), class_names, fontsize=8)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(
        "Matriz de confusão" + (" (normalizada por classe real)" if normalize else "")
    )

    limiar = cm.max() / 2 if cm.size else 0
    for i in range(n):
        for j in range(n):
            valor = cm[i, j]
            if valor == 0:
                continue
            texto = f"{valor:.2f}" if normalize else str(int(valor))
            ax.text(
                j,
                i,
                texto,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if valor > limiar else "black",
            )

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_history(history, path: str | Path | None = None):
    """Curvas de treino x validação por época (loss, acurácia, F1 macro e LR).

    Aceita o `History` do `model.fit`, um dict {métrica: [valores]} ou o
    caminho do `history.csv` salvo pelo `CSVLogger` de `build_callbacks`.
    """
    hist = _history_to_dict(history)
    paineis = [
        ("Loss", ["loss", "val_loss"]),
        ("Acurácia", ["accuracy", "val_accuracy"]),
        ("F1 macro (validação)", ["val_f1_macro"]),
        ("Learning rate", ["learning_rate"]),
    ]
    paineis = [(t, [k for k in ks if k in hist]) for t, ks in paineis]
    paineis = [(t, ks) for t, ks in paineis if ks]
    if not paineis:
        raise ValueError(f"Nenhuma métrica conhecida no histórico: {sorted(hist)}")

    fig, axes = plt.subplots(
        1, len(paineis), figsize=(4.5 * len(paineis), 3.8), squeeze=False
    )
    for ax, (titulo, chaves) in zip(axes[0], paineis):
        for k in chaves:
            valores = hist[k]
            ax.plot(
                range(1, len(valores) + 1),
                valores,
                marker="o",
                markersize=3,
                label="validação" if k.startswith("val_") else "treino",
            )
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        if titulo == "Learning rate":
            ax.set_yscale("log")
        if len(chaves) > 1:
            ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def _history_to_dict(history) -> dict[str, list[float]]:
    if hasattr(history, "history"):
        return dict(history.history)
    if isinstance(history, (str, Path)):
        with open(history, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return (
            {k: [float(r[k]) for r in rows] for k in rows[0] if k != "epoch"}
            if rows
            else {}
        )
    return dict(history)
