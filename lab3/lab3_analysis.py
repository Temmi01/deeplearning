"""ЛР3: аналіз журналів і тестова оцінка вибраної моделі на CPU."""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
from torchvision.models import resnet18

from lab3_train import (
    CLASSES, IMG_SIZE, Images, image_path, load_runs, macro_f1,
    make_loader, predict, read_split, select_run, sha256,
)


def learning_curves(runs, path):
    figure, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    colors = {"scratch": "#555555", "frozen": "#176b91", "unfreeze": "#b54f24"}
    styles = {1: "-", 2: "--", 3: "-."}
    for log in runs:
        regime, seed = log["regime"], log["seed"]
        epochs = [row["epoch"] for row in log["epochs"]]
        style = {"color": colors[regime], "linestyle": styles[seed], "marker": ".",
                 "label": f"{regime}, seed {seed}"}
        loss_ax.plot(epochs, [row["train_loss"] for row in log["epochs"]], **style)
        f1_ax.plot(epochs, [row["validation_f1"] for row in log["epochs"]], **style)
    for ax, ylabel in ((loss_ax, "Втрата на train"), (f1_ax, "Macro-F1 на validation")):
        ax.set(xlabel="Епоха", ylabel=ylabel, xticks=epochs)
        ax.grid(alpha=0.22)
    loss_ax.legend(fontsize=8)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_confusion(matrix, path):
    figure, ax = plt.subplots(figsize=(7.2, 6), layout="constrained")
    heatmap = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "#153044"
        ax.text(j, i, str(value), ha="center", va="center", color=color)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Передбачений клас", ylabel="Справжній клас")
    figure.colorbar(heatmap, ax=ax, shrink=0.78, label="Кількість зображень")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def select_errors(rows, labels, predictions, probabilities, matrix):
    pairs = [(i, j) for (i, j), count in np.ndenumerate(matrix) if i != j and count]
    pairs.sort(key=lambda pair: (-matrix[pair], *pair))
    errors = []
    # Перший файл кожної пари, потім другий і так далі, якщо пар менше п'яти.
    files_by_pair = [
        sorted(np.flatnonzero((labels == i) & (predictions == j)), key=lambda k: rows[k]["file"])
        for i, j in pairs
    ]
    for offset in range(max(map(len, files_by_pair), default=0)):
        for (i, j), files in zip(pairs, files_by_pair):
            if offset < len(files):
                k = files[offset]
                errors.append({"file": rows[k]["file"], "true": CLASSES[i],
                               "predicted": CLASSES[j], "probability": float(probabilities[k, j]),
                               "count": int(matrix[i, j])})
            if len(errors) == 5:
                return errors
    return errors


def plot_errors(root, errors, path):
    figure, axes = plt.subplots(1, 5, figsize=(12, 3.6), layout="constrained")
    figure.get_layout_engine().set(h_pad=0.12)
    for ax in axes:
        ax.axis("off")
    for ax, error in zip(axes, errors):
        row = {"file": error["file"], "label": error["true"], "part": "test"}
        with Image.open(image_path(root, row)) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f'{error["file"]}\n{error["true"]} → {error["predicted"]}\n'
                     f'p = {error["probability"]:.3f}', fontsize=10)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def evaluate_test(root, rows, checkpoint_path, winner, figures):
    if sha256(checkpoint_path) != winner["weights_sha256"]:
        raise ValueError("Файл ваг не збігається з журналом вибраного запуску.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["regime"], checkpoint["seed"], checkpoint["classes"], checkpoint["img_size"]) != (
        winner["regime"], winner["seed"], list(CLASSES), IMG_SIZE
    ):
        raise ValueError("Метадані моделі не відповідають правилу вибору.")
    model = resnet18(weights=None, num_classes=len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    test = Images(root, rows, "test")
    logits, labels = predict(model, make_loader(test), torch.device("cpu"))
    predictions = logits.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=range(len(CLASSES)))
    probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    errors = select_errors(test.rows, labels, predictions, probabilities, matrix)
    plot_confusion(matrix, figures / "confusion_matrix.png")
    plot_errors(root, errors, figures / "test_errors.png")
    accuracy = float(np.mean(labels == predictions))
    return {
        "macro_f1": macro_f1(labels, predictions),
        "accuracy": accuracy,
        "accuracy_standard_error": math.sqrt(accuracy * (1 - accuracy) / len(labels)),
        "n": len(labels), "confusion_matrix": matrix,
        "errors": errors,
    }


def print_results(runs, effects, winner, test):
    print("| # | Режим | Зерно | Втрата train (8) | Macro-F1 validation (8) | Час, с |")
    print("|---|---|---:|---:|---:|---:|")
    for i, log in enumerate(runs, start=1):
        final = log["epochs"][-1]
        print(f'| {i} | {log["regime"]} | {log["seed"]} | {final["train_loss"]:.6f} | '
              f'{final["validation_f1"]:.6f} | {log["seconds"]:.2f} |')
    print()
    for name, value in effects.items():
        print(f"{name} = {value:.6f}")
    winner_f1 = winner["epochs"][-1]["validation_f1"]
    print(f'\nВибрано {winner["regime"]}, seed {winner["seed"]}. '
          f'Validation macro-F1 = {winner_f1:.6f}.')
    print(f'Test macro-F1 = {test["macro_f1"]:.6f}, accuracy = {test["accuracy"]:.6f}, '
          f'SE(accuracy) = {test["accuracy_standard_error"]:.6f}, n = {test["n"]}.')
    print("\n| Справжній / передбачений | " + " | ".join(CLASSES) + " |")
    print("|---|" + "---:|" * len(CLASSES))
    for label, counts in zip(CLASSES, test["confusion_matrix"]):
        print(f"| {label} | " + " | ".join(map(str, counts)) + " |")
    print("\n| Файл | Справжній клас | Передбачений клас | Імовірність | Кількість |")
    print("|---|---|---|---:|---:|")
    for row in test["errors"]:
        print(f'| {row["file"]} | {row["true"]} | {row["predicted"]} | '
              f'{row["probability"]:.6f} | {row["count"]} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    runs = load_runs(args.out)
    rows = read_split(args.out / "split.csv")
    winner, means, scores = select_run(runs)
    r_frozen = max(scores["frozen"]) - min(scores["frozen"])
    r_unfreeze = max(scores["unfreeze"]) - min(scores["unfreeze"])
    delta_pre = means["frozen"] - means["scratch"]
    delta_unf = means["unfreeze"] - means["frozen"]
    effects = {
        "mean_frozen": means["frozen"], "range_frozen": r_frozen,
        "mean_unfreeze": means["unfreeze"], "range_unfreeze": r_unfreeze,
        "R": max(r_frozen, r_unfreeze), "delta_pre": delta_pre, "delta_unf": delta_unf,
        "delta_total": delta_pre + delta_unf,
    }
    figures = args.out / "results"
    figures.mkdir(exist_ok=True)
    learning_curves(runs, figures / "learning_curves.png")
    test = evaluate_test(args.data, rows, args.out / "lab3_model.pt", winner, figures)
    print_results(runs, effects, winner, test)


if __name__ == "__main__":
    main()
