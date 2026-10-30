"""ЛР2, пункти 2-4: пошук порога, перевірки та тестова оцінка.

Працює лише зі збережених прогнозів (без CSV і без навчання):
    uv run python lab2_analysis.py --saved results

Правило: прогноз "шахрайство", якщо s >= t. Вартість: C(t) = 10 * FN + FP.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

FN_COST = 10
FIXED_T = 0.5
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def cost(fp, fn):
    return FN_COST * fn + fp


def confusion_direct(s, y, t):
    """Пряме застосування правила s >= t. Один повний прохід на кожен поріг,
    тому використовується лише в малих перевірках."""
    pred = s >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def candidate_table(s, y):
    """Усі унікальні оцінки та +inf, рядки за спаданням порога.

    Сортуємо один раз (np.unique), рахуємо позитивні й негативні в кожній групі
    однакових оцінок і беремо суфіксні суми: для t = u[k] прогноз 1 отримують
    усі групи від k і вище. Рівні оцінки лежать в одній групі, тому їм завжди
    дістається однаковий прогноз. Разом O(n log n).
    """
    u, inv = np.unique(s, return_inverse=True)
    inv = inv.ravel()
    pos = np.bincount(inv, weights=(y == 1), minlength=len(u)).astype(np.int64)
    neg = np.bincount(inv, weights=(y == 0), minlength=len(u)).astype(np.int64)

    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    n_pos, n_neg = int(pos.sum()), int(neg.sum())

    table = pd.DataFrame(
        {"threshold": u.astype(np.float64), "TP": tp, "FP": fp, "FN": n_pos - tp, "TN": n_neg - fp}
    )
    table = table.iloc[::-1]
    all_negative = pd.DataFrame(
        [{"threshold": np.inf, "TP": 0, "FP": 0, "FN": n_pos, "TN": n_neg}]
    )
    table = pd.concat([all_negative, table], ignore_index=True)
    table["C"] = FN_COST * table["FN"] + table["FP"]
    return table[COLUMNS]


def best_threshold(table):
    """Поріг з мінімальною вартістю. Таблиця йде за спаданням порога, а idxmin
    повертає перший мінімум, тому при рівній вартості береться більший поріг."""
    return float(table.loc[table["C"].idxmin(), "threshold"])


def brute_table(s, y):
    """Повільна незалежна версія для перевірки candidate_table."""
    rows = []
    for t in [np.inf] + sorted({float(v) for v in s}, reverse=True):
        tp, fp, fn, tn = confusion_direct(s, y, t)
        rows.append((t, tp, fp, fn, tn, cost(fp, fn)))
    return pd.DataFrame(rows, columns=COLUMNS)


def best_brute(table):
    lowest = table["C"].min()
    return float(table.loc[table["C"] == lowest, "threshold"].max())


SMALL_SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}


def run_checks():
    """Три малі набори з умови. Кидає помилку, якщо швидка й повільна версії розходяться."""
    results = {}
    for name, (scores, labels) in SMALL_SETS.items():
        s, y = np.array(scores), np.array(labels)
        fast = candidate_table(s, y)
        slow = brute_table(s, y)
        if not fast.equals(slow):
            raise RuntimeError(f"Set {name}: tables differ\n{fast}\n{slow}")
        t_fast, t_slow = best_threshold(fast), best_brute(slow)
        if t_fast != t_slow:
            raise RuntimeError(f"Set {name}: threshold {t_fast} != {t_slow}")
        results[name] = (fast, t_fast)
    return results


def table_row(table, t, label):
    r = table[table["threshold"] == t].iloc[0]
    row = {"rule": label, "threshold": float(r["threshold"])}
    for col in COLUMNS[1:]:
        row[col] = int(r[col])
    return row


def save_curves(out, y, s, auc):
    """PR- і ROC-криві на validation."""
    precision, recall, _ = precision_recall_curve(y, s)
    plt.figure(figsize=(5.5, 4.2))
    plt.plot(recall, precision, color="tab:blue")
    plt.xlabel("Повнота (recall)")
    plt.ylabel("Точність (precision)")
    plt.title("Крива точність-повнота, validation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y, s)
    plt.figure(figsize=(5.5, 4.2))
    plt.plot(fpr, tpr, color="tab:blue", label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="випадкове вгадування")
    plt.xlabel("Частка хибних тривог (FPR)")
    plt.ylabel("Частка виявлених шахрайств (TPR)")
    plt.title("ROC-крива, validation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150)
    plt.close()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_predictions(directory):
    path = directory / "predictions.npz"
    if not path.is_file():
        sys.exit(f"{path} not found. Run lab2_starter.py first, or pass the right --saved directory.")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main():
    # не падати на консолі, яка не вміє кодувати якийсь символ
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Threshold analysis from saved predictions.")
    parser.add_argument("--saved", type=Path, required=True, help="directory with predictions.npz")
    out = parser.parse_args().saved
    data = load_predictions(out)

    # Спочатку пункт 3: до тестової вибірки переходимо лише після успішних перевірок.
    print("== Checks on sets A, B, C: fast and brute force agree exactly ==")
    for name, (table, t_star) in run_checks().items():
        table.to_csv(out / f"check_{name}.csv", index=False)
        print(f"Set {name}, t* = {t_star}\n{table.to_string(index=False)}\n")

    # Пункт 2: validation
    y_val, s_val = data["validation_y"], data["validation_scores"]
    table = candidate_table(s_val, y_val)
    table.to_csv(out / "validation_candidates.csv", index=False)
    t_star = best_threshold(table)

    has_both_classes = len(np.unique(y_val)) == 2
    auc = float(roc_auc_score(y_val, s_val)) if has_both_classes else None  # з одним класом криві не визначені

    # 0.5 рідко буває точною оцінкою; найменший кандидат >= 0.5 дає ті самі
    # прогнози (+inf є завжди, тож результат ніколи не порожній)
    t_half = table.loc[table["threshold"] >= FIXED_T, "threshold"].min()
    rule_rows = [
        {**table_row(table, t_half, "default 0.5"), "threshold": FIXED_T},
        table_row(table, t_star, "min cost t*"),
        table_row(table, np.inf, "all negative"),
    ]
    if has_both_classes:
        save_curves(out, y_val, s_val, auc)
    print(f"== Validation (ROC-AUC = {auc}) ==\n{pd.DataFrame(rule_rows).to_string(index=False)}\n")
    write_json(out / "threshold.json", {"t_star": str(t_star), "val_roc_auc": auc})

    # Пункт 4: test із зафіксованим порогом
    t = float(json.loads((out / "threshold.json").read_text(encoding="utf-8"))["t_star"])
    y_test = data["test_y"]
    s_test = data["test_scores"].astype(np.float64)
    pred = s_test >= t
    tp = int(np.sum(pred & (y_test == 1)))
    fp = int(np.sum(pred & (y_test == 0)))
    fn = int(np.sum(~pred & (y_test == 1)))
    tn = int(np.sum(~pred & (y_test == 0)))

    # невизначена метрика (нульовий знаменник) записується як None
    metrics = {
        "t_star": str(t),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "C": cost(fp, fn),
        "precision": tp / (tp + fp) if tp + fp > 0 else None,
        "recall": tp / (tp + fn) if tp + fn > 0 else None,
    }
    write_json(out / "test_metrics.json", metrics)
    print(f"== Test (t* = {t}) ==\n{pd.DataFrame([metrics]).to_string(index=False)}\n")

    is_error = pred != (y_test == 1)
    errors = pd.DataFrame(
        {
            "csv_row": data["test_row_ids"],
            "error": np.where(y_test == 1, "FN", "FP"),
            "label": y_test,
            "score": s_test,
            "amount": data["test_amount"],
            "score_minus_t": s_test - t,
        }
    )[is_error].sort_values("csv_row")

    first = pd.concat([errors[errors["error"] == "FP"].head(5), errors[errors["error"] == "FN"].head(5)])
    first.to_csv(out / "test_errors_top5.csv", index=False)
    print(f"== First 5 FP and first 5 FN by CSV row ==\n{first.to_string(index=False)}")


if __name__ == "__main__":
    main()
