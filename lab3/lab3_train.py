"""ЛР3: перевірки, сім запусків на validation і збереження вибраної моделі."""

import argparse
import csv
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
PARTS = ("train", "validation", "test")
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EPOCHS = 8
RUNS = [
    ("scratch", 1),
    ("frozen", 1), ("frozen", 2), ("frozen", 3),
    ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3),
]
BACKBONE = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
EVAL_MODULES = {"scratch": (), "frozen": BACKBONE, "unfreeze": BACKBONE[:-1]}
GROUPS = {
    "scratch": [("all", 3e-3, 11_179_590)],
    "frozen": [("fc", 3e-3, 3_078)],
    "unfreeze": [("fc", 3e-3, 3_078), ("layer4", 3e-4, 8_393_728)],
}
CONTROL = [
    [(200, 1996761), (218, 4837323), (219, 4837559)],
    [(200, 2032625), (237, 5276962), (237, 5291326)],
    [(200, 2183923), (276, 6135652), (277, 6157250)],
    [(200, 1950345), (262, 5789535), (263, 5849765)],
    [(200, 1987613), (255, 5655068), (255, 5668011)],
    [(200, 1841817), (250, 5541845), (251, 5547533)],
]
BN_PREDICTION = (
    "У frozen з model.train() параметри кістяка залишаться сталими, "
    "але running_mean, running_var і лічильники BN зміняться. "
    "Втрата на train теж може спадати, тому сама крива не виявить помилку."
)


def image_path(root, row):
    folder = "seg_train" if row["part"] == "train" else "seg_test"
    return root / folder / folder / row["label"] / row["file"]


def make_split(root, path):
    rows = []
    for folder in ("seg_train", "seg_test"):
        rng = np.random.default_rng(0)
        for label in CLASSES:
            files = sorted(p.name for p in (root / folder / folder / label).glob("*.jpg"))
            indices = np.arange(len(files))
            rng.shuffle(indices)
            if folder == "seg_train":
                pieces = [("train", indices[:200])]
            else:
                cut = len(files) // 2
                pieces = [("validation", indices[:cut]), ("test", indices[cut:])]
            for part, selected in pieces:
                rows.extend({"file": files[i], "label": label, "part": part} for i in selected)

    print("Контроль поділу (кількість / сума імен):")
    check_split(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "label", "part"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_split(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_split(rows):
    names = {row["file"] for row in rows}
    if len(names) != len(rows) or len(rows) != 4200:
        raise ValueError("У поділі є дублікати або загальна кількість не дорівнює 4200.")
    print(f"{'Клас':12}" + "".join(f"{part:>22}" for part in PARTS))
    for label, expected_row in zip(CLASSES, CONTROL):
        cells = []
        for part, expected in zip(PARTS, expected_row):
            files = [r["file"] for r in rows if r["label"] == label and r["part"] == part]
            actual = (len(files), sum(int(Path(name).stem) for name in files))
            if actual != expected:
                raise ValueError(f"Поділ {label}/{part}: {actual}, очікується {expected}.")
            cells.append(f"{actual[0]} / {actual[1]}")
        print(f"{label:12}" + "".join(f"{cell:>22}" for cell in cells))


class Images(Dataset):
    def __init__(self, root, rows, part):
        self.root = root
        self.rows = [row for row in rows if row["part"] == part]
        if part == "train":
            spatial = [transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                       transforms.RandomHorizontalFlip()]
        else:
            spatial = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        self.transform = transforms.Compose([
            *spatial, transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        # Validation has the same pixels each epoch and is shared by all seven runs.
        self.cache = {} if part == "validation" else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.cache is not None and index in self.cache:
            return self.cache[index]
        row = self.rows[index]
        with Image.open(image_path(self.root, row)) as image:
            sample = self.transform(image.convert("RGB")), CLASSES.index(row["label"])
        if self.cache is not None:
            self.cache[index] = sample
        return sample


def make_loader(dataset, train=False):
    return DataLoader(dataset, batch_size=32 if train else 64, shuffle=train,
                      drop_last=False, num_workers=0)


def setup_run(regime, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, len(CLASSES))
    for name in EVAL_MODULES[regime]:
        getattr(model, name).requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    groups = []
    for name, lr, _ in GROUPS[regime]:
        module = model if name == "all" else getattr(model, name)
        groups.append({"params": list(module.parameters()), "lr": lr, "name": name})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    return model, optimizer


def set_train_mode(model, regime):
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def train_batch(model, optimizer, loss_fn, images, labels, device):
    optimizer.zero_grad()
    loss = loss_fn(model(images.to(device)), labels.to(device))
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, targets = [], []
    for x, y in loader:
        scores.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(scores), np.concatenate(targets)


def macro_f1(labels, predictions):
    return float(f1_score(labels, predictions, labels=list(range(len(CLASSES))),
                          average="macro", zero_division=0))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(out, regime, seed):
    return out / "runs" / f"{regime}_seed{seed}"


def load_runs(out):
    current_hash = sha256(out / "split.csv")
    runs = []
    for regime, seed in RUNS:
        folder = run_dir(out, regime, seed)
        log = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if log["split_sha256"] != current_hash:
            raise ValueError(f"Хеш split.csv не збігається із журналом {folder.name}.")
        if (log["regime"], log["seed"]) != (regime, seed) or len(log["epochs"]) != EPOCHS:
            raise ValueError(f"Некоректний журнал {folder.name}.")
        runs.append(log)
    return runs


def select_run(runs):
    final_f1 = {regime: [] for regime in GROUPS}
    for log in runs:
        final_f1[log["regime"]].append(log["epochs"][-1]["validation_f1"])
    means = {regime: float(np.mean(values)) for regime, values in final_f1.items()}
    # За точної рівності зберігається порядок режимів і зерен у RUNS.
    regime = max(means, key=means.get)
    winner = max((log for log in runs if log["regime"] == regime),
                 key=lambda log: log["epochs"][-1]["validation_f1"])
    return winner, means, final_f1

def git_info():
    repo = Path(__file__).resolve().parent.parent.as_posix()
    command = ["git", "-c", f"safe.directory={repo}", "-C", repo]
    try:
        commit = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(command + ["status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def plot_batch(images, labels, path):
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    rgb = (images[:16] * std + mean).clamp(0, 1).permute(0, 2, 3, 1)
    figure, axes = plt.subplots(4, 4, figsize=(8, 8), layout="constrained")
    for ax, image, label in zip(axes.flat, rgb, labels):
        ax.imshow(image)
        ax.set_title(CLASSES[int(label)])
        ax.axis("off")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parameter_check(model, optimizer, regime):
    total = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    params_in_groups = [p for group in optimizer.param_groups for p in group["params"]]
    groups = [{"name": g["name"], "lr": g["lr"], "count": sum(p.numel() for p in g["params"])}
              for g in optimizer.param_groups]
    actual = [(g["name"], g["lr"], g["count"]) for g in groups]
    if (total != 11_179_590 or actual != GROUPS[regime]
            or sorted(map(id, params_in_groups)) != sorted(map(id, trainable))):
        raise RuntimeError(f"Параметри або групи оптимізатора не збігаються: {regime}.")
    return {"total": total, "trainable": sum(p.numel() for p in trainable), "groups": groups}


def state_check(regime, loader, loss_fn, device, wrong_mode=False):
    model, optimizer = setup_run(regime, 1, device)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if wrong_mode:
        model.train()
    else:
        set_train_mode(model, regime)
    losses = []
    for x, y in islice(loader, 5):
        losses.append(train_batch(model, optimizer, loss_fn, x, y, device).item())
    after = model.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    label = f"{regime} з model.train()" if wrong_mode else regime
    print(f"{label}: зміни після 5 кроків")
    print(f"{'Модуль':10}{'Параметри':>12}{'BN':>12}{'Збіг':>8}")
    yes_no = {True: "так", False: "ні", None: "немає BN"}
    passed = True
    for module in (*BACKBONE, "fc"):
        prefix = module + "."
        params = [name for name, _ in model.named_parameters() if name.startswith(prefix)]
        stats = [name for name in before if name.startswith(prefix)
                 and name.endswith((".running_mean", ".running_var"))]
        counters = [name for name in before if name.startswith(prefix)
                    and name.endswith(".num_batches_tracked")]
        param_change = any(name in changed for name in params)
        stat_change = any(name in changed for name in stats) if stats else None
        counter_change = any(name in changed for name in counters) if counters else None
        updates_params = module not in EVAL_MODULES[regime]
        updates_bn = updates_params or wrong_mode
        matches = (param_change == updates_params
                   and (not stats or (stat_change == updates_bn and counter_change == updates_bn)))
        print(f"{module:10}{yes_no[param_change]:>12}{yes_no[stat_change]:>12}{yes_no[matches]:>8}")
        passed = passed and matches
    if not passed:
        raise RuntimeError(f"Неочікувані зміни state_dict: {label}.")
    if wrong_mode:
        print("Втрати п'яти пакетів:", ", ".join(f"{loss:.4f}" for loss in losses))
        for stat in ("running_mean", "running_var"):
            name = f"layer1.0.bn1.{stat}"
            delta = (after[name] - before[name]).abs().max().item()
            print(f"Максимальна абсолютна зміна {name} = {delta:.6f}")


def preflight(figures, loader, device, loss_fn):
    print(f"Втрата для однакових імовірностей: {math.log(len(CLASSES)):.6f}")
    for regime in GROUPS:
        model, optimizer = setup_run(regime, 1, device)
        counts = parameter_check(model, optimizer, regime)
        print(f"{regime}: {counts['trainable']} із {counts['total']} параметрів навчаються")
        for group in counts["groups"]:
            print(f"  {group['name']}: {group['count']} параметрів, lr={group['lr']}")
        if regime in ("scratch", "frozen"):
            set_train_mode(model, regime)
            x, y = next(iter(loader))
            if regime == "scratch":
                plot_batch(x, y, figures / "train_batch.png")
            with torch.no_grad():
                loss = loss_fn(model(x.to(device)), y.to(device)).item()
            print(f"  Початкова втрата: {loss:.6f}")
        del optimizer, model
    for regime in ("frozen", "unfreeze"):
        state_check(regime, loader, loss_fn, device)
    print("Прогноз:", BN_PREDICTION, flush=True)
    state_check("frozen", loader, loss_fn, device, wrong_mode=True)
    print("Поділ, параметри і зміни стану збігаються з очікуваннями.", flush=True)


def run_settings(model, optimizer, train, validation, regime):
    return {
        "model": "torchvision.models.resnet18",
        "weights": None if regime == "scratch" else "IMAGENET1K_V1",
        "fc": {"in_features": 512, "out_features": 6, "bias": True, "init": "Linear defaults"},
        "eval_modules_during_train": EVAL_MODULES[regime],
        "parameters": parameter_check(model, optimizer, regime),
        "optimizer": {"name": "AdamW", **optimizer.defaults},
        "batch_norm": {"eps": model.bn1.eps, "momentum": model.bn1.momentum,
                       "affine": True, "track_running_stats": True},
        "loss": {"name": "CrossEntropyLoss", "reduction": "mean", "weight": None,
                 "ignore_index": -100, "label_smoothing": 0.0},
        "dtype": "float32",
        "transforms": {"rgb": True, "train": repr(train.dataset.transform),
                       "validation": repr(validation.dataset.transform),
                       "test": repr(validation.dataset.transform)},
        "loaders": {
            "train": {"batch_size": train.batch_size, "shuffle": True},
            "validation": {"batch_size": validation.batch_size, "shuffle": False},
            "test": {"batch_size": validation.batch_size, "shuffle": False},
            "common": {"drop_last": False, "num_workers": 0, "pin_memory": False,
                       "generator": None, "custom_sampler": None, "timeout": 0,
                       "persistent_workers": False, "prefetch_factor": None},
        },
        "epochs": EPOCHS, "steps_per_epoch": len(train), "last_train_batch_size": 16,
        "selection_epoch": EPOCHS, "validation": "eval + no_grad; argmax, macro-F1 over 6 classes",
        "lr_scheduler": None, "gradient_clipping": None, "amp": False, "early_stopping": False,
        "cpu_threads": torch.get_num_threads(),
    }


def train_run(out, weights_dir, regime, seed, train, validation, device, loss_fn, git, split_hash):
    model, optimizer = setup_run(regime, seed, device)
    settings = run_settings(model, optimizer, train, validation, regime)
    folder = run_dir(out, regime, seed)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{folder.name}: старт", flush=True)
    history = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        set_train_mode(model, regime)
        total_loss = torch.zeros((), device=device)
        for x, y in train:
            total_loss += train_batch(model, optimizer, loss_fn, x, y, device) * len(y)
            steps += 1
        loss = total_loss.item() / len(train.dataset)
        logits, labels = predict(model, validation, device)
        score = macro_f1(labels, logits.argmax(axis=1))
        history.append({"epoch": epoch, "train_loss": loss, "validation_f1": score})
        print(f"  {epoch}/{EPOCHS}: loss={loss:.4f}, val F1={score:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    if steps != 304 or logits.shape != (1498, 6):
        raise RuntimeError(f"Неочікувані кроки або форма логітів: {steps}, {logits.shape}.")
    np.save(folder / "val_logits.npy", logits)
    checkpoint = weights_dir / f"{folder.name}.pt"
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "regime": regime, "seed": seed, "classes": list(CLASSES), "img_size": IMG_SIZE},
               checkpoint)
    device_name = platform.processor()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_name = "Apple MPS"
    log = {
        "regime": regime, "seed": seed, "settings": settings, "git": git,
        "split_sha256": split_hash, "weights_sha256": sha256(checkpoint),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": torchvision.__version__, "numpy": np.__version__},
        "device": str(device), "device_name": device_name,
        "epochs": history, "optimizer_steps": steps, "seconds": elapsed,
    }
    (folder / "run.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    kind = args.device
    if kind == "auto":
        if torch.cuda.is_available():
            kind = "cuda"
        elif torch.backends.mps.is_available():
            kind = "mps"
        else:
            kind = "cpu"
    device = torch.device(kind)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    print(f"Пристрій: {device}", flush=True)
    git = git_info()
    figures = args.out / "results"
    figures.mkdir(parents=True, exist_ok=True)
    split = args.out / "split.csv"
    make_split(args.data, split)
    rows = read_split(split)
    split_hash = sha256(split)
    train = make_loader(Images(args.data, rows, "train"), train=True)
    val_images = Images(args.data, rows, "validation")
    validation = make_loader(val_images)
    loss_fn = torch.nn.CrossEntropyLoss()
    preflight(figures, train, device, loss_fn)
    # Заповнюємо кеш до відліку часу, щоб лише перший запуск не платив за декодування.
    for i in range(len(val_images)):
        val_images[i]
    with TemporaryDirectory(dir=args.out, prefix=".weights-") as temporary:
        weights_dir = Path(temporary)
        for regime, seed in RUNS:
            train_run(args.out, weights_dir, regime, seed, train, validation, device, loss_fn,
                      git, split_hash)
        winner, means, _ = select_run(load_runs(args.out))
        name = run_dir(args.out, winner["regime"], winner["seed"]).name
        (weights_dir / f"{name}.pt").replace(args.out / "lab3_model.pt")
    print(f"Середні validation macro-F1: {means}")
    print(f"Вибрано {name}: lab3_model.pt")


if __name__ == "__main__":
    main()
