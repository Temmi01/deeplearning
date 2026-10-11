import os
import numpy as np
import pandas as pd
import torch
from mlp_numpy import load_data, init_params, loss_and_grads, loss_fn

Xtr, ytr, _, _ = load_data()
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

for mode, make_error in (("correct", False), ("error", True)):
    p = init_params()
    print(f"\nРежим: {mode}")

    # 2 Звірка з PyTorch
    loss_np, g = loss_and_grads(Xtr, ytr, p, make_error)

    # реалізація в PyTorch
    l1 = torch.nn.Linear(4, 8).double()
    l2 = torch.nn.Linear(8, 3).double()
    with torch.no_grad():
        l1.weight.copy_(torch.from_numpy(p["W1"].T))
        l1.bias.copy_(torch.from_numpy(p["b1"]))
        l2.weight.copy_(torch.from_numpy(p["W2"].T))
        l2.bias.copy_(torch.from_numpy(p["b2"]))
    net = torch.nn.Sequential(l1, torch.nn.ReLU(), l2)

    out = net(torch.from_numpy(Xtr))
    loss_torch = torch.nn.functional.cross_entropy(out, torch.from_numpy(ytr))
    loss_torch.backward()
    loss_t = loss_torch.item()
    
    gt = {"W1": l1.weight.grad.numpy().T, "b1": l1.bias.grad.numpy(),
          "W2": l2.weight.grad.numpy().T, "b2": l2.bias.grad.numpy()}

    # звірка результатів
    print(f"\nВтрата NumPy  = {loss_np}\nВтрата PyTorch = {loss_t}")
    torch_rows = []
    
    d = abs(loss_np - loss_t)
    torch_rows.append(["Втрата", d, "так" if d <= 1e-12 else "ні"])
    
    for k in ("W1", "b1", "W2", "b2"):
        d = np.max(np.abs(g[k] - gt[k]))
        torch_rows.append([f"Градієнт {k}", d, "так" if d <= 1e-12 else "ні"])

    df_torch = pd.DataFrame(torch_rows, columns=["Величина", "Макс. абс. різниця", "Пройдено"])
    print("\n## Звірка з PyTorch")
    print(df_torch)
    df_torch.to_csv(os.path.join(results_dir, f"{mode}_torch_check.csv"), index=False)

    # перевірка окремих градієнтів
    eps = 1e-6
    num_rows = []
    for k, ix in (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))):
        orig = p[k][ix]
        p[k][ix] = orig + eps
        L_plus = loss_fn(Xtr, ytr, p)
        p[k][ix] = orig - eps
        L_minus = loss_fn(Xtr, ytr, p)
        p[k][ix] = orig
        g_num = (L_plus - L_minus) / (2 * eps)
        g_manual = g[k][ix]
        d = abs(g_num - g_manual)
        num_rows.append([f"{k}{list(ix)}", g_manual, g_num, d, "так" if d <= 1e-7 else "ні"])

    df_num = pd.DataFrame(num_rows, columns=["Параметр", "backward()", "Чисельна похідна",
                                             "Абс. різниця", "Пройдено"])
    print("\n## Чисельна перевірка (центральна різниця, eps=e-6)")
    print(df_num)
    df_num.to_csv(os.path.join(results_dir, f"{mode}_numeric_check.csv"), index=False)
