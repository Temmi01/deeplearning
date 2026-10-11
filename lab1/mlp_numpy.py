import numpy as np
from sklearn.datasets import load_iris

def load_data():
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float64)
    rng = np.random.default_rng(0)
    
    tr, te = [], []
    for c in (0, 1, 2):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        tr.append(idx[:35])
        te.append(idx[35:])
        
    tr, te = np.concatenate(tr), np.concatenate(te)
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0, ddof=0)
    return (X[tr] - mean) / std, y[tr], (X[te] - mean) / std, y[te]

def init_params():
    rng = np.random.default_rng(0)
    
    W1 = rng.normal(0.0, np.sqrt(2 / 4), size=(4, 8)) # He
    W2 = rng.normal(0.0, np.sqrt(2 / (8 + 3)), size=(8, 3)) # Xavier
    return {"W1": W1, "b1": np.zeros(8), "W2": W2, "b2": np.zeros(3)}

def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]         
    a1 = np.maximum(z1, 0.0)            
    z2 = a1 @ p["W2"] + p["b2"]         
    return z2, {"X": X, "z1": z1, "a1": a1}

def cross_entropy(z2, y): 
    shifted = z2 - z2.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean()), logp

def loss_fn(X, y, p):
    z2, _ = forward(X, p)
    return cross_entropy(z2, y)[0]

def backward(z2, cache, y, p, make_error=False):
    # L = 1/N * sum( -(max (X*W1 + b1)*W2 + b2) + log(sum(exp(max (X*W1 + b1)*W2 + b2))) )
    N = len(y)
    _, logp = cross_entropy(z2, y)

    # L = -z2[y] + log(sum(exp(z2)))
    # L' true = -1 + exp(z2) / sum(exp(z2))
    # L' false = exp(z2) / sum(exp(z2))
    dz2 = np.exp(logp) - np.eye(3)[y]
           
    if not make_error:
        dz2 /= N   
    
    g = {}
    # z2 = a1 * W2 + b2
    g["W2"] = cache["a1"].T @ dz2       
    g["b2"] = dz2.sum(axis=0)            
    da1 = dz2 @ p["W2"].T  
                 
    # a1 = max(z1, 0)
    dz1 = da1 * (cache["z1"] > 0)   
        
    # z1 = X * W1 + b1
    g["W1"] = cache["X"].T @ dz1        
    g["b1"] = dz1.sum(axis=0)           
    return g

def loss_and_grads(X, y, p, make_error=False):
    z2, cache = forward(X, p)
    loss, _ = cross_entropy(z2, y)
    return loss, backward(z2, cache, y, p, make_error)
