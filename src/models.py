"""
models.py -- NumPy models with hand-derived gradients.

All neural models share: GELU activations, Glorot-uniform initialisation,
mean-squared-error loss averaged over samples and the three targets, Adam
(beta1=0.9, beta2=0.999, eps=1e-8) and an L2 penalty of 1e-5 on every weight
matrix (added to the gradient; biases are not penalised).

  MLP          5 -> 128 -> 80 -> 3                 (fine-tuning, EWC, replay,
                                                    independent, joint)
  Progressive  one MLP column per task; column t also receives
               tanh(h2 of every earlier column) @ U_j in its second hidden
               pre-activation (Rusu et al. 2016). Earlier columns are frozen.
  NestAnt-CMS  shared encoder 5 -> 128 -> 64, a 16-d code per task and one
               head per task (64+16) -> 80 -> 3. Encoder, first code and first
               head train on task 1; afterwards the encoder is frozen and only
               the current task's code and head train.
"""
from __future__ import annotations

import numpy as np

WD = 1e-5


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    s = np.sqrt(2.0 / np.pi)
    t = np.tanh(s * (x + 0.044715 * x ** 3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * s * (1.0 + 3 * 0.044715 * x ** 2)


def glorot(rng, shape):
    lim = np.sqrt(6.0 / (shape[0] + shape[-1]))
    return rng.uniform(-lim, lim, size=shape)


def n_params(p: dict) -> int:
    return int(sum(v.size for v in p.values()))


class Adam:
    """Adam with one state per parameter key; only keys passed to step move."""

    def __init__(self, params: dict, lr: float):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = {k: 0 for k in params}

    def add(self, params: dict):
        for k, v in params.items():
            if k not in self.m:
                self.m[k] = np.zeros_like(v); self.v[k] = np.zeros_like(v); self.t[k] = 0

    def step(self, params: dict, grads: dict, keys, lr: float | None = None,
             b1=0.9, b2=0.999, eps=1e-8):
        lr = self.lr if lr is None else lr
        for k in keys:
            g = grads[k]
            self.t[k] += 1
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t[k])
            vh = self.v[k] / (1 - b2 ** self.t[k])
            params[k] = params[k] - lr * mh / (np.sqrt(vh) + eps)


def add_l2(params: dict, grads: dict):
    for k in grads:
        if k.startswith("W") or k.startswith("U"):
            grads[k] = grads[k] + WD * params[k]


# ---------------------------------------------------------------- MLP
def mlp_init(rng, n_in=5, h1=128, h2=80, n_out=3):
    return dict(W1=glorot(rng, (n_in, h1)), b1=np.zeros(h1),
                W2=glorot(rng, (h1, h2)), b2=np.zeros(h2),
                W3=glorot(rng, (h2, n_out)), b3=np.zeros(n_out))


def mlp_forward(p, X):
    a1 = X @ p["W1"] + p["b1"]; h1 = gelu(a1)
    a2 = h1 @ p["W2"] + p["b2"]; h2 = gelu(a2)
    return h2 @ p["W3"] + p["b3"], dict(X=X, a1=a1, h1=h1, a2=a2, h2=h2)


def _mlp_deltas(p, c, dy):
    d2 = (dy @ p["W3"].T) * gelu_grad(c["a2"])
    d1 = (d2 @ p["W2"].T) * gelu_grad(c["a1"])
    return d1, d2


def mlp_backward(p, c, y, yhat):
    """Gradient of mean_{n,k} (yhat - y)^2 (no L2 term)."""
    N, K = y.shape
    dy = (2.0 / (N * K)) * (yhat - y)
    d1, d2 = _mlp_deltas(p, c, dy)
    return dict(W3=c["h2"].T @ dy, b3=dy.sum(0), W2=c["h1"].T @ d2, b2=d2.sum(0),
                W1=c["X"].T @ d1, b1=d1.sum(0))


def mlp_fisher(p, X, y, batch=2048):
    """Exact diagonal empirical Fisher: mean over samples of the squared
    per-sample gradient of the per-sample loss (1/K) sum_k (yhat-y)^2."""
    F = {k: np.zeros_like(v) for k, v in p.items()}
    K = y.shape[1]
    for s in range(0, len(X), batch):
        yhat, c = mlp_forward(p, X[s:s + batch])
        dy = (2.0 / K) * (yhat - y[s:s + batch])          # per-sample output grads
        d1, d2 = _mlp_deltas(p, c, dy)
        for W, b, a, d in (("W3", "b3", c["h2"], dy), ("W2", "b2", c["h1"], d2),
                           ("W1", "b1", c["X"], d1)):
            F[W] += (a ** 2).T @ (d ** 2)
            F[b] += (d ** 2).sum(0)
    return {k: v / len(X) for k, v in F.items()}


# ---------------------------------------------------------------- Progressive
def prog_column_init(rng, n_prev, n_in=5, h1=128, h2=80, n_out=3):
    p = mlp_init(rng, n_in, h1, h2, n_out)
    for j in range(n_prev):
        p[f"U{j}"] = glorot(rng, (h2, h2))
    return p


def prog_hidden(columns, X):
    """Second-layer activations h2 of every (frozen) column, in order."""
    hs = []
    for col in columns:
        a2 = gelu(X @ col["W1"] + col["b1"]) @ col["W2"] + col["b2"]
        for j, hj in enumerate(hs):
            a2 = a2 + np.tanh(hj) @ col[f"U{j}"]
        hs.append(gelu(a2))
    return hs


def prog_forward(col, X, prev_h):
    a1 = X @ col["W1"] + col["b1"]; h1 = gelu(a1)
    a2 = h1 @ col["W2"] + col["b2"]
    lat = [np.tanh(h) for h in prev_h]
    for j, l in enumerate(lat):
        a2 = a2 + l @ col[f"U{j}"]
    h2 = gelu(a2)
    return h2 @ col["W3"] + col["b3"], dict(X=X, a1=a1, h1=h1, a2=a2, h2=h2, lat=lat)


def prog_backward(col, c, y, yhat):
    N, K = y.shape
    dy = (2.0 / (N * K)) * (yhat - y)
    d2 = (dy @ col["W3"].T) * gelu_grad(c["a2"])
    d1 = (d2 @ col["W2"].T) * gelu_grad(c["a1"])
    g = dict(W3=c["h2"].T @ dy, b3=dy.sum(0), W2=c["h1"].T @ d2, b2=d2.sum(0),
             W1=c["X"].T @ d1, b1=d1.sum(0))
    for j, l in enumerate(c["lat"]):
        g[f"U{j}"] = l.T @ d2
    return g


# ---------------------------------------------------------------- NestAnt-CMS
def cms_init(rng, n_tasks, n_in=5, hidden=128, embed=64, code=16, head=80, n_out=3):
    p = dict(W1=glorot(rng, (n_in, hidden)), b1=np.zeros(hidden),
             W2=glorot(rng, (hidden, embed)), b2=np.zeros(embed))
    for t in range(n_tasks):
        p[f"E{t}"] = glorot(rng, (1, code))[0]
        p[f"W3_{t}"] = glorot(rng, (embed + code, head)); p[f"b3_{t}"] = np.zeros(head)
        p[f"W4_{t}"] = glorot(rng, (head, n_out)); p[f"b4_{t}"] = np.zeros(n_out)
    return p


def cms_forward(p, X, t):
    """Forward pass for rows that all belong to task t."""
    a1 = X @ p["W1"] + p["b1"]; h = gelu(a1)
    z = h @ p["W2"] + p["b2"]
    zc = np.concatenate([z, np.broadcast_to(p[f"E{t}"], (len(X), p[f"E{t}"].size))], 1)
    a3 = zc @ p[f"W3_{t}"] + p[f"b3_{t}"]; f = gelu(a3)
    return f @ p[f"W4_{t}"] + p[f"b4_{t}"], dict(X=X, a1=a1, h=h, z=z, zc=zc, a3=a3, f=f)


def cms_backward(p, c, y, yhat, t):
    N, K = y.shape
    dy = (2.0 / (N * K)) * (yhat - y)
    d3 = (dy @ p[f"W4_{t}"].T) * gelu_grad(c["a3"])
    dzc = d3 @ p[f"W3_{t}"].T
    e = c["z"].shape[1]
    dz, dE = dzc[:, :e], dzc[:, e:].sum(0)
    d1 = (dz @ p["W2"].T) * gelu_grad(c["a1"])
    return {f"W4_{t}": c["f"].T @ dy, f"b4_{t}": dy.sum(0),
            f"W3_{t}": c["zc"].T @ d3, f"b3_{t}": d3.sum(0), f"E{t}": dE,
            "W2": c["h"].T @ dz, "b2": dz.sum(0), "W1": c["X"].T @ d1, "b1": d1.sum(0)}


def cms_task_params(p, n_tasks_used):
    """Parameters that exist after n_tasks_used tasks: encoder plus one code
    and one head per task."""
    keys = ["W1", "b1", "W2", "b2"]
    for t in range(n_tasks_used):
        keys += [f"E{t}", f"W3_{t}", f"b3_{t}", f"W4_{t}", f"b4_{t}"]
    return int(sum(p[k].size for k in keys))
