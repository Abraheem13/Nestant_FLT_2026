"""
test_gradients.py -- central finite-difference checks of every hand-derived
gradient used in the benchmark, at the exact layer sizes used in the runs.

For each parameter array the reported error is
    ||g_analytic - g_numeric||_2 / max(||g_analytic||_2, ||g_numeric||_2).
The per-sample Fisher is checked against a loop of single-sample gradients.
Exits non-zero if any error exceeds 1e-6.
"""
import json
import sys

import numpy as np

import models as M

EPS = 1e-6
TOL = 1e-6


def loss(yhat, y):
    return float(((yhat - y) ** 2).mean())


def fd_check(params, keys, f_loss, grads):
    worst = {}
    for k in keys:
        p = params[k]; gn = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index; o = p[i]
            p[i] = o + EPS; lp = f_loss()
            p[i] = o - EPS; lm = f_loss()
            p[i] = o; gn[i] = (lp - lm) / (2 * EPS)
            it.iternext()
        ga = grads[k]
        worst[k] = float(np.linalg.norm(ga - gn) / max(np.linalg.norm(ga), np.linalg.norm(gn), 1e-300))
    return worst


def main():
    rng = np.random.default_rng(1)
    N = 6
    X = rng.standard_normal((N, 5)); y = rng.standard_normal((N, 3))
    out = {}

    p = M.mlp_init(rng)
    yhat, c = M.mlp_forward(p, X)
    out["mlp"] = fd_check(p, list(p), lambda: loss(M.mlp_forward(p, X)[0], y), M.mlp_backward(p, c, y, yhat))

    # Fisher: per-sample gradients of the per-sample loss, squared and averaged
    F = M.mlp_fisher(p, X, y, batch=4)
    ref = {k: np.zeros_like(v) for k, v in p.items()}
    for n in range(N):
        yh, cn = M.mlp_forward(p, X[n:n + 1])
        g = M.mlp_backward(p, cn, y[n:n + 1], yh)          # = grad of (1/K) sum_k (.)^2
        for k in ref:
            ref[k] += g[k] ** 2 / N
    out["fisher"] = {k: float(np.linalg.norm(F[k] - ref[k]) / np.linalg.norm(ref[k])) for k in F}

    cols = [M.prog_column_init(rng, 0), M.prog_column_init(rng, 1)]
    col = M.prog_column_init(rng, 2)
    prev = M.prog_hidden(cols, X)
    yhat, c = M.prog_forward(col, X, prev)
    out["progressive"] = fd_check(col, list(col), lambda: loss(M.prog_forward(col, X, prev)[0], y),
                                  M.prog_backward(col, c, y, yhat))

    p = M.cms_init(rng, 3); t = 1
    yhat, c = M.cms_forward(p, X, t)
    g = M.cms_backward(p, c, y, yhat, t)
    out["cms"] = fd_check(p, list(g), lambda: loss(M.cms_forward(p, X, t)[0], y), g)

    worst = max(v for d in out.values() for v in d.values())
    for name, d in out.items():
        print(f"{name:<12} max relative error {max(d.values()):.2e}  ({len(d)} arrays)")
    print(f"overall max relative error {worst:.2e}")
    json.dump(dict(errors=out, overall=worst), open("../results/gradient_check.json", "w"), indent=1)
    sys.exit(0 if worst < TOL else 1)


if __name__ == "__main__":
    main()
