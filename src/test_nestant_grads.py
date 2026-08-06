"""
test_nestant_grads.py — finite-difference gradient check on the NestAnt
forward/backward pair. This is not shipped with the paper; it is a
correctness test for the hand-written backprop.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from nestant import (
    NestAntConfig, init_params, forward, backward
)


def loss_from_params(params, X, y, tasks):
    y_hat, _ = forward(params, X, tasks)
    diff = (y_hat - y)
    return float((diff ** 2).mean())


def check():
    rng = np.random.default_rng(42)
    N, nf, nt = 8, 5, 3
    X = rng.standard_normal((N, nf))
    y = rng.standard_normal((N, nt))
    tasks = rng.integers(0, 3, size=N)

    cfg = NestAntConfig(n_features=nf, n_targets=nt, n_bands=3,
                        hidden=16, embed=8, band_embed=4)
    params = init_params(cfg)

    y_hat, cache = forward(params, X, tasks)
    g_slow, g_fast, _ = backward(params, cache, y, y_hat, weight_decay=0.0)

    eps = 1e-5
    max_rel = 0.0
    for group, grads in (("slow", g_slow), ("fast", g_fast)):
        for name, g_analytic in grads.items():
            p = getattr(params, name)
            g_num = np.zeros_like(p)
            it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                orig = p[idx]
                p[idx] = orig + eps
                Lp = loss_from_params(params, X, y, tasks)
                p[idx] = orig - eps
                Lm = loss_from_params(params, X, y, tasks)
                p[idx] = orig
                g_num[idx] = (Lp - Lm) / (2 * eps)
                it.iternext()
            diff = np.abs(g_analytic - g_num)
            denom = np.abs(g_analytic) + np.abs(g_num) + 1e-12
            rel = (diff / denom).max()
            max_rel = max(max_rel, rel)
            print(f"{group}/{name:<4}  shape={p.shape}  max_rel_err={rel:.3e}")
    print(f"\nOverall max relative error: {max_rel:.3e}")
    assert max_rel < 1e-4, "Gradient check failed"
    print("Gradient check PASSED")


if __name__ == "__main__":
    check()
