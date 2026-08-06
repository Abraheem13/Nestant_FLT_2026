"""
test_cms_grads.py -- finite-difference gradient check for nestant_cms.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from nestant_cms import (CMSConfig, init_cms, cms_forward, cms_backward)

def loss(p, X, y, tasks):
    yhat, _ = cms_forward(p, X, tasks)
    return float(((yhat - y) ** 2).mean())

def check():
    rng = np.random.default_rng(0)
    N, nf, nt, nb = 12, 5, 3, 3
    X = rng.standard_normal((N, nf))
    y = rng.standard_normal((N, nt))
    tasks = rng.integers(0, nb, size=N)
    cfg = CMSConfig(n_features=nf, n_targets=nt, n_bands=nb,
                    hidden=16, embed=8, band_embed=4, fast_hidden=12)
    p = init_cms(cfg)
    yhat, cache = cms_forward(p, X, tasks)
    grads, _ = cms_backward(p, cache, y, yhat, n_bands=nb, weight_decay=0.0)
    eps = 1e-5
    max_rel = 0.0
    # Check each parameter array
    for name in ("W1","b1","W2","b2","Eb","W3b","b3b","W4b","b4b"):
        arr = getattr(p, name)
        ga = grads[name]
        gn = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            Lp = loss(p, X, y, tasks)
            arr[idx] = orig - eps
            Lm = loss(p, X, y, tasks)
            arr[idx] = orig
            gn[idx] = (Lp - Lm) / (2*eps)
            it.iternext()
        denom = np.abs(ga) + np.abs(gn) + 1e-12
        rel = (np.abs(ga - gn) / denom).max()
        max_rel = max(max_rel, rel)
        print(f"{name:<5} shape={arr.shape}  max_rel_err={rel:.3e}")
    print(f"Overall max rel err: {max_rel:.3e}")
    assert max_rel < 1e-4
    print("PASSED")


if __name__ == "__main__":
    check()
