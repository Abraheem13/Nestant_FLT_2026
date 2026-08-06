"""
nestant_cms.py -- NestAnt with per-band fast heads.

Realises the Continuum Memory System (CMS) idea from NL Section 7 in its
strongest form for our setting: each frequency band has its own dedicated
fast-level associative memory M^{fast}_b. When training on band b, only
the slow prior (W1,W2) and band-b's fast head (W3_b, W4_b) update. Bands
b' != b are untouched, so by construction there is zero forgetting on the
fast level. The slow level can still drift across tasks; this is
controllable via slow_lr or via stage-wise slow freezing.

Forward path for an input belonging to band b:
    z = M_slow(x)             (shared physics prior)
    f = GELU(W3_b @ [z; e_b] + b3_b)
    y = W4_b @ f + b4_b

Architecture motivation in the paper: this is the strict NL recipe applied
to multi-band antenna design. The slow level encodes universal patch-antenna
physics that holds across bands (transmission-line model intuition), while
the fast level for each band encodes band-specific deviations from that
prior. The result is a model that IS continual by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nestant import (_glorot, _gelu, _gelu_grad, AdamState, _init_adam_like,
                     adam_step, iterate_minibatches)


# =====================================================================
# Config / params
# =====================================================================
@dataclass
class CMSConfig:
    n_features: int
    n_targets: int
    n_bands: int = 3
    hidden: int = 128
    embed: int = 64
    band_embed: int = 16
    fast_hidden: int = 80
    slow_lr: float = 1e-3
    fast_lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    batch_size: int = 1024
    seed: int = 0


@dataclass
class CMSParams:
    # Slow level (shared)
    W1: np.ndarray; b1: np.ndarray
    W2: np.ndarray; b2: np.ndarray
    # Per-band fast heads -- stored as 3D arrays indexed by band
    Eb:  np.ndarray   # (n_bands, band_embed)
    W3b: np.ndarray   # (n_bands, embed+band_embed, fast_hidden)
    b3b: np.ndarray   # (n_bands, fast_hidden)
    W4b: np.ndarray   # (n_bands, fast_hidden, n_targets)
    b4b: np.ndarray   # (n_bands, n_targets)


def init_cms(cfg: CMSConfig) -> CMSParams:
    rng = np.random.default_rng(cfg.seed)
    return CMSParams(
        W1=_glorot(rng, (cfg.n_features, cfg.hidden)),
        b1=np.zeros(cfg.hidden),
        W2=_glorot(rng, (cfg.hidden, cfg.embed)),
        b2=np.zeros(cfg.embed),
        Eb=_glorot(rng, (cfg.n_bands, cfg.band_embed)),
        W3b=np.stack([_glorot(rng, (cfg.embed + cfg.band_embed, cfg.fast_hidden))
                      for _ in range(cfg.n_bands)], axis=0),
        b3b=np.zeros((cfg.n_bands, cfg.fast_hidden)),
        W4b=np.stack([_glorot(rng, (cfg.fast_hidden, cfg.n_targets))
                      for _ in range(cfg.n_bands)], axis=0),
        b4b=np.zeros((cfg.n_bands, cfg.n_targets)),
    )


@dataclass
class CMSOpt:
    slow: dict[str, AdamState]
    Eb: AdamState
    W3b: list[AdamState]   # one Adam state per band
    b3b: list[AdamState]
    W4b: list[AdamState]
    b4b: list[AdamState]


def init_cms_opt(p: CMSParams, n_bands: int) -> CMSOpt:
    return CMSOpt(
        slow={k: _init_adam_like(getattr(p, k)) for k in ("W1","b1","W2","b2")},
        Eb=_init_adam_like(p.Eb),
        W3b=[_init_adam_like(p.W3b[i]) for i in range(n_bands)],
        b3b=[_init_adam_like(p.b3b[i]) for i in range(n_bands)],
        W4b=[_init_adam_like(p.W4b[i]) for i in range(n_bands)],
        b4b=[_init_adam_like(p.b4b[i]) for i in range(n_bands)],
    )


# =====================================================================
# Forward / backward (uses gather-by-band)
# =====================================================================
def cms_forward(p: CMSParams, X: np.ndarray, tasks: np.ndarray):
    """Forward pass with per-row band routing.

    Fast-path: when all rows belong to one band b, use dense matmul with
    p.W3b[b], p.W4b[b]. This is exactly the continual-protocol case and
    is ~10x faster than einsum.
    """
    h_pre = X @ p.W1 + p.b1
    h = _gelu(h_pre)
    z = h @ p.W2 + p.b2
    b_emb = p.Eb[tasks]
    zc = np.concatenate([z, b_emb], axis=1)

    # Fast path: single-band batch
    unique_bands = np.unique(tasks)
    if unique_bands.size == 1:
        b = int(unique_bands[0])
        f_pre = zc @ p.W3b[b] + p.b3b[b]
        f = _gelu(f_pre)
        y_hat = f @ p.W4b[b] + p.b4b[b]
    else:
        W3 = p.W3b[tasks]
        b3 = p.b3b[tasks]
        f_pre = np.einsum("ni,nio->no", zc, W3) + b3
        f = _gelu(f_pre)
        W4 = p.W4b[tasks]
        b4 = p.b4b[tasks]
        y_hat = np.einsum("ni,nio->no", f, W4) + b4

    cache = dict(X=X, tasks=tasks, h_pre=h_pre, h=h, z=z, b_emb=b_emb,
                 zc=zc, f_pre=f_pre, f=f,
                 _single_band=int(unique_bands[0]) if unique_bands.size==1 else None)
    return y_hat, cache


def cms_backward(p: CMSParams, cache, y_true, y_hat, n_bands: int,
                 weight_decay: float = 0.0):
    N = y_true.shape[0]
    diff = (y_hat - y_true)
    loss_val = float((diff ** 2).mean())
    dy = (2.0 / (N * y_true.shape[1])) * diff   # (N, n_targets)

    tasks = cache["tasks"]; f = cache["f"]; zc = cache["zc"]
    sb = cache.get("_single_band", None)

    dW4b = np.zeros_like(p.W4b)
    db4b = np.zeros_like(p.b4b)
    dW3b = np.zeros_like(p.W3b)
    db3b = np.zeros_like(p.b3b)

    if sb is not None:
        # Fast path: single band b
        b = sb
        dW4b[b] = f.T @ dy
        db4b[b] = dy.sum(0)
        df = dy @ p.W4b[b].T
        df_pre = df * _gelu_grad(cache["f_pre"])
        dW3b[b] = zc.T @ df_pre
        db3b[b] = df_pre.sum(0)
        dzc = df_pre @ p.W3b[b].T
    else:
        # General path
        for b in range(n_bands):
            m = (tasks == b)
            if not m.any():
                continue
            dW4b[b] = f[m].T @ dy[m]
            db4b[b] = dy[m].sum(0)
        W4_per_row = p.W4b[tasks]
        df = np.einsum("no,nio->ni", dy, W4_per_row)
        df_pre = df * _gelu_grad(cache["f_pre"])
        for b in range(n_bands):
            m = (tasks == b)
            if not m.any():
                continue
            dW3b[b] = zc[m].T @ df_pre[m]
            db3b[b] = df_pre[m].sum(0)
        W3_per_row = p.W3b[tasks]
        dzc = np.einsum("ni,nji->nj", df_pre, W3_per_row)

    embed_dim = cache["z"].shape[1]
    dz = dzc[:, :embed_dim]
    dbe = dzc[:, embed_dim:]

    dEb = np.zeros_like(p.Eb)
    np.add.at(dEb, tasks, dbe)

    h = cache["h"]
    dW2 = h.T @ dz
    db2 = dz.sum(0)
    dh = dz @ p.W2.T
    dh_pre = dh * _gelu_grad(cache["h_pre"])
    X = cache["X"]
    dW1 = X.T @ dh_pre
    db1 = dh_pre.sum(0)

    if weight_decay > 0.0:
        dW1 += weight_decay * p.W1
        dW2 += weight_decay * p.W2
        for b in range(n_bands):
            dW3b[b] += weight_decay * p.W3b[b]
            dW4b[b] += weight_decay * p.W4b[b]

    grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2,
                 Eb=dEb, W3b=dW3b, b3b=db3b, W4b=dW4b, b4b=db4b)
    return grads, loss_val


def cms_update(p: CMSParams, grads, opt: CMSOpt,
               slow_lr: float, fast_lr: float, weight_decay: float,
               active_bands: list[int] | None = None):
    """Apply Adam updates. active_bands restricts which fast-head bands
    update -- by default ALL bands are updated, even if their gradient is
    zero (which happens automatically when a band has no rows). For the
    continual protocol we pass active_bands=[current_task] to mathematically
    enforce that other bands' fast heads are untouched."""
    # Slow
    for name in ("W1","b1","W2","b2"):
        cur = getattr(p, name)
        new, st = adam_step(cur, grads[name], opt.slow[name], slow_lr,
                            weight_decay=(weight_decay if "W" in name else 0.0))
        setattr(p, name, new); opt.slow[name] = st

    # Eb update for ALL rows that occurred (handled by gradient being 0 elsewhere)
    p.Eb, opt.Eb = adam_step(p.Eb, grads["Eb"], opt.Eb, fast_lr,
                             weight_decay=0.0)

    # Per-band fast heads
    n_bands = p.W3b.shape[0]
    bands = active_bands if active_bands is not None else list(range(n_bands))
    for b in bands:
        p.W3b[b], opt.W3b[b] = adam_step(p.W3b[b], grads["W3b"][b],
                                          opt.W3b[b], fast_lr,
                                          weight_decay=weight_decay)
        p.b3b[b], opt.b3b[b] = adam_step(p.b3b[b], grads["b3b"][b],
                                          opt.b3b[b], fast_lr, 0.0)
        p.W4b[b], opt.W4b[b] = adam_step(p.W4b[b], grads["W4b"][b],
                                          opt.W4b[b], fast_lr,
                                          weight_decay=weight_decay)
        p.b4b[b], opt.b4b[b] = adam_step(p.b4b[b], grads["b4b"][b],
                                          opt.b4b[b], fast_lr, 0.0)


def cms_predict(p: CMSParams, X: np.ndarray, tasks: np.ndarray):
    y_hat, _ = cms_forward(p, X, tasks)
    return y_hat
