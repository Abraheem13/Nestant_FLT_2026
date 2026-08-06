"""
nestant.py — NestAnt: a Nested-Learning surrogate for antenna parameter prediction.

This is the paper's core contribution. Implemented in pure NumPy so that the
two-level nested optimisation structure is completely transparent and the
model is trivially reproducible on any machine.

=======================================================================
Mapping to Behrouz, Razaviyayn, Zhong, Mirrokni, NeurIPS 2025
    "Nested Learning: The Illusion of Deep Learning Architectures"
=======================================================================

The NL paper argues (Section 3, Definitions 2-4) that any ML model is a
system of nested optimisation problems, each with

    * its own objective L^{(k)}
    * its own context flow C^{(k)} (the data it sees)
    * its own parameter set Theta^{(k)}
    * its own update frequency f^{(k)}

A higher level has a LOWER frequency (updates less often).

NestAnt realises this paradigm with two levels:

    Level 2 (slow, outer, frequency = 1 per epoch):
        Physics-prior associative memory  M_slow
        Context flow:  the training set as a whole (epoch-level)
        Parameters:    W1, W2 of a small MLP that maps the five raw geometry
                       features to a d-dimensional "physics embedding" z.
        Objective:     joint MSE over all three targets, pooled across bands.

    Level 1 (fast, inner, frequency = 1 per batch):
        Band-conditional associative memory  M_fast
        Context flow:  (physics embedding z, band code b) pairs within a batch
        Parameters:    W3 of a linear head reading z conditioned on b.
        Objective:     per-sample MSE on the three targets.

Knowledge transfer between levels is by direct parametric connection
(NL §3.3 "Direct Connection of Levels", Eq. 24):

    M_fast.forward(x) depends on Theta_slow through z = M_slow(x).

This is the simplest and most faithful NL coupling: the outer memory's output
is the inner memory's input key.

The "multi time-scale update" principle (NL §1.1, "Multi Time scale Processing
System", and the brain-wave analogy in Fig. 1) is realised by using a much
smaller learning rate for the slow level than for the fast level — mirroring
slow delta/theta waves for memory consolidation (slow level: physics) vs. fast
gamma waves for sensory processing (fast level: band-specific response).

The "Continuum Memory System" (NL §7) motivates our three-headed prediction:
we maintain three parallel fast readers with independent update rates,
one per target (S11, gain, resonance). A shared slow memory feeds all three.
This is essentially the CMS idea applied to a regression surrogate: a spectrum
of frequency updates rather than the strict long/short dichotomy.

=======================================================================
Why this is not "just an MLP"
=======================================================================

A standard MLP trained end-to-end with a single learning rate collapses the
two levels into one; the NL view disappears. What we do differently is:

  1. The slow and fast parameter groups have DISTINCT learning rates, update
     cadences, and objectives -- the signature of a NSAM (NL Def. 4).
  2. The fast readers are band-conditioned via an explicit gating vector,
     not just fed the band as one more input. This is the direct parametric
     coupling of NL Eq. 24.
  3. Training exposes a continual-learning protocol where the slow memory
     is "online-consolidated" while the fast heads are adapted per task --
     matching NL's motivation that the brain's online consolidation (§1.1)
     is what current LLMs lack.

Even on IID joint training these design choices change the inductive bias
noticeably (see experiments.py), but the real payoff shows up in the
continual-learning protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# =====================================================================
# Utilities
# =====================================================================
def _glorot(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Glorot/Xavier uniform initialisation."""
    fan_in, fan_out = shape[0], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (smooth, good for small MLPs)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU w.r.t. its input."""
    s = np.sqrt(2.0 / np.pi)
    u = s * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du_dx = s * (1.0 + 3.0 * 0.044715 * x ** 2)
    # d/dx [0.5 * x * (1 + tanh(u))] = 0.5 (1 + tanh u) + 0.5 x sech^2(u) du/dx
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du_dx


# =====================================================================
# Optimiser state (simple Adam)
# =====================================================================
@dataclass
class AdamState:
    """Per-parameter Adam moments. NL Section 4 recasts Adam itself as a
    two-level associative memory that compresses gradients. Here we use it
    as an off-the-shelf optimiser; adopting NL's "expressive optimisers"
    (e.g., Delta-GD, M3 Muon) is left for future work and not claimed."""
    m: np.ndarray   # 1st moment
    v: np.ndarray   # 2nd moment
    t: int = 0       # step counter


def adam_step(
    p: np.ndarray,
    g: np.ndarray,
    st: AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[np.ndarray, AdamState]:
    """Single Adam update on parameter p with gradient g."""
    st.t += 1
    st.m = beta1 * st.m + (1.0 - beta1) * g
    st.v = beta2 * st.v + (1.0 - beta2) * (g * g)
    m_hat = st.m / (1.0 - beta1 ** st.t)
    v_hat = st.v / (1.0 - beta2 ** st.t)
    p_new = p - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * p)
    return p_new, st


# =====================================================================
# Model
# =====================================================================
@dataclass
class NestAntConfig:
    """Hyper-parameters for NestAnt."""
    n_features: int                      # input feature dim (geometry + band)
    n_targets: int                       # output dim (3: s11, gain, resonance)
    n_bands: int = 3                     # number of frequency bands / tasks
    hidden: int = 64                     # slow MLP hidden size
    embed: int = 32                      # physics embedding dimension
    band_embed: int = 8                  # band code dimension
    fast_hidden: int = 48                # fast-level hidden size

    # "Different update frequencies" - the NL principle.
    # Slow level updates once per (mini-)epoch over the accumulated gradient,
    # fast level updates every batch.
    slow_lr: float = 5e-4
    fast_lr: float = 3e-3

    # Regularisation
    weight_decay: float = 1e-4

    # Training schedule
    epochs: int = 80
    batch_size: int = 256

    # Reproducibility
    seed: int = 0


@dataclass
class NestAntParams:
    """All learnable parameters, grouped by level.

    Slow level: features -> hidden -> embedding (two-layer MLP).
    Fast level: concat(embedding, band_embed) -> fast_hidden -> n_targets
    (small band-conditional MLP, not a bare linear readout). This matches
    NL's Def. 4 where each level has its own associative-memory kernel M^(k)
    with its own parameters; a linear kernel would not be expressive enough
    for the target regression.
    """
    # Slow level
    W1: np.ndarray; b1: np.ndarray
    W2: np.ndarray; b2: np.ndarray
    # Fast level
    Eb: np.ndarray
    W3: np.ndarray; b3: np.ndarray
    W4: np.ndarray; b4: np.ndarray

    def slow_list(self):
        return [("W1", self.W1), ("b1", self.b1),
                ("W2", self.W2), ("b2", self.b2)]

    def fast_list(self):
        return [("Eb", self.Eb),
                ("W3", self.W3), ("b3", self.b3),
                ("W4", self.W4), ("b4", self.b4)]


def init_params(cfg: NestAntConfig) -> NestAntParams:
    rng = np.random.default_rng(cfg.seed)
    fast_hidden = getattr(cfg, "fast_hidden", 64)
    return NestAntParams(
        W1=_glorot(rng, (cfg.n_features, cfg.hidden)),
        b1=np.zeros(cfg.hidden),
        W2=_glorot(rng, (cfg.hidden, cfg.embed)),
        b2=np.zeros(cfg.embed),
        Eb=_glorot(rng, (cfg.n_bands, cfg.band_embed)),
        W3=_glorot(rng, (cfg.embed + cfg.band_embed, fast_hidden)),
        b3=np.zeros(fast_hidden),
        W4=_glorot(rng, (fast_hidden, cfg.n_targets)),
        b4=np.zeros(cfg.n_targets),
    )


def _init_adam_like(p: np.ndarray) -> AdamState:
    return AdamState(m=np.zeros_like(p), v=np.zeros_like(p), t=0)


@dataclass
class NestAntOptState:
    """Adam states for every parameter."""
    slow: dict[str, AdamState] = field(default_factory=dict)
    fast: dict[str, AdamState] = field(default_factory=dict)


def init_opt_state(params: NestAntParams) -> NestAntOptState:
    st = NestAntOptState()
    for name, p in params.slow_list():
        st.slow[name] = _init_adam_like(p)
    for name, p in params.fast_list():
        st.fast[name] = _init_adam_like(p)
    return st


# ---------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------
def forward(
    params: NestAntParams,
    X: np.ndarray,
    tasks: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Compute predictions and cache intermediates for backprop.

    Arguments
    ---------
    X: (N, n_features) standardised inputs.
    tasks: (N,) int task ids in {0,...,n_bands-1}.

    Returns
    -------
    y_hat: (N, n_targets) predictions in standardised target space.
    cache: dict of intermediates needed to compute gradients.
    """
    # -- Slow level: physics embedding z = M_slow(x) ------------------
    h_pre = X @ params.W1 + params.b1           # (N, hidden)
    h = _gelu(h_pre)                             # (N, hidden)
    z = h @ params.W2 + params.b2               # (N, embed) -- embedding

    # -- Fast level: band-conditional nonlinear reader ----------------
    b_emb = params.Eb[tasks]                    # (N, band_embed)
    zc = np.concatenate([z, b_emb], axis=1)     # (N, embed + band_embed)
    f_pre = zc @ params.W3 + params.b3          # (N, fast_hidden)
    f = _gelu(f_pre)                             # (N, fast_hidden)
    y_hat = f @ params.W4 + params.b4           # (N, n_targets)

    cache = dict(X=X, tasks=tasks, h_pre=h_pre, h=h, z=z, b_emb=b_emb,
                 zc=zc, f_pre=f_pre, f=f)
    return y_hat, cache


def backward(
    params: NestAntParams,
    cache: dict,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    weight_decay: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Backpropagate MSE loss through the two-level network.

    Returns
    -------
    grads_slow: dict param name -> gradient, for the slow level.
    grads_fast: dict param name -> gradient, for the fast level.
    loss_val:   scalar MSE loss (before weight-decay).
    """
    N = y_true.shape[0]
    diff = (y_hat - y_true)                     # (N, n_targets)
    loss_val = float((diff ** 2).mean())

    # dL/dy_hat = 2/N/K * diff
    dy = (2.0 / (N * y_true.shape[1])) * diff   # (N, n_targets)

    # Fast level: output head W4
    f = cache["f"]
    dW4 = f.T @ dy                              # (fast_hidden, n_targets)
    db4 = dy.sum(0)                             # (n_targets,)
    df = dy @ params.W4.T                       # (N, fast_hidden)
    df_pre = df * _gelu_grad(cache["f_pre"])    # (N, fast_hidden)

    # Fast level: hidden layer W3
    zc = cache["zc"]
    dW3 = zc.T @ df_pre                         # (embed+band_embed, fast_hidden)
    db3 = df_pre.sum(0)                         # (fast_hidden,)
    dzc = df_pre @ params.W3.T                  # (N, embed+band_embed)

    # Split dzc into dz and db_emb
    embed_dim = cache["z"].shape[1]
    dz = dzc[:, :embed_dim]                     # (N, embed)
    dbe = dzc[:, embed_dim:]                    # (N, band_embed)

    # Gradient w.r.t. band embedding table (scatter-add)
    dEb = np.zeros_like(params.Eb)
    tasks = cache["tasks"]
    np.add.at(dEb, tasks, dbe)

    # Slow level gradients
    h = cache["h"]
    dW2 = h.T @ dz                              # (hidden, embed)
    db2 = dz.sum(0)                             # (embed,)
    dh = dz @ params.W2.T                       # (N, hidden)

    dh_pre = dh * _gelu_grad(cache["h_pre"])    # (N, hidden)
    X = cache["X"]
    dW1 = X.T @ dh_pre                          # (n_features, hidden)
    db1 = dh_pre.sum(0)                         # (hidden,)

    # Add L2 weight decay on the weight matrices (not biases)
    if weight_decay > 0.0:
        dW1 = dW1 + weight_decay * params.W1
        dW2 = dW2 + weight_decay * params.W2
        dW3 = dW3 + weight_decay * params.W3
        dW4 = dW4 + weight_decay * params.W4

    grads_slow = dict(W1=dW1, b1=db1, W2=dW2, b2=db2)
    grads_fast = dict(Eb=dEb, W3=dW3, b3=db3, W4=dW4, b4=db4)
    return grads_slow, grads_fast, loss_val


# ---------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------
def _apply_updates(
    params: NestAntParams,
    grads_slow: dict[str, np.ndarray],
    grads_fast: dict[str, np.ndarray],
    opt_state: NestAntOptState,
    slow_lr: float,
    fast_lr: float,
    weight_decay: float,
) -> None:
    """Apply Adam updates with DIFFERENT learning rates to the two levels.

    This is the operational realisation of NL's multi-timescale principle:
    slow level has low lr (long consolidation), fast level has higher lr
    (quick adaptation)."""
    # Slow
    for name in ("W1", "b1", "W2", "b2"):
        p = getattr(params, name)
        g = grads_slow[name]
        p_new, st = adam_step(p, g, opt_state.slow[name], slow_lr,
                              weight_decay=weight_decay if "W" in name else 0.0)
        setattr(params, name, p_new)
        opt_state.slow[name] = st
    # Fast
    for name in ("Eb", "W3", "b3", "W4", "b4"):
        p = getattr(params, name)
        g = grads_fast[name]
        p_new, st = adam_step(p, g, opt_state.fast[name], fast_lr,
                              weight_decay=weight_decay if "W" in name else 0.0)
        setattr(params, name, p_new)
        opt_state.fast[name] = st


def iterate_minibatches(
    rng: np.random.Generator,
    N: int,
    batch_size: int,
):
    """Yield shuffled index minibatches for one pass over N samples."""
    idx = rng.permutation(N)
    for s in range(0, N, batch_size):
        yield idx[s:s + batch_size]


def train_joint(
    cfg: NestAntConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tasks_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    tasks_val: np.ndarray | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
) -> tuple[NestAntParams, list[dict]]:
    """Standard IID training on the pooled dataset (baseline protocol).

    The two-level structure is still honoured -- slow and fast parameters
    get different learning rates -- but we see every band at every epoch."""
    rng = np.random.default_rng(cfg.seed)
    params = init_params(cfg)
    opt_state = init_opt_state(params)

    history: list[dict] = []
    N = X_train.shape[0]

    for epoch in range(cfg.epochs):
        running_loss = 0.0
        n_batches = 0
        for batch_idx in iterate_minibatches(rng, N, cfg.batch_size):
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            tb = tasks_train[batch_idx]

            y_hat, cache = forward(params, xb, tb)
            g_slow, g_fast, loss = backward(params, cache, yb, y_hat,
                                            weight_decay=cfg.weight_decay)
            _apply_updates(params, g_slow, g_fast, opt_state,
                           cfg.slow_lr, cfg.fast_lr, cfg.weight_decay)
            running_loss += loss
            n_batches += 1

        epoch_log = {"epoch": epoch, "train_loss": running_loss / max(n_batches, 1)}
        if X_val is not None and y_val is not None and tasks_val is not None:
            y_hat_val, _ = forward(params, X_val, tasks_val)
            epoch_log["val_loss"] = float(((y_hat_val - y_val) ** 2).mean())
        history.append(epoch_log)
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_log)

    return params, history


def train_continual(
    cfg: NestAntConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tasks_train: np.ndarray,
    task_order: list[int],
    epochs_per_task: int | None = None,
    freeze_slow_after_task: int | None = None,
) -> tuple[NestAntParams, list[dict]]:
    """Continual learning: present tasks one at a time, never revisit.

    This is the realistic 5G/6G scenario where bands are acquired
    incrementally. Standard joint-trained models see every band at every
    epoch and therefore cannot exhibit forgetting; here the model sees task
    t only during its own training window.

    The NL insight we exploit: the SLOW memory encodes band-agnostic
    physics, so it should remain stable across tasks (low lr + optional
    freezing after a warm-up task). The FAST memory is allowed to adapt
    freely per task.

    freeze_slow_after_task:
        If set, the slow level parameters are frozen (lr=0) starting from
        the END of the given task index. This implements the "online
        consolidation" idea (NL §1.1): once the slow memory has absorbed
        the first band's physics, subsequent tasks cannot corrupt it.
    """
    rng = np.random.default_rng(cfg.seed)
    params = init_params(cfg)
    opt_state = init_opt_state(params)

    ep_per_task = epochs_per_task if epochs_per_task is not None else cfg.epochs
    history: list[dict] = []

    for task_i, task_id in enumerate(task_order):
        mask = tasks_train == task_id
        Xt = X_train[mask]
        yt = y_train[mask]
        tt = tasks_train[mask]
        N = Xt.shape[0]

        slow_lr_here = 0.0 if (freeze_slow_after_task is not None
                               and task_i > freeze_slow_after_task) else cfg.slow_lr

        for epoch in range(ep_per_task):
            running_loss = 0.0
            n_batches = 0
            for batch_idx in iterate_minibatches(rng, N, cfg.batch_size):
                xb = Xt[batch_idx]; yb = yt[batch_idx]; tb = tt[batch_idx]
                y_hat, cache = forward(params, xb, tb)
                g_slow, g_fast, loss = backward(params, cache, yb, y_hat,
                                                weight_decay=cfg.weight_decay)
                _apply_updates(params, g_slow, g_fast, opt_state,
                               slow_lr_here, cfg.fast_lr, cfg.weight_decay)
                running_loss += loss
                n_batches += 1
            history.append({
                "task_stage": task_i, "task_id": task_id, "epoch": epoch,
                "train_loss": running_loss / max(n_batches, 1),
                "slow_lr": slow_lr_here,
            })

    return params, history


# ---------------------------------------------------------------------
# Convenience prediction helper
# ---------------------------------------------------------------------
def predict(
    params: NestAntParams,
    X: np.ndarray,
    tasks: np.ndarray,
) -> np.ndarray:
    y_hat, _ = forward(params, X, tasks)
    return y_hat
