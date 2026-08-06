"""
baselines.py — Baseline surrogate models for comparison.

We include:
  * PlainMLP: an ordinary single-level MLP (same total capacity as NestAnt
    but no slow/fast separation) -- this is the fairest apples-to-apples
    control to show that the two-level NL structure *itself* is what helps.
  * Gaussian Process Regression (sklearn GPR, isotropic RBF + WhiteKernel)
  * Random Forest
  * XGBoost
  * Fourier-Series Regression (FSR) -- reproducing Aghoutane et al. 2026
  * Sum-of-Sines Regression (SSR) -- reproducing Aghoutane et al. 2026

All models implement the same .fit(X, y) / .predict(X) API on standardised
numpy arrays for direct head-to-head comparison.

Franklin Open paper:  Aghoutane, El Ghzaoui, El Faylali, Bossoufi,
    "Efficient 5G MIMO Antenna Development Using Machine Learning-Driven
    Parameter Prediction", Franklin Open 2026, doi:10.1016/j.fraope.2026.100608.
Their FSR/SSR are nonlinear curve-fits; we reproduce the spirit of their
approach by fitting the same functional forms in standardised feature space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


# =====================================================================
# Interface
# =====================================================================
class Surrogate(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# =====================================================================
# Plain MLP (ablation control for NestAnt)
# =====================================================================
def _gelu(x): return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715*x**3)))
def _gelu_grad(x):
    s = np.sqrt(2.0/np.pi); u = s * (x + 0.044715*x**3); t = np.tanh(u)
    du = s * (1.0 + 3.0*0.044715*x**2)
    return 0.5*(1.0 + t) + 0.5*x*(1.0 - t**2)*du


@dataclass
class PlainMLPConfig:
    n_in: int
    n_out: int
    hidden: int = 96        # chosen so that total params ~ NestAnt's
    lr: float = 1e-3
    epochs: int = 80
    batch_size: int = 256
    weight_decay: float = 1e-4
    seed: int = 0


class PlainMLP:
    """A deliberately-ablative single-level MLP.

    Uses one global learning rate -- no slow/fast split -- so that any
    advantage of NestAnt over this model is attributable to the nested-
    learning structure, not to model capacity."""

    def __init__(self, cfg: PlainMLPConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        lim1 = np.sqrt(6.0 / (cfg.n_in + cfg.hidden))
        lim2 = np.sqrt(6.0 / (cfg.hidden + cfg.hidden))
        lim3 = np.sqrt(6.0 / (cfg.hidden + cfg.n_out))
        self.W1 = rng.uniform(-lim1, lim1, (cfg.n_in, cfg.hidden))
        self.b1 = np.zeros(cfg.hidden)
        self.W2 = rng.uniform(-lim2, lim2, (cfg.hidden, cfg.hidden))
        self.b2 = np.zeros(cfg.hidden)
        self.W3 = rng.uniform(-lim3, lim3, (cfg.hidden, cfg.n_out))
        self.b3 = np.zeros(cfg.n_out)
        # Adam states
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k))
                   for k in ("W1","b1","W2","b2","W3","b3")}
        self._v = {k: np.zeros_like(getattr(self, k))
                   for k in ("W1","b1","W2","b2","W3","b3")}
        self._t = 0

    def _forward(self, X):
        h1_pre = X @ self.W1 + self.b1
        h1 = _gelu(h1_pre)
        h2_pre = h1 @ self.W2 + self.b2
        h2 = _gelu(h2_pre)
        y_hat = h2 @ self.W3 + self.b3
        return y_hat, (X, h1_pre, h1, h2_pre, h2)

    def _backward(self, cache, y, y_hat):
        X, h1_pre, h1, h2_pre, h2 = cache
        N, K = y.shape
        dy = (2.0 / (N * K)) * (y_hat - y)
        dW3 = h2.T @ dy; db3 = dy.sum(0); dh2 = dy @ self.W3.T
        dh2_pre = dh2 * _gelu_grad(h2_pre)
        dW2 = h1.T @ dh2_pre; db2 = dh2_pre.sum(0); dh1 = dh2_pre @ self.W2.T
        dh1_pre = dh1 * _gelu_grad(h1_pre)
        dW1 = X.T @ dh1_pre; db1 = dh1_pre.sum(0)
        wd = self.cfg.weight_decay
        dW1 = dW1 + wd * self.W1; dW2 = dW2 + wd * self.W2; dW3 = dW3 + wd * self.W3
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)

    def _adam_apply(self, grads):
        self._t += 1
        lr = self.cfg.lr
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in grads:
            g = grads[k]
            self._m[k] = b1 * self._m[k] + (1-b1) * g
            self._v[k] = b2 * self._v[k] + (1-b2) * g*g
            m_hat = self._m[k] / (1 - b1**self._t)
            v_hat = self._v[k] / (1 - b2**self._t)
            setattr(self, k, getattr(self, k) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def fit(self, X, y):
        rng = np.random.default_rng(self.cfg.seed)
        N = X.shape[0]
        for _ in range(self.cfg.epochs):
            perm = rng.permutation(N)
            for s in range(0, N, self.cfg.batch_size):
                idx = perm[s:s+self.cfg.batch_size]
                xb, yb = X[idx], y[idx]
                y_hat, cache = self._forward(xb)
                grads = self._backward(cache, yb, y_hat)
                self._adam_apply(grads)
        return self

    def predict(self, X):
        y_hat, _ = self._forward(X)
        return y_hat


# =====================================================================
# GPR wrapper (sklearn)
# =====================================================================
class GPRWrapper:
    """Multi-output GPR via independent regressors (sklearn trick).
    GPR scales poorly; we subsample the training set."""
    def __init__(self, max_train: int = 2000, seed: int = 0,
                 length_scale: float = 1.0, alpha: float = 1e-3):
        self.max_train = max_train
        self.seed = seed
        self.length_scale = length_scale
        self.alpha = alpha
        self.models = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            idx = rng.choice(len(X), size=self.max_train, replace=False)
            X_s, y_s = X[idx], y[idx]
        else:
            X_s, y_s = X, y
        K = y_s.shape[1]
        self.models = []
        for j in range(K):
            kernel = (ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
                      * RBF(length_scale=self.length_scale,
                            length_scale_bounds=(1e-2, 1e2))
                      + WhiteKernel(noise_level=0.1,
                                    noise_level_bounds=(1e-5, 1e1)))
            gp = GaussianProcessRegressor(kernel=kernel,
                                          alpha=self.alpha,
                                          normalize_y=True,
                                          n_restarts_optimizer=2,
                                          random_state=self.seed + j)
            gp.fit(X_s, y_s[:, j])
            self.models.append(gp)
        return self

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.stack(preds, axis=1)


# =====================================================================
# Random Forest wrapper
# =====================================================================
class RFWrapper:
    def __init__(self, n_estimators: int = 300, max_depth: int | None = None,
                 seed: int = 0, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# =====================================================================
# XGBoost wrapper (one model per target, multi-output)
# =====================================================================
class XGBWrapper:
    def __init__(self, n_estimators: int = 400, max_depth: int = 6,
                 learning_rate: float = 0.08, seed: int = 0):
        if not HAS_XGB:
            raise ImportError("xgboost not available")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.models = None

    def fit(self, X, y):
        self.models = []
        for j in range(y.shape[1]):
            m = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed + j,
                verbosity=0,
                tree_method="hist",
                n_jobs=-1,
            )
            m.fit(X, y[:, j])
            self.models.append(m)
        return self

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.stack(preds, axis=1)


# =====================================================================
# Fourier-Series Regression (FSR) and Sum-of-Sines Regression (SSR)
# Re-implemented per Aghoutane et al. 2026 Eqs. for Fourier/Sum-of-Sine.
# Both are small nonlinear curve-fits originally defined over a 1-D frequency
# axis. To make them usable on our multi-feature regression we fit them
# target-by-target on a PROJECTION of X (leading principal component),
# preserving the spirit of the original 1-D fits.
# =====================================================================
from scipy.optimize import least_squares


def _fsr_model(theta, x, order=6):
    a0 = theta[0]
    w = theta[1]
    out = np.full_like(x, a0, dtype=float)
    for n in range(order):
        a = theta[2 + 2*n]
        b = theta[3 + 2*n]
        out = out + a * np.cos((n+1)*w*x) + b * np.sin((n+1)*w*x)
    return out


def _ssr_model(theta, x, order=6):
    out = np.zeros_like(x, dtype=float)
    for n in range(order):
        a = theta[3*n + 0]
        b = theta[3*n + 1]
        c = theta[3*n + 2]
        out = out + a * np.sin(b * x + c)
    return out


class FSRWrapper:
    """Fourier-Series Regression per target, on the 1st PC of X."""
    def __init__(self, order: int = 6, seed: int = 0):
        self.order = order; self.seed = seed
        self.pc = None; self.thetas = None; self.n_out = None

    def _project(self, X):
        # Normalised 1st PC of the training X
        Xc = X - self.x_mean
        return Xc @ self.pc

    def fit(self, X, y):
        self.x_mean = X.mean(0)
        Xc = X - self.x_mean
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.pc = vt[0]  # (n_features,)
        x1 = Xc @ self.pc
        self.n_out = y.shape[1]
        self.thetas = []
        rng = np.random.default_rng(self.seed)
        for j in range(self.n_out):
            theta0 = np.zeros(2 + 2*self.order)
            theta0[0] = y[:, j].mean()
            theta0[1] = 1.0 + 0.1*rng.standard_normal()
            def resid(th, x=x1, yj=y[:, j]):
                return _fsr_model(th, x, self.order) - yj
            res = least_squares(resid, theta0, max_nfev=4000)
            self.thetas.append(res.x)
        return self

    def predict(self, X):
        x1 = self._project(X)
        cols = [_fsr_model(self.thetas[j], x1, self.order) for j in range(self.n_out)]
        return np.stack(cols, axis=1)


class SSRWrapper:
    """Sum-of-Sines Regression per target, on the 1st PC of X."""
    def __init__(self, order: int = 6, seed: int = 0):
        self.order = order; self.seed = seed
        self.pc = None; self.thetas = None; self.n_out = None

    def fit(self, X, y):
        self.x_mean = X.mean(0)
        Xc = X - self.x_mean
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.pc = vt[0]
        x1 = Xc @ self.pc
        self.n_out = y.shape[1]
        self.thetas = []
        rng = np.random.default_rng(self.seed)
        for j in range(self.n_out):
            theta0 = rng.standard_normal(3*self.order) * 0.5
            def resid(th, x=x1, yj=y[:, j]):
                return _ssr_model(th, x, self.order) - yj
            res = least_squares(resid, theta0, max_nfev=4000)
            self.thetas.append(res.x)
        return self

    def predict(self, X):
        Xc = X - self.x_mean
        x1 = Xc @ self.pc
        cols = [_ssr_model(self.thetas[j], x1, self.order) for j in range(self.n_out)]
        return np.stack(cols, axis=1)
