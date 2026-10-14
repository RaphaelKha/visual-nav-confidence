"""Learned confidence: predict the variance of the visual error from observable indicators.

Model: log sigma^2 = w . phi(features). Trained by Gaussian negative log-likelihood (NLL)
on the observed visual error (visual position - ground truth, on training flights).
The NLL penalizes overconfidence and underconfidence equally: the model is forced to be honest.
"""
import numpy as np
from scipy.optimize import minimize


class LogVarModel:
    def __init__(self, l2=1e-3, lv_clip=(-4.0, 8.0)):
        self.l2, self.lv_clip = l2, lv_clip      # lv_clip: numerical guard on the predicted log-variance

    def _phi(self, feats):
        f = (feats - self.mu) / self.sd
        return np.column_stack([np.ones(len(f)), f])

    def fit(self, feats, err):
        """feats: (K,d) indicators; err: (K,m) visual error (m), isotropic variance per axis (m = 2 or 3)."""
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-9
        Phi = self._phi(feats)
        e2 = (err**2).sum(1)                                # |e|^2, all axes
        m = err.shape[1]

        def nll(w):
            lv = Phi @ w
            val = 0.5 * np.mean(m * lv + e2 * np.exp(-lv)) + self.l2 * np.sum(w[1:] ** 2)
            g = Phi.T @ (0.5 * (m - e2 * np.exp(-lv))) / len(e2)
            g[1:] += 2 * self.l2 * w[1:]
            return val, g

        w0 = np.zeros(Phi.shape[1])
        w0[0] = np.log(e2.mean() / m)
        self.w = minimize(nll, w0, jac=True, method="L-BFGS-B").x
        return self

    def predict_var(self, feats):
        lv = np.clip(self._phi(feats) @ self.w, *self.lv_clip)
        return np.exp(lv)
