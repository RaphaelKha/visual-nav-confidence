"""Sanity check: with the true visual noise variance (oracle), the filter must be consistent."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kf import run_filter  # noqa: E402
from sim import generate  # noqa: E402

BURN = 100  # 10 s warm-up (in visual measurements)


def test_oracle_filter_is_consistent():
    nis, nees = [], []
    for seed in range(5):
        s = generate(seed, T=200)
        r = run_filter(s, lambda k: s.sigma[k] ** 2)
        nis.append(r["nis"][BURN:])
        nees.append(r["nees"][BURN:])
    nis, nees = np.concatenate(nis), np.concatenate(nees)
    assert 1.8 < nis.mean() < 2.2, nis.mean()      # ideal: 2
    assert 3.4 < nees.mean() < 4.7, nees.mean()    # ideal: 4
