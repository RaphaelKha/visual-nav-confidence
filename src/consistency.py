"""Filter consistency tests: is the filter honest about its uncertainty?

NIS  (Normalized Innovation Squared): innovation^T S^-1 innovation, chi2 with dim(z) = 2 dof.
                                      Computable in flight, no ground truth needed.
NEES (Normalized Estimation Error Squared): error^T P^-1 error, chi2 with dim(tested state) = 4 dof
                                      (position + velocity). Needs ground truth.
A consistent filter has E[NIS] = 2, E[NEES] = 4, and ~95 % of values below the 95 % chi2 quantile.
"""
import numpy as np
from scipy import stats


def nis_coverage(nis, dof=2, level=0.95):
    """Share of NIS values inside the two-sided `level` chi2 interval (ideal: `level`)."""
    lo, hi = stats.chi2.ppf([(1 - level) / 2, 1 - (1 - level) / 2], dof)
    return float(np.mean((nis >= lo) & (nis <= hi)))


def frac_below_95(values, dof):
    """Share of values below the 95 % chi2 quantile (ideal: 0.95). Sensitive to overconfidence."""
    return float(np.mean(values <= stats.chi2.ppf(0.95, dof)))
