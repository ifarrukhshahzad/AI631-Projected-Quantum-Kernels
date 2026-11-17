"""Geometric difference between a classical and a quantum kernel matrix.
Owner: Haider (week 2).

Ideal definition (Huang et al. 2021, Eq. 5), kernels normalised to trace N:
    g(K_C || K_Q) = sqrt( || sqrt(K_Q) K_C^{-1} sqrt(K_Q) ||_inf )
Classical kernel matrices are often nearly singular, so, as in Huang et al.
(Appendix F.3, Eqs. F19-F20), we use the regularised version
    g_gen(lam) = sqrt( || sqrt(K_Q) sqrt(K_C) (K_C + lam I)^{-2} sqrt(K_C) sqrt(K_Q) || )
    g_tra(lam) = lam * sqrt( || sqrt(K_Q) (K_C + lam I)^{-2} sqrt(K_Q) || )
g_gen reduces to the ideal g when lam = 0. Huang et al. pick the largest lam
with g_tra <= 0.045, but with only 2-8 features our classical kernel matrices
are effectively low-rank and that rule is rarely satisfiable (g then blows up
with 1/sqrt(lam)). We therefore report g_gen at ONE fixed, pre-declared
lam = LAM_FIXED = 0.1 (10% of the mean eigenvalue of the trace-normalised K_C)
for every comparison, plus the lam curve and whether Huang's rule is feasible.
lam is never chosen by accuracy. Square roots are principal square roots by
eigendecomposition, with tiny negative eigenvalues (rounding) clipped to 0.
"""
import numpy as np

LAMBDA_GRID = tuple(10.0 ** np.arange(-6, 1.01, 0.25))   # 1e-6 ... 10
G_TRA_MAX = 0.045
LAM_FIXED = 0.1


def normalise_trace(K):
    """Symmetrise K and scale it so that trace(K) = N (number of samples)."""
    K = np.asarray(K, float)
    K = (K + K.T) / 2
    return K * len(K) / np.trace(K)


def psd_sqrt(K):
    """Principal square root of a symmetric PSD matrix."""
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def _parts(KC, KQ):
    w, V = np.linalg.eigh(KC)
    w = np.clip(w, 0, None)
    return w, V, psd_sqrt(KQ)


def g_terms(KC, KQ, lam, parts=None):
    """(g_gen, g_tra) for already trace-normalised kernels."""
    w, V, SQ = parts if parts is not None else _parts(KC, KQ)
    A = SQ @ V                                   # sqrt(K_Q) in K_C's eigenbasis
    gen = A * (np.sqrt(w) / (w + lam))           # sqrt(K_Q) sqrt(K_C) (K_C + lam I)^-1
    tra = A * (1.0 / (w + lam))                  # sqrt(K_Q) (K_C + lam I)^-1
    g_gen = np.sqrt(np.linalg.norm(gen @ gen.T, 2))
    g_tra = lam * np.sqrt(np.linalg.norm(tra @ tra.T, 2))
    return float(g_gen), float(g_tra)


def geometric_difference(KC, KQ, lam=None):
    """Return (g_gen, g_tra) at lam (default LAM_FIXED) for raw kernels on the
    SAME samples. Both kernels are trace-normalised first."""
    lam = LAM_FIXED if lam is None else lam
    KC, KQ = normalise_trace(KC), normalise_trace(KQ)
    return g_terms(KC, KQ, lam)


def huang_lambda(KC, KQ):
    """Largest lam on LAMBDA_GRID with g_tra <= G_TRA_MAX (Huang et al.'s rule),
    or None if no grid value satisfies it. Reported only."""
    KC, KQ = normalise_trace(KC), normalise_trace(KQ)
    parts = _parts(KC, KQ)
    ok = [lam for lam in LAMBDA_GRID if g_terms(KC, KQ, lam, parts)[1] <= G_TRA_MAX]
    return max(ok) if ok else None


def g_sensitivity(KC, KQ, lams=(1e-3, 1e-2, 1e-1, 1.0)):
    """g_gen for a few fixed lambdas (reported only, never used to choose)."""
    KC, KQ = normalise_trace(KC), normalise_trace(KQ)
    parts = _parts(KC, KQ)
    return {lam: g_terms(KC, KQ, lam, parts)[0] for lam in lams}


def top_direction(KC, KQ, lam):
    """Engineered-label targets: top eigenvector v of
    sqrt(K_Q) sqrt(K_C) (K_C + lam I)^-2 sqrt(K_C) sqrt(K_Q)  (the g_gen matrix)
    and y = sqrt(K_Q) v. Kernels must already be trace-normalised."""
    w, V, SQ = _parts(KC, KQ)
    gen = (SQ @ V) * (np.sqrt(w) / (w + lam))
    M = gen @ gen.T
    _, U = np.linalg.eigh((M + M.T) / 2)
    return SQ @ U[:, -1]
