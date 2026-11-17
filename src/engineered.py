"""Engineered labels that favour the quantum kernel (Huang et al. 2021, Sec. 3).
Owner: Haider (week 2).

Idea: keep the SAME images, but replace the T-shirt/Shirt labels with labels
that the quantum kernel finds easy and classical kernels find hard. Huang et
al. build them from the direction that maximises the geometric difference:
take the top eigenvector v of
    sqrt(K_Q) sqrt(K_C) (K_C + lam I)^-2 sqrt(K_C) sqrt(K_Q)
set y = sqrt(K_Q) v, and split at the median to get two balanced classes.
This is a positive control: if an advantage is built into the labels, our
pipeline (tuning, falsifier) should detect it.

Fixed generator (decided before any engineered-label accuracy is computed):
    quantum K_Q:   ideal projected kernel, L = 2, alpha = 1, gamma scale = 2
    classical K_C: sum of ALL 9 classical kernel settings in the tuning grids
                   (RBF gamma 0.01/0.1/1/10, polynomial degree 2-5, linear),
                   each trace-normalised, so the labels are hard for the whole
                   classical search space, not for one kernel only
    lam:           GEN_LAMBDAS = 0.1 (the lam used for g everywhere), 0.01,
                   0.001; all three are reported (smaller lam pushes the
                   labels further into directions K_C cannot represent)
Labels are generated on all 120 images of a seed (train + test) with their
own PCA fitted on those 120 images. This is only the labelling step, as in
Huang et al.; every model is then trained exactly as before (PCA fitted on
training images only, tuned on inner CV, scored once on test).
"""
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from .baselines import GRIDS
from .common import encoding_circuit, make_preprocessor
from .geometry import g_terms, normalise_trace, top_direction
from .projected import median_gamma, pauli_features_exact, projected_kernel

GEN_ALPHA = 1.0
GEN_GSCALE = 2.0
GEN_LAYERS = 2
GEN_LAMBDAS = (0.1, 0.01, 0.001)


def generator_kernels(X_all, n_qubits):
    """(K_C, K_Q), both trace-normalised, of the fixed label generator."""
    F = make_preprocessor(n_qubits, GEN_ALPHA)(X_all)(X_all)
    P = pauli_features_exact(encoding_circuit(n_qubits, GEN_LAYERS), F)
    KQ = projected_kernel(P, gamma=median_gamma(P, GEN_GSCALE))
    ks = [rbf_kernel(F, gamma=g) for g in GRIDS["rbf"]["gamma"]]
    ks += [polynomial_kernel(F, degree=d, gamma=1, coef0=1) for d in GRIDS["poly"]["degree"]]
    ks += [linear_kernel(F)]
    KC = sum(normalise_trace(k) for k in ks)
    return normalise_trace(KC), normalise_trace(KQ)


def engineered_labels(KC, KQ, lam):
    """Labels in {-1, +1} from trace-normalised generator kernels.

    Median split of y = sqrt(K_Q) v; ties broken by sample index, so the two
    classes are exactly balanced over all images. Also returns g_gen and
    g_tra of the generator pair at this lam.
    """
    y = top_direction(KC, KQ, lam)
    order = np.argsort(y, kind="stable")
    labels = np.ones(len(y), dtype=int)
    labels[order[: len(y) // 2]] = -1
    g_gen, g_tra = g_terms(KC, KQ, lam)
    return labels, dict(g_gen=g_gen, g_tra=g_tra)
