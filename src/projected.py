"""Projected quantum kernel: one-qubit reduced density matrices (1-RDMs).
Owner: Shams (week 1: 1-RDM features; week 2: kernel and tuning).

For each sample x and each qubit k we need the three Pauli expectation
values <X_k>, <Y_k>, <Z_k>. They fully determine the 1-RDM:
    rho_k(x) = (I + <X_k> X + <Y_k> Y + <Z_k> Z) / 2
Measurement bases (synopsis Sec. IV-A):
    Z: measure directly
    X: apply H, then measure
    Y: apply S^dagger, then H, then measure
One circuit per basis gives that Pauli for ALL qubits at once, so each
sample costs 3 circuits (not one per pair, as the fidelity kernel does).
For our ideal Ry-CNOT encoding the amplitudes are real, so <Y_k> = 0.
"""
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator

from .common import SHOTS

SIM = AerSimulator(max_parallel_experiments=0, max_parallel_threads=0)
BASES = ("X", "Y", "Z")

_I = np.eye(2, dtype=complex)
_PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def _pauli_label(n, k, p):
    """Qiskit Pauli label for P on qubit k (qubit 0 is the rightmost char)."""
    s = ["I"] * n
    s[n - 1 - k] = p
    return "".join(s)


def pauli_features_exact(enc, A):
    """Ideal <X_k>, <Y_k>, <Z_k> from statevectors.

    Returns array of shape (n_samples, n_qubits, 3), last axis = (X, Y, Z).
    """
    A = np.asarray(A, float)
    n = enc.num_qubits
    ops = [[Pauli(_pauli_label(n, k, p)) for p in BASES] for k in range(n)]
    out = np.zeros((len(A), n, 3))
    for i, a in enumerate(A):
        sv = Statevector(enc.assign_parameters(a))
        for k in range(n):
            for j in range(3):
                out[i, k, j] = sv.expectation_value(ops[k][j]).real
    return out


def _basis_circuit(enc, a, basis):
    qc = enc.assign_parameters(a)
    if basis == "X":
        qc.h(range(qc.num_qubits))
    elif basis == "Y":
        qc.sdg(range(qc.num_qubits))
        qc.h(range(qc.num_qubits))
    qc.measure_all()
    return qc


def pauli_features_sampled(enc, A, shots=SHOTS, seed=0, sim=None):
    """Shot-based <X_k>, <Y_k>, <Z_k>: 3 circuits per sample, as on hardware.

    <P_k> = P(bit k = 0) - P(bit k = 1) in the matching basis.
    `sim` lets week 2 pass a noisy AerSimulator; default is noise-free.
    Returns array of shape (n_samples, n_qubits, 3).
    """
    sim = SIM if sim is None else sim
    A = np.asarray(A, float)
    n = enc.num_qubits
    circs = [_basis_circuit(enc, a, b) for a in A for b in BASES]
    tcircs = transpile(circs, sim)
    res = sim.run(tcircs, shots=shots, seed_simulator=seed).result()
    out = np.zeros((len(A), n, 3))
    for c in range(len(circs)):
        i, j = divmod(c, 3)
        for bits, cnt in res.get_counts(c).items():
            bits = bits.replace(" ", "")
            for k in range(n):
                out[i, k, j] += cnt if bits[n - 1 - k] == "0" else -cnt
    return out / shots


def rdms_from_features(F):
    """1-RDMs from Pauli features: shape (n_samples, n_qubits, 2, 2)."""
    F = np.asarray(F, float)
    rho = np.broadcast_to(_I, F.shape[:2] + (2, 2)).copy()
    for j, p in enumerate(BASES):
        rho = rho + F[..., j, None, None] * _PAULI[p]
    return rho / 2


def rdm_checks(F):
    """Sanity checks of the 1-RDMs: trace = 1, Bloch length <= 1 (valid
    state), smallest eigenvalue >= 0 (exact features; sampled ones may dip
    slightly below 0 because of shot noise)."""
    rho = rdms_from_features(F)
    tr = np.trace(rho, axis1=-2, axis2=-1).real
    bloch = np.linalg.norm(F, axis=-1)
    eig = np.linalg.eigvalsh(rho)
    return dict(max_trace_error=float(np.abs(tr - 1).max()),
                max_bloch_length=float(bloch.max()),
                min_eigenvalue=float(eig.min()))


def projected_kernel(FA, FB=None, gamma=1.0):
    """Projected quantum kernel from Pauli features (week 2):
        k(x, x') = exp(-gamma * sum_k ||rho_k(x) - rho_k(x')||_F^2),
    with ||rho_k(x) - rho_k(x')||_F^2 = 0.5 * sum_P (<P_k>_x - <P_k>_x')^2.
    FA, FB: arrays of shape (n_samples, n_qubits, 3). FB=None gives the
    symmetric training kernel. Always PSD (Gaussian kernel on features).
    """
    FA = np.asarray(FA, float).reshape(len(FA), -1)
    FB = FA if FB is None else np.asarray(FB, float).reshape(len(FB), -1)
    d2 = 0.5 * ((FA[:, None, :] - FB[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


def median_gamma(F, scale=1.0):
    """Bandwidth gamma = scale / median off-diagonal squared 1-RDM distance,
    computed on TRAINING features only (scale-free choice of gamma)."""
    F = np.asarray(F, float).reshape(len(F), -1)
    d2 = 0.5 * ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[~np.eye(len(F), dtype=bool)])
    return float(scale / med) if med > 0 else float(scale)


# Tuning budget shared with every other model (synopsis: same budget for all)
ALPHAS = (1.0, 2.0)                               # encoding scale
GAMMA_SCALES = (0.5, 2.0)                         # gamma = scale / median squared distance
C_GRID = (0.001, 0.01, 0.1, 1, 10, 100, 1000)     # same C grid as the classical SVMs


def tune_projected(enc, prep, Xtr, ytr, Xte, yte, folds, seed, shots=SHOTS):
    """Tune and evaluate the projected kernel with the 28-setting budget
    (2 alphas x 2 gamma scales x 7 C), exactly as in notebook 03.

    prep: make_preprocessor(n_qubits, alpha) factory from common.py.
    Tuning uses inner CV on sampled training kernels only. Returns
    (result dict, exact training kernel, sampled training kernel).
    """
    from .common import cv_score_precomputed, evaluate_precomputed, mean_offdiag
    feats, best = {}, dict(score=-1)
    for a in ALPHAS:
        tf = prep(enc.num_qubits, a)(Xtr)            # fitted on training data only
        Atr, Ate = tf(Xtr), tf(Xte)
        Fs_tr = pauli_features_sampled(enc, Atr, shots=shots, seed=seed)
        Fs_te = pauli_features_sampled(enc, Ate, shots=shots, seed=seed + 1)
        feats[a] = (Atr, Ate, Fs_tr, Fs_te)
        for gs in GAMMA_SCALES:
            gam = median_gamma(Fs_tr, gs)
            K = projected_kernel(Fs_tr, gamma=gam)
            for C in C_GRID:
                s = cv_score_precomputed(K, ytr, folds, C)
                if s > best["score"]:
                    best = dict(score=s, alpha=a, gscale=gs, gamma=gam, C=C)
    Atr, Ate, Fs_tr, Fs_te = feats[best["alpha"]]
    gam = best["gamma"]
    K_s, Kt_s = projected_kernel(Fs_tr, gamma=gam), projected_kernel(Fs_te, Fs_tr, gamma=gam)
    Fe_tr, Fe_te = pauli_features_exact(enc, Atr), pauli_features_exact(enc, Ate)
    K_e, Kt_e = projected_kernel(Fe_tr, gamma=gam), projected_kernel(Fe_te, Fe_tr, gamma=gam)
    out = dict(best=best,
               sampled=evaluate_precomputed(K_s, Kt_s, ytr, yte, best["C"]),
               exact=evaluate_precomputed(K_e, Kt_e, ytr, yte, best["C"]),
               mean_K=mean_offdiag(K_s))
    return out, K_e, K_s
