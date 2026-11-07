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
