"""Fidelity quantum kernel  k(x, x') = |<psi(x)|psi(x')>|^2.
Owner: Farrukh (week 1).

Three versions of the same kernel:
  fidelity_kernel_sampled  -- main results. One compute-uncompute circuit
                              U(x')^dag U(x) per PAIR, sampled with shots on
                              AerSimulator, exactly as NB11 and hardware do.
  fidelity_kernel_exact    -- ideal reference (statevectors, no shot noise).
                              Used for g_lambda and for checking the sampled one.
  product_kernel_closed    -- closed form of the 1-layer kernel, where the CNOT
                              block cancels: prod_i cos^2((a_i - b_i) / 2).
Inputs A, B are rotation angles (alpha * x) from common.make_preprocessor.
"""
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import unitary_overlap
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from .common import SHOTS

SIM = AerSimulator(max_parallel_experiments=0, max_parallel_threads=0)
_OV_CACHE = {}

ONE_Q = ["rx", "ry", "rz", "h", "x", "y", "z", "sx", "u"]
TWO_Q = ["cx", "cz"]


def overlap_circuit(enc):
    """Transpiled, still-parameterised U(p1)^dag U(p2) with measurements."""
    key = id(enc)
    if key not in _OV_CACHE:
        ov = unitary_overlap(enc, enc)
        ov.measure_all()
        _OV_CACHE[key] = transpile(ov, SIM)
    return _OV_CACHE[key]


def _run_pairs(tov, pairs, shots, seed):
    """Probability of the all-zero outcome for each (a, b) pair (NB11 code)."""
    if not pairs:
        return np.zeros(0)
    cols = {}
    for prm in tov.parameters:                  # named p1[k] / p2[k]
        k, side = int(prm.name.split("[")[1][:-1]), prm.name[1]
        cols[prm] = [float(ab[0 if side == "1" else 1][k]) for ab in pairs]
    res = SIM.run(tov, shots=shots, parameter_binds=[cols],
                  seed_simulator=seed).result()
    return np.array([res.data(j)["counts"].get("0x0", 0) / shots
                     for j in range(len(pairs))])


def fidelity_kernel_sampled(enc, A, B=None, shots=SHOTS, seed=0):
    """Sampled fidelity kernel.

    B=None: symmetric training kernel (upper triangle sampled, diagonal = 1,
    since k(x, x) = 1 exactly). Otherwise returns the len(A) x len(B) block,
    e.g. A = test angles, B = training angles.
    """
    tov = overlap_circuit(enc)
    A = np.asarray(A, float)
    if B is None:
        idx = [(i, j) for i in range(len(A)) for j in range(i + 1, len(A))]
        vals = _run_pairs(tov, [(A[i], A[j]) for i, j in idx], shots, seed)
        K = np.eye(len(A))
        for (i, j), v in zip(idx, vals):
            K[i, j] = K[j, i] = v
        return K
    B = np.asarray(B, float)
    return _run_pairs(tov, [(a, b) for a in A for b in B],
                      shots, seed).reshape(len(A), len(B))


def _states(enc, A):
    return np.array([Statevector(enc.assign_parameters(a)).data for a in A])


def fidelity_kernel_exact(enc, A, B=None):
    """Ideal fidelity kernel from statevectors (no shots, no noise)."""
    SA = _states(enc, np.asarray(A, float))
    SB = SA if B is None else _states(enc, np.asarray(B, float))
    return np.abs(SA.conj() @ SB.T) ** 2


def product_kernel_closed(A, B=None):
    """Closed form of the 1-layer kernel: prod_i cos^2((a_i - b_i) / 2)."""
    A = np.asarray(A, float)
    B = A if B is None else np.asarray(B, float)
    return np.prod(np.cos((A[:, None, :] - B[None, :, :]) / 2) ** 2, axis=2)


def overlap_resources(enc):
    """Qubits, depth and two-qubit gate count of one overlap circuit,
    in the NB11 basis (rx, ry, rz, h, ..., cx, cz)."""
    import numpy as _np
    ov = unitary_overlap(enc.assign_parameters(_np.zeros(enc.num_parameters)),
                         enc.assign_parameters(_np.ones(enc.num_parameters)))
    tc = transpile(ov, basis_gates=ONE_Q + TWO_Q, optimization_level=0)
    n2 = sum(1 for ins in tc.data if ins.operation.name in TWO_Q)
    return dict(qubits=enc.num_qubits, depth=tc.depth(), two_qubit_gates=n2)
