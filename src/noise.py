"""Noise model: simulator copy of a real IBM device (week 2).
Owner: Shams.

Synopsis Sec. IV-C: "a backend-derived Qiskit Aer model". We use IBM's
published calibration snapshot of ibm_kingston (FakeKingston in
qiskit-ibm-runtime), the 156-qubit Heron device our week 1 test job ran on.
AerSimulator.from_backend() copies its gate errors, readout errors, T1/T2
and qubit connectivity. A fixed snapshot (not live calibration) keeps every
run reproducible and costs no IBM hardware time.

Circuits are transpiled to the device (native gates rz/sx/x/cz, physical
qubits, optimization level 1, seed_transpiler 0), so the noisy circuits are
the ones the hardware would actually run. Shots match the ideal runs (2048).
Noisy results include both shot noise and modelled device errors.
"""
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import unitary_overlap
from qiskit_aer import AerSimulator

from .common import SHOTS
from .projected import BASES, _basis_circuit

BACKEND_NAME = "ibm_kingston (FakeKingston snapshot)"
_TRANSPILE = dict(optimization_level=1, seed_transpiler=0)
_SIM = None


def noisy_simulator():
    """AerSimulator with the FakeKingston noise model (built once)."""
    global _SIM
    if _SIM is None:
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        _SIM = AerSimulator.from_backend(FakeKingston())
    return _SIM


def pauli_features_noisy(enc, A, shots=SHOTS, seed=0):
    """Noisy <X_k>, <Y_k>, <Z_k>: same 3 basis circuits per sample as
    projected.pauli_features_sampled, run on the device noise model.
    Returns array of shape (n_samples, n_qubits, 3)."""
    sim = noisy_simulator()
    A = np.asarray(A, float)
    n = enc.num_qubits
    circs = [_basis_circuit(enc, a, b) for a in A for b in BASES]
    res = sim.run(transpile(circs, sim, **_TRANSPILE), shots=shots,
                  seed_simulator=seed).result()
    out = np.zeros((len(A), n, 3))
    for c in range(len(circs)):
        i, j = divmod(c, 3)
        for bits, cnt in res.get_counts(c).items():
            bits = bits.replace(" ", "")
            for k in range(n):
                out[i, k, j] += cnt if bits[n - 1 - k] == "0" else -cnt
    return out / shots


def _noisy_overlap(enc):
    sim = noisy_simulator()
    ov = unitary_overlap(enc, enc)
    ov.measure_all()
    return transpile(ov, sim, **_TRANSPILE)


def _run_pairs_noisy(tov, pairs, shots, seed):
    cols = {}
    for prm in tov.parameters:                  # named p1[k] / p2[k]
        k, side = int(prm.name.split("[")[1][:-1]), prm.name[1]
        cols[prm] = [float(ab[0 if side == "1" else 1][k]) for ab in pairs]
    res = noisy_simulator().run(tov, shots=shots, parameter_binds=[cols],
                                seed_simulator=seed).result()
    return np.array([res.data(j)["counts"].get("0x0", 0) / shots
                     for j in range(len(pairs))])


def fidelity_kernel_noisy(enc, A, B=None, shots=SHOTS, seed=0):
    """Noisy fidelity-kernel estimate: probability of the all-zero outcome of
    U(x')^dag U(x) on the device noise model (under noise this is a circuit
    probability, not a fidelity of mixed states; synopsis Sec. IV-A).
    B=None: symmetric training kernel, upper triangle run, diagonal set to 1
    as in the ideal case (k(x, x) is not measured)."""
    tov = _noisy_overlap(enc)
    A = np.asarray(A, float)
    if B is None:
        idx = [(i, j) for i in range(len(A)) for j in range(i + 1, len(A))]
        vals = _run_pairs_noisy(tov, [(A[i], A[j]) for i, j in idx], shots, seed)
        K = np.eye(len(A))
        for (i, j), v in zip(idx, vals):
            K[i, j] = K[j, i] = v
        return K
    B = np.asarray(B, float)
    return _run_pairs_noisy(tov, [(a, b) for a in A for b in B],
                            shots, seed).reshape(len(A), len(B))


def noisy_resources(enc):
    """Depth, two-qubit gate count and physical qubits of the transpiled
    noisy circuits (overlap circuit and one projected basis circuit, the
    latter with generic angles 0.5 so no gate is optimised away)."""
    tov = _noisy_overlap(enc)
    tz = transpile(_basis_circuit(enc, np.full(enc.num_parameters, 0.5), "Z"),
                   noisy_simulator(), **_TRANSPILE)
    return dict(overlap_depth=tov.depth(), overlap_cz=int(tov.count_ops().get("cz", 0)),
                basis_depth=tz.depth(), basis_cz=int(tz.count_ops().get("cz", 0)),
                physical_qubits=list(tz.layout.final_index_layout()))
