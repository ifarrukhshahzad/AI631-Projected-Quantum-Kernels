"""Real IBM hardware run of the projected-kernel features (week 3).
Owner: Farrukh.

Only the projected kernel runs on hardware: it needs 3 circuits per image
(Z, X, Y bases), while the fidelity kernel needs one circuit per PAIR of
images (6,360 per seed), far beyond the Open Plan budget (600 s / 28 days).

Design:
  * backend ibm_kingston (156-qubit Heron), the device copied by our noise
    model (src/noise.py), so noisy-simulator and hardware results are
    directly comparable;
  * same transpilation as the noise model (optimisation level 1, fixed
    transpiler seed) and the same physical qubits 0..n-1 (initial_layout);
  * same 2048 shots as the sampled and noisy runs;
  * 2 extra readout-calibration circuits (all qubits |0>, all qubits |1>),
    which give per-qubit measurement error rates for week 4 mitigation;
  * a dry run first executes the same transpiled circuits on an Aer
    simulator with the FakeKingston noise model (no quota used).
The raw counts are saved to disk before anything else is computed, so a
paid hardware run never has to be repeated.
"""
import json

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.executor_sampler import Sampler

from .common import SHOTS
from .projected import BASES, _basis_circuit

BACKEND_NAME = "ibm_kingston"
_TRANSPILE = dict(optimization_level=1, seed_transpiler=0)


def build_circuits(enc, A):
    """3 basis circuits per sample (order: sample-major, then X, Y, Z),
    followed by 2 readout-calibration circuits (all 0, all 1)."""
    n = enc.num_qubits
    circs = [_basis_circuit(enc, a, b) for a in np.asarray(A, float) for b in BASES]
    zero = QuantumCircuit(n)
    zero.measure_all()
    one = QuantumCircuit(n)
    one.x(range(n))
    one.measure_all()
    return circs + [zero, one]


def transpile_for(backend, circs, n_qubits):
    """Transpile to the device on physical qubits 0..n-1 (as the noise model)."""
    return transpile(circs, backend, initial_layout=list(range(n_qubits)), **_TRANSPILE)


def dry_run_counts(fake_backend, tcircs, shots=SHOTS, seed=0):
    """Dry run: the SAME transpiled circuits on an Aer simulator built from
    the fake backend's noise model (as notebook 06). Uses no IBM quota.
    Returns the counts list in the same order as the real run."""
    sim = AerSimulator.from_backend(fake_backend)
    res = sim.run(tcircs, shots=shots, seed_simulator=seed).result()
    return [res.get_counts(i) for i in range(len(tcircs))]


def submit(backend, tcircs, shots=SHOTS):
    """Submit without waiting; returns the job (save job.job_id() at once)."""
    return Sampler(mode=backend).run(tcircs, shots=shots)


def counts_from_result(res, n_circuits):
    return [res[i].data.meas.get_counts() for i in range(n_circuits)]


def features_from_counts(counts, n_samples, n_qubits, shots=SHOTS):
    """<X_k>, <Y_k>, <Z_k> per sample from the basis-circuit counts
    (same convention as projected.pauli_features_sampled).
    Returns array (n_samples, n_qubits, 3)."""
    out = np.zeros((n_samples, n_qubits, 3))
    for c in range(3 * n_samples):
        i, j = divmod(c, 3)
        for bits, cnt in counts[c].items():
            bits = bits.replace(" ", "")
            for k in range(n_qubits):
                out[i, k, j] += cnt if bits[n_qubits - 1 - k] == "0" else -cnt
    return out / shots


def readout_errors(counts_zero, counts_one, n_qubits, shots=SHOTS):
    """Per-qubit readout error rates from the two calibration circuits:
    p(1|0) from the all-0 circuit, p(0|1) from the all-1 circuit."""
    p10, p01 = np.zeros(n_qubits), np.zeros(n_qubits)
    for bits, cnt in counts_zero.items():
        for k in range(n_qubits):
            if bits[n_qubits - 1 - k] == "1":
                p10[k] += cnt
    for bits, cnt in counts_one.items():
        for k in range(n_qubits):
            if bits[n_qubits - 1 - k] == "0":
                p01[k] += cnt
    return p10 / shots, p01 / shots


def save_counts(path, job_id, backend_name, counts, meta):
    path.write_text(json.dumps(dict(job_id=job_id, backend=backend_name,
                                    meta=meta, counts=counts), indent=1))


def load_counts(path):
    d = json.loads(path.read_text())
    return d["job_id"], d["backend"], d["meta"], d["counts"]
