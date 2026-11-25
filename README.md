# Projected Quantum Kernels and the Geometric Test

**AI 631 Quantum Machine Learning, LUMS, Fall 2026, Project #6 (Group 6)**
Farrukh Shahzad (25280034) · Shams ul Arifeen (25280080) · Haider Nadeem (25280003)

We test the main idea of Huang et al., *"Power of data in quantum machine learning"* (Nature Communications 12, 2631, 2021), at small scale: the **geometric difference** $g$ between a classical and a quantum kernel says whether a quantum advantage is *possible* on a dataset. We compare a **fidelity** quantum kernel and a **projected** quantum kernel against tuned classical models on Fashion-MNIST (T-shirt/top vs. Shirt), at 2, 4, 6 and 8 qubits, over 5 random splits, in perfect simulation, under a realistic device noise model, and on the real IBM quantum computer `ibm_kingston`.

Every number in our reports comes from a file in `results/` written by one of the notebooks below.

---

## Results at a glance

Test accuracy, mean over 5 seeds (80 training and 40 test images per seed). $C^\ast$ = the classical SVM with the best inner-CV score. $\Delta A$ = quantum minus $C^\ast$. **Improvement rule (fixed in the synopsis):** mean $\Delta A$ over the 5 seeds must exceed its standard deviation.

**1. Real labels (T-shirt vs. Shirt): no quantum advantage.**

| qubits | best classical $C^\ast$ | fidelity kernel | projected kernel | $\Delta A$, best quantum vs. $C^\ast$ | $g$ (projected vs. $C^\ast$) |
|---|---|---|---|---|---|
| 2 | 0.795 | 0.830 | 0.785 | −0.010 | 3.02 |
| 4 | 0.820 | 0.790 | 0.720 | −0.055 | 2.95 |
| 6 | 0.825 | 0.790 | 0.750 | −0.035 | 3.56 |
| 8 | 0.840 | 0.810 | 0.740 | −0.030 | 3.19 |

$g \approx 3$ everywhere: the quantum kernels are geometrically different from the classical ones, so an advantage is *possible*, but the real labels do not use that difference. Across every group of settings we ran (real labels, engineered labels, noise model), $g$ and $\Delta A$ are essentially uncorrelated: $r$ between −0.04 and +0.21 (notebook 08).

**2. Engineered labels (Huang et al.'s construction): the projected kernel wins at 6 qubits only.**

| $\lambda$ | $\Delta A$ at n = 2 | n = 4 | **n = 6** | n = 8 |
|---|---|---|---|---|
| 0.1 | −0.035 | +0.010 | **+0.050 ✓** | +0.045 |
| 0.01 | +0.005 | +0.025 | **+0.075 ✓** | +0.055 |
| 0.001 | +0.035 | +0.035 | **+0.075 ✓** | +0.025 |

✓ = passes the improvement rule.

**3. Noise: the projected kernel is robust; the fidelity kernel is not.** Under the `ibm_kingston` noise model, accuracies change by at most 1.5 points. However, $g$ of the fidelity kernel moves by −9% to +13%, while $g$ of the projected kernel moves by under 1.5% (notebook 06).

**4. Does the 6-qubit advantage survive a real device?** (notebooks 10 and 11, engineered labels, $n=6$)

| $\lambda$ | ideal, 5 seeds | noise model, 5 seeds | noise model + readout mitigation | **real `ibm_kingston`, seed 0** |
|---|---|---|---|---|
| 0.1 | +0.050 ✓ | +0.045 ✓ | +0.040 ✓ | −0.050 (ideal for seed 0: +0.000) |
| 0.01 | +0.075 ✓ | +0.095 ✓ | +0.080 | **+0.075** (ideal: +0.125) |
| 0.001 | +0.075 ✓ | +0.060 ✓ | +0.070 ✓ | **+0.025** (ideal: +0.100) |

On the noise model, the advantage holds. On the real device it **shrinks but stays positive for 2 of 3 label sets**. The noise model did not predict this drop. Readout mitigation does not fix it, because the real readout errors are already ≤ 2%. This rests on one hardware seed and 40 test images (one image = 2.5 points), so it is a signal, not a proven effect.

---

## Repository layout

```
├── src/                  shared code, imported by every notebook
├── notebooks/            01-11, run in order (see below)
├── results/
│   ├── splits/           saved train/test indices for every seed (identical for everyone)
│   ├── tables/           one JSON per notebook, seed and qubit count
│   ├── kernels/          saved kernel matrices (.npy)
│   ├── features/         saved Pauli expectation values (.npy)
│   ├── hardware/         raw IBM counts, job IDs, noise-model dry runs
│   └── figures/          every figure in the reports (.png)
├── report/               IEEE LaTeX reports (week 1, week 2, ...)
├── synopsis/             project synopsis (PDF and LaTeX)
├── data/                 Fashion-MNIST, downloaded automatically (not committed)
└── requirements.txt
```

File names follow one rule: `results/<kind>/<name>_seed<seed>_q<qubits>.<ext>` (`src/common.result_path`).

---

## Setup

Python 3.13 on Windows, macOS or Linux. Main versions: Qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0, scikit-learn 1.9.1, numpy 2.5.4.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install pylatexenc          # only for the circuit drawings in notebook 09
```

**Data.** You do not need to download anything by hand. The first call to `get_split` downloads the four official Fashion-MNIST files (about 30 MB) from Zalando Research into `data/fashion/`.

**IBM account (only to submit new hardware jobs).** Not needed to reproduce our results: the hardware notebooks load the saved counts. To submit jobs yourself, save your account once in a terminal. **Never put the key in a file in this repository.**

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your key>", instance="<your CRN>")
```

---

## How to reproduce

Open the notebooks from the `notebooks/` folder and run them **in order** with *Run All*. Every random step has a fixed seed (data splits, inner CV folds, shot sampling, transpilation), so rerunning on the same computer gives the same numbers. The only exception is real hardware, whose raw counts are saved and reloaded instead of rerun. On a different computer, the simulator can shift an occasional shot count by one. In our check on Linux this changed one tied tuning choice: projected kernel, seed 0, $n=4$, which moves the $n=4$ projected mean from 0.720 to 0.735. Every other reported number matched. The saved files are the results we report.

| # | notebook | owner | what it does | main outputs | run time |
|---|---|---|---|---|---|
| 01 | `01_data_baselines` | Haider | Data, splits, classical baselines (logistic regression; SVMs with RBF, polynomial, linear kernels), $C^\ast$ | `tables/baselines_v2_*` | minutes |
| 02 | `02_fidelity_kernel` | Farrukh | Fidelity kernel: exact, sampled (2048 shots), 1-layer closed form; 28-setting tuning; shot study | `kernels/fidelity_*`, `tables/fidelity_v2_*`, `fidelity_vs_qubits.png`, `fidelity_shots.png` | long (sampled kernels: one circuit per pair of images) |
| 03 | `03_projected_kernel` | Shams | Projected kernel: 1-RDM features (exact and sampled), kernel, 28-setting tuning | `features/pauli_*`, `kernels/projected_*`, `tables/projected_v2_*` | minutes |
| 04 | `04_geometric_difference` | Haider | Geometric difference $g$ of each quantum kernel vs. $C^\ast$, λ sensitivity | `tables/geometry_*`, `g_vs_qubits_projected.png`, `deltaA_vs_g_original.png` | minutes |
| 05 | `05_engineered_dataset` | Haider | Engineered labels (positive control); all models retrained on them | `tables/engineered_lam*`, `best_classical_vs_quantum.png`, `deltaA_real_vs_engineered.png` | minutes |
| 06 | `06_noise_and_shots` | Shams | Both kernels on the `ibm_kingston` noise model; $g$ under noise; four-level comparison (section 6) | `tables/noise_*`, `kernels/fidelity_noisy_*`, `noise_accuracy_levels.png`, `noise_g_ideal_vs_noisy.png`, `four_level_features.png` | 5-10 min with saved kernels (40 min from scratch) |
| 07 | `07_hardware` | Farrukh | Real `ibm_kingston`, seed 0, $n=6$, real-label settings ($\alpha=2$) | `hardware/hw_counts_seed0_q6.json`, `tables/hardware_projected_*` | seconds (loads saved counts) |
| 08 | `08_final_results` | Shams | Best classical vs. best quantum, $\Delta A$ vs. $g$ across every setting | `tables/final_summary.json`, `final_best_classical_vs_best_quantum.png`, `final_deltaA_vs_g_all.png` | seconds |
| 09 | `09_circuit_diagrams` | Haider | Circuit diagrams of the encoding and both kernels, and the transpiled hardware circuit; cost table | `circuit_encoding.png`, `circuit_fidelity.png`, `circuit_projected.png`, `circuit_projected_transpiled_kingston.png` | under 1 min |
| 10 | `10_hardware_advantage` | Farrukh | Real `ibm_kingston` for the advantage test ($\alpha=1$), with noise-model dry run and quota guard | `hardware/hw_counts_a1_*`, `hardware/dryrun_counts_a1_*`, `features/pauli_hardware_a1_*` | under 1 min (loads saved counts) |
| 11 | `11_advantage_under_noise` | Shams | Engineered labels at $n=6$: ideal vs. noise model vs. real hardware, with and without readout mitigation | `tables/advantage_noise_*`, `advantage_under_noise.png` | about 1 min |

`NB11_capstone_starter.ipynb` is the course's starter notebook (Module 8), kept for reference. Our evaluation protocol follows it.

**Hardware notebooks without an IBM account.** Keep `RUN_HARDWARE = False` in section 1 of notebooks 07 and 10. They then load the saved counts and never contact IBM.

---

## Source code (`src/`)

| module | owner | contents | used by |
|---|---|---|---|
| `common.py` | Farrukh | data download and splits, leakage-free preprocessing (PCA + scaling fitted on training images only), encoding circuit, inner CV folds, precomputed-kernel SVM evaluation, file naming | every notebook |
| `fidelity.py` | Farrukh | fidelity kernel: compute-uncompute circuit, sampled and exact versions, 1-layer closed form | 02 |
| `hardware.py` | Farrukh | IBM hardware: circuit building (3 per image + 2 readout-calibration circuits), transpilation to physical qubits 0..n−1, dry run, submission, counts → features, readout errors, readout mitigation | 07, 09, 10, 11 |
| `projected.py` | Shams | projected kernel: Pauli features (exact, sampled), 1-RDMs and checks, kernel, median-$\gamma$ rule, 28-setting tuning | 03, 05, 06, 07, 09, 10, 11, and `engineered`, `noise`, `hardware` |
| `noise.py` | Shams | `FakeKingston` noise model, noisy Pauli features, noisy fidelity kernel, transpiled circuit sizes | 06, 09 |
| `baselines.py` | Haider | classical tuning grids and models, $C^\ast$, classical kernel matrices, label-shuffle check | 01, 04, 05, 06, 07, 11, and `engineered` |
| `geometry.py` | Haider | trace normalisation, $g_{\text{gen}}$ and $g_{\text{tra}}$, Huang's λ rule, sensitivity, top direction | 04, 05, 06, 07, 11, and `engineered` |
| `engineered.py` | Haider | fixed label generator (ideal projected kernel vs. sum of all 9 classical kernels) and engineered labels | 05, 11 |

---

## Method in brief

**Data.** Fashion-MNIST class 0 (T-shirt/top, label −1) vs. class 6 (Shirt, +1). Seeds 0, 100, 200, 300, 400; each seed draws 40 training and 20 test images per class. The indices are saved in `results/splits/`.

**Preprocessing.** PCA to $n$ features ($n$ = number of qubits), scaled to $[-1, 1]$, multiplied by $\alpha$. Both PCA and scaling are fitted on the training images only.

**Encoding circuit.** $U(x)$ = [$R_y(\alpha x_i)$ on every qubit, then a CNOT chain] × $L$, with $L = 2$. With $L = 1$ the fidelity kernel has a classical closed form, $\prod_i \cos^2\big((a_i-b_i)/2\big)$, which we use as a check. Diagrams: notebook 09.

**Fidelity kernel.** $k(x,x') = |\langle\psi(x')|\psi(x)\rangle|^2$, the probability of the all-zero outcome of $U(x')^\dagger U(x)$. It needs one circuit per pair of images: 6,360 per seed and setting.

**Projected kernel.** For each qubit $k$, measure $\langle X_k\rangle, \langle Y_k\rangle, \langle Z_k\rangle$ (3 circuits per image: $Z$; $X$ via $H$; $Y$ via $S^\dagger H$). This gives the one-qubit reduced density matrices $\rho_k(x)$ and the kernel $k(x,x') = \exp\big(-\gamma\sum_k\|\rho_k(x)-\rho_k(x')\|_F^2\big)$, with $\gamma = s\,/$ median squared distance of the training features.

**Equal tuning budget.** Every model is tuned by 5-fold inner CV on the training images only and then scored once on the test images. The budgets:

- fidelity: 4 values of $\alpha$ × 7 values of $C$ = 28 settings;
- projected: 2 values of $\alpha$ × 2 values of $s$ × 7 values of $C$ = 28 settings;
- SVM RBF and SVM polynomial: 28 settings each;
- linear SVM and logistic regression: 7 settings each.

$C^\ast$ is the classical SVM with the best inner-CV score. $Q^\ast$ is the quantum kernel with the better inner-CV score; ties go to the projected kernel, which needs fewer circuits.

**Geometric difference.** We use $g_{\text{gen}}$ and $g_{\text{tra}}$ from Huang et al. (Eqs. F19-F20), on trace-normalised kernels, at a fixed $\lambda = 0.1$. Huang's own λ rule is feasible in only 15 of 60 projected and 27 of 60 fidelity cases, because our classical kernel matrices are nearly low-rank (notebook 04).

**Engineered labels.** We follow Huang et al.'s construction: $y = \sqrt{K^Q}\,v$, where $v$ is the top eigenvector of the $g_{\text{gen}}$ matrix, then a median split. The generator kernels are fixed before any accuracy is computed: the ideal projected kernel ($\alpha = 1$, $s = 2$) and the sum of all 9 classical kernels in the grids. We use $\lambda \in \{0.1, 0.01, 0.001\}$.

**Noise model.** `FakeKingston` (IBM's calibration snapshot of `ibm_kingston`) via `AerSimulator.from_backend`. Circuits are transpiled to the device (optimisation level 1, `seed_transpiler=0`, physical qubits 0..n−1) and run with 2048 shots.

**Real hardware.** `ibm_kingston` (156-qubit Heron) through the Qiskit Runtime `Sampler`. Each job runs 362 circuits (120 images × 3 bases, plus 2 readout-calibration circuits with all qubits in $|0\rangle$ and all in $|1\rangle$), with 2048 shots.

| job | notebook | seed, $n$, $\alpha$ | purpose | quantum time |
|---|---|---|---|---|
| `dc273udhdvas73bload0` | 07 | 0, 6, 2 | four-level comparison on real labels | 197 s |
| `dc3bqo5hdvas73bniep0` | 10 | 0, 6, 1 | advantage test on engineered labels | 197 s |

**Readout mitigation.** Readout errors turn a true expectation value into $\langle P\rangle_{\text{meas}} = (1-p_{10}-p_{01})\langle P\rangle + (p_{01}-p_{10})$. We invert this per qubit, using that run's own calibration circuits (`hardware.mitigate_readout`).

---

## Honest limits

- **Small scale.** 80 training and 40 test images per seed, and 2-8 qubits. One test image is 2.5 accuracy points.
- **Heuristic improvement rule.** "Mean $\Delta A$ > std over 5 seeds" is the rule fixed in our synopsis. It is not a significance test.
- **Transductive engineered labels.** They are built from all 120 images of a seed (training and test), as in Huang et al. The models themselves never see test images during training or tuning.
- **Fidelity kernel and PSD.** The diagonal is set to 1 rather than measured. Sampled and noisy fidelity kernels can have small negative eigenvalues; we report them and apply no repair.
- **Not re-tuned under noise.** Notebook 06 keeps the ideal settings under noise. Notebook 11 re-tunes $\gamma$ and $C$ on each level's own features, with $\alpha$ fixed at 1, the value tuning chose in all 15 engineered runs at $n = 6$.
- **Limited hardware.** The IBM Open Plan allows 600 s per 28 days. That allowed only the projected kernel (3 circuits per image, versus one per pair for fidelity) and one seed per experiment.
- **Snapshot noise model.** The noise model is a fixed calibration snapshot. Its qubit 1 has a 17% readout error; the real device showed ≤ 2% on our runs.

---

## Reference

H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven and J. R. McClean, "Power of data in quantum machine learning," *Nature Communications* **12**, 2631 (2021).
