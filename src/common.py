"""Shared code for AI 631 Project 6 (projected quantum kernels).

Every notebook imports from here, so all three members use exactly the same
data, splits, preprocessing, encoding circuit and evaluation code.
Owner: Farrukh (week 1).

Usage from a notebook in notebooks/:
    import sys; sys.path.insert(0, "..")
    from src.common import *
"""
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Constants (one place, used everywhere)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLITS_DIR = RESULTS / "splits"

SEEDS = [0, 100, 200, 300, 400]   # 5 seeds, spaced (NB11 advice)
N_INNER_FOLDS = 5                 # inner CV on training data, for tuning only

# Fashion-MNIST classes: 0 = T-shirt/top -> label -1, 6 = Shirt -> label +1
CLASS_NEG, CLASS_POS = 0, 6

# Start small (NB11 sizing: sampled kernels cost O(M^2) circuits).
# Synopsis target is 150 / 50 per class; raise these once timing is known.
N_TRAIN_PER_CLASS = 40
N_TEST_PER_CLASS = 20

SHOTS = 2048                      # NB11 default shot budget per circuit


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
_DATA = None

DATA_DIR = ROOT / "data" / "fashion"          # not committed (see .gitignore)
FASHION_URL = ("https://github.com/zalandoresearch/fashion-mnist/"
               "raw/master/data/fashion/")
FASHION_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def _download_fashion():
    """Download the 4 official Fashion-MNIST files once (~30 MB in total)."""
    import urllib.request
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in FASHION_FILES:
        path = DATA_DIR / f
        if not path.exists():
            print("downloading", f)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(FASHION_URL + f, tmp)
            tmp.replace(path)


def _read_idx(path, offset):
    import gzip
    with gzip.open(path, "rb") as fh:
        return np.frombuffer(fh.read(), dtype=np.uint8, offset=offset)


def load_fashion_pair():
    """Return (X, y, orig_idx) for the two chosen Fashion-MNIST classes.

    X: pixels scaled to [0, 1], shape (n, 784)
    y: labels in {-1, +1}
    orig_idx: row numbers in the full 70,000 images (60k train + 10k test,
              in the official order)
    Source: the official Zalando Research files, downloaded once to data/.
    """
    global _DATA
    if _DATA is None:
        _download_fashion()
        imgs, labs = [], []
        for part in ("train", "t10k"):
            imgs.append(_read_idx(DATA_DIR / f"{part}-images-idx3-ubyte.gz", 16)
                        .reshape(-1, 784))
            labs.append(_read_idx(DATA_DIR / f"{part}-labels-idx1-ubyte.gz", 8))
        X_all, t = np.concatenate(imgs), np.concatenate(labs).astype(int)
        keep = np.where((t == CLASS_NEG) | (t == CLASS_POS))[0]
        X = X_all[keep].astype(np.float32) / 255.0
        y = np.where(t[keep] == CLASS_POS, 1, -1)
        _DATA = (X, y, keep)
    return _DATA


def get_split(seed, n_train_per_class=N_TRAIN_PER_CLASS,
              n_test_per_class=N_TEST_PER_CLASS):
    """Balanced train/test split for one seed: (Xtr, ytr, Xte, yte).

    The seed controls both the subsample and the split. The chosen indices
    are saved to results/splits/ the first time, and loaded afterwards, so
    every member and every notebook uses identical splits.
    """
    X, y, orig = load_fashion_pair()
    name = f"split_seed{seed}_tr{n_train_per_class}_te{n_test_per_class}.npz"
    path = SPLITS_DIR / name
    if path.exists():
        d = np.load(path)
        tr, te = d["train"], d["test"]
    else:
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for c in (-1, 1):
            idx = rng.permutation(np.where(y == c)[0])
            tr.append(idx[:n_train_per_class])
            te.append(idx[n_train_per_class:n_train_per_class + n_test_per_class])
        tr = rng.permutation(np.concatenate(tr))
        te = rng.permutation(np.concatenate(te))
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, train=tr, test=te,
                 train_orig=orig[tr], test_orig=orig[te])
    return X[tr], y[tr], X[te], y[te]


# ---------------------------------------------------------------------------
# 2. Leakage-free preprocessing (NB11 factory pattern)
# ---------------------------------------------------------------------------
def make_preprocessor(n_qubits, alpha=1.0):
    """Factory: fit(Xtr) -> transform(X).

    PCA to n_qubits components, then min-max scaling to [-1, 1], both fitted
    on TRAINING data only. transform() returns rotation angles alpha * x,
    which go straight into the encoding circuit.
    """
    def fit(Xtr, ytr=None):
        pca = PCA(n_components=n_qubits, random_state=0).fit(Xtr)
        scaler = MinMaxScaler((-1, 1)).fit(pca.transform(Xtr))

        def transform(X):
            return alpha * scaler.transform(pca.transform(X))
        return transform
    return fit


# ---------------------------------------------------------------------------
# 3. Encoding circuit (synopsis Sec. IV-A)
# ---------------------------------------------------------------------------
def encoding_circuit(n_qubits, layers=2):
    """U(x): [Ry(x_i) on every qubit, then a CNOT chain] repeated `layers` times.

    Parameters are named x[0..n-1]; pass angles = alpha * x (from
    make_preprocessor). layers=2 is the main setting; layers=1 is the
    baseline whose fidelity kernel is classically easy (CNOTs cancel).
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for i in range(n_qubits):
            qc.ry(x[i], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    return qc


# ---------------------------------------------------------------------------
# 4. Evaluation with precomputed kernels
# ---------------------------------------------------------------------------
def inner_folds(ytr, seed):
    """Stratified inner folds on the TRAINING set (for tuning only)."""
    skf = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(ytr)), ytr))


def cv_score_precomputed(K_train, ytr, folds, C=1.0):
    """Mean inner-CV accuracy from one training kernel, using sub-blocks.

    No new circuits are needed: each fold reuses rows/columns of K_train.
    Note: the preprocessing inside K_train was fitted on the whole training
    set, never on test data.
    """
    scores = []
    for a, b in folds:
        clf = SVC(kernel="precomputed", C=C).fit(K_train[np.ix_(a, a)], ytr[a])
        scores.append(accuracy_score(ytr[b], clf.predict(K_train[np.ix_(b, a)])))
    return float(np.mean(scores))


def evaluate_precomputed(K_train, K_test, ytr, yte, C=1.0):
    """Train an SVM on K_train (n_tr x n_tr), score on K_test (n_te x n_tr)."""
    clf = SVC(kernel="precomputed", C=C).fit(K_train, ytr)
    return dict(acc=float(accuracy_score(yte, clf.predict(K_test))),
                auc=float(roc_auc_score(yte, clf.decision_function(K_test))))


# ---------------------------------------------------------------------------
# 5. Small diagnostics and file naming
# ---------------------------------------------------------------------------
def mean_offdiag(K):
    """Mean off-diagonal kernel value (NB11: explains concentration)."""
    return float(K[~np.eye(len(K), dtype=bool)].mean())


def min_eig(K):
    """Smallest eigenvalue; negative values mean K is not PSD."""
    return float(np.linalg.eigvalsh((K + K.T) / 2).min())


def result_path(kind, name, seed, n_qubits, ext):
    """results/<kind>/<name>_seed<seed>_q<n>.<ext>, folder created if needed."""
    folder = RESULTS / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}_seed{seed}_q{n_qubits}.{ext}"
