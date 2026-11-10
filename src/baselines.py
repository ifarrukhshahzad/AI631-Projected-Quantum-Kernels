"""Classical baselines: majority class, logistic regression and tuned
RBF / polynomial / linear kernel SVMs.
Owner: Haider (week 1).

All models use the SAME features as the quantum kernels (PCA to n_qubits,
scaled to [-1, 1], fitted on training data only) and are tuned on the SAME
inner folds, so every comparison is like for like.
classical_gram() returns the tuned kernel matrices, which the geometric
difference (week 2) needs.
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Hyperparameter grids (tuned on inner CV of the training set only)
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
GRIDS = {
    "rbf": {"C": C_GRID, "gamma": [0.001, 0.01, 0.1, 1, 10, 100]},
    "poly": {"C": C_GRID, "degree": [2, 3, 4],
             "gamma": [0.01, 0.1, 1], "coef0": [0, 1]},
    "linear": {"C": C_GRID},
}
LOGREG_GRID = {"C": C_GRID}


def tuned_svm(kind, Xtr, ytr, folds):
    """SVM with kernel `kind` ('rbf', 'poly', 'linear'), tuned on `folds`."""
    gs = GridSearchCV(SVC(kernel=kind), GRIDS[kind], cv=folds,
                      scoring="accuracy")
    return gs.fit(Xtr, ytr)


def _scores(model, Xte, yte):
    pred = model.predict(Xte)
    if hasattr(model, "decision_function"):
        score = model.decision_function(Xte)
    else:
        score = model.predict_proba(Xte)[:, 1]
    return dict(acc=float(accuracy_score(yte, pred)),
                auc=float(roc_auc_score(yte, score)))


def run_baselines(Ftr, ytr, Fte, yte, folds):
    """Fit every baseline on training features, score once on test.

    Returns {name: dict(acc, auc, inner_cv, params)}. inner_cv is the
    training cross-validation accuracy used to pick C* (the falsifier's
    comparator); it never uses test data.
    """
    out = {}
    maj = DummyClassifier(strategy="most_frequent").fit(Ftr, ytr)
    out["Majority class"] = dict(**_scores(maj, Fte, yte),
                                 inner_cv=None, params={})

    lr = GridSearchCV(LogisticRegression(max_iter=5000), LOGREG_GRID,
                      cv=folds, scoring="accuracy").fit(Ftr, ytr)
    out["Logistic regression"] = dict(**_scores(lr, Fte, yte),
                                      inner_cv=float(lr.best_score_),
                                      params=lr.best_params_)

    for kind, label in [("rbf", "SVM RBF (tuned)"),
                        ("poly", "SVM polynomial (tuned)"),
                        ("linear", "SVM linear (tuned)")]:
        gs = tuned_svm(kind, Ftr, ytr, folds)
        out[label] = dict(**_scores(gs, Fte, yte),
                          inner_cv=float(gs.best_score_),
                          params=gs.best_params_)
    return out


def best_classical(results):
    """Name of C*: the SVM kernel with the highest inner-CV accuracy."""
    svms = {k: v for k, v in results.items() if k.startswith("SVM")}
    return max(svms, key=lambda k: svms[k]["inner_cv"])


def classical_gram(kind, params, A, B=None):
    """Kernel matrix of a tuned classical kernel (for g_lambda in week 2)."""
    B = A if B is None else B
    if kind == "rbf":
        return rbf_kernel(A, B, gamma=params["gamma"])
    if kind == "poly":
        return polynomial_kernel(A, B, degree=params["degree"],
                                 gamma=params["gamma"], coef0=params["coef0"])
    if kind == "linear":
        return linear_kernel(A, B)
    raise ValueError(kind)


def label_shuffle_check(Ftr, ytr, Fte, yte, folds, seed=0):
    """NB11 leakage check: train the tuned RBF SVM on permuted labels.
    Test accuracy must fall to chance; otherwise information is leaking."""
    rng = np.random.default_rng(seed)
    gs = tuned_svm("rbf", Ftr, rng.permutation(ytr), folds)
    return float(accuracy_score(yte, gs.predict(Fte)))
