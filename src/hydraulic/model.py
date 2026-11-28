"""Model selection, evaluation and prediction for the four components.

Evaluation protocol per target:
  1. Hold out a time-blocked 20 % test set: every 5th block of consecutive
     cycles, with a purge gap of cycles removed on each side so that no
     training cycle is a near-copy neighbour of a test cycle. It is used
     exactly once, at the end.
  2. On the remaining cycles, compare candidate models with purged
     block cross-validation (see config.TARGET_CV).
  3. Pick the model with the best mean CV accuracy.
  4. Fit it on the training cycles and report accuracy and macro F1 on the
     time-blocked test set.
  5. For comparison, also score the same model on a random stratified
     80/20 split. The difference shows how much a random split flatters it.
  6. Refit on all data for deployment.

Choosing the model on CV and reporting on an untouched, time-separated test
set keeps the reported number honest: the test set played no part in the
choice, and it contains no near-duplicates of training cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    N_TIME_BLOCKS,
    PURGE_CYCLES,
    TARGET_CV,
    TARGETS,
    TEST_EVERY,
    TEST_OFFSET,
)
from .features import feature_names, sensor_of

N_FOLDS = 5


def candidate_models(seed: int = 42) -> dict[str, ClassifierMixin]:
    return {
        # linear baseline; needs scaled features
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000)
        ),
        # tree ensembles; scale-invariant
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def _purge(n: int, test: np.ndarray, gap: int) -> np.ndarray:
    """Mask of positions within `gap` of a test position (test itself excluded)."""
    near = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(test)
    for d in range(1, gap + 1):
        near[np.clip(idx - d, 0, n - 1)] = True
        near[np.clip(idx + d, 0, n - 1)] = True
    return near & ~test


def time_blocked_holdout(
    n: int,
    n_blocks: int = N_TIME_BLOCKS,
    every: int = TEST_EVERY,
    offset: int = TEST_OFFSET,
    purge: int = PURGE_CYCLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Train and test positions for a time-blocked hold-out with a purge gap.

    The n cycles (in time order) are cut into `n_blocks` contiguous blocks.
    Blocks offset, offset+every, ... form the test set. Up to `purge` cycles
    on each side of a test block are removed from training.
    """
    block = np.arange(n) * n_blocks // n
    test = (block % every) == offset
    gap = min(purge, max(1, n // n_blocks // 2))
    train = ~test & ~_purge(n, test, gap)
    return np.flatnonzero(train), np.flatnonzero(test)


class PurgedBlockKFold:
    """K-fold over contiguous blocks of rows, with a purge gap.

    Rows must be in time order. They are cut into `n_blocks` contiguous
    blocks; fold f validates on blocks f, f+k, f+2k, ... and trains on the
    rest minus `purge` rows on each side of every validation block.
    Interleaving the blocks keeps every condition level in each fold even
    when a level only occurs in a few long periods.
    """

    def __init__(self, n_splits: int = N_FOLDS, n_blocks: int = 15, purge: int = PURGE_CYCLES):
        self.n_splits = n_splits
        self.n_blocks = n_blocks
        self.purge = purge

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        block = np.arange(n) * self.n_blocks // n
        gap = min(self.purge, max(1, n // self.n_blocks // 2))
        for f in range(self.n_splits):
            val = (block % self.n_splits) == f
            train = ~val & ~_purge(n, val, gap)
            yield np.flatnonzero(train), np.flatnonzero(val)


def make_cv(scheme: str, seed: int = 42):
    if scheme == "purged":
        return PurgedBlockKFold()
    if scheme == "blocked":
        return KFold(N_FOLDS, shuffle=False)
    if scheme == "stratified":
        return StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    raise ValueError(f"unknown CV scheme: {scheme}")


@dataclass
class TargetResult:
    target: str
    cv_scheme: str
    cv_scores: dict[str, tuple[float, float]]  # model -> (mean, std) CV accuracy
    best_model: str
    test_accuracy: float
    test_f1_macro: float
    random_split_accuracy: float  # same model, random stratified 80/20 split
    confusion: np.ndarray
    labels: list[int]


@dataclass
class TrainingReport:
    results: dict[str, TargetResult] = field(default_factory=dict)

    def cv_table(self) -> pd.DataFrame:
        """Mean CV accuracy of every candidate, one row per target."""
        return pd.DataFrame(
            {t: {m: s[0] for m, s in r.cv_scores.items()} for t, r in self.results.items()}
        ).T.round(4)

    def summary(self) -> pd.DataFrame:
        rows = []
        for t, r in self.results.items():
            mean, std = r.cv_scores[r.best_model]
            rows.append({
                "component": t,
                "cv_scheme": r.cv_scheme,
                "best_model": r.best_model,
                "cv_accuracy": f"{mean:.4f} ± {std:.4f}",
                "test_accuracy": round(r.test_accuracy, 4),
                "test_f1_macro": round(r.test_f1_macro, 4),
                "random_split_accuracy": round(r.random_split_accuracy, 4),
            })
        return pd.DataFrame(rows).set_index("component")


def train_all(
    X: pd.DataFrame,
    profile: pd.DataFrame,
    targets: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, ClassifierMixin], TrainingReport]:
    targets = targets or list(TARGETS)
    X = X.reset_index(drop=True)
    best: dict[str, ClassifierMixin] = {}
    report = TrainingReport()

    train_idx, test_idx = time_blocked_holdout(len(X))

    for target in targets:
        y = profile[target].to_numpy()
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]

        scheme = TARGET_CV.get(target, "purged")
        cv_scores = {}
        for name, model in candidate_models(seed).items():
            s = cross_val_score(model, X_tr, y_tr, cv=make_cv(scheme, seed))
            cv_scores[name] = (float(s.mean()), float(s.std()))
        winner = max(cv_scores, key=lambda m: cv_scores[m][0])

        model = candidate_models(seed)[winner].fit(X_tr, y_tr)
        pred = model.predict(X_te)

        # for comparison only: the same model on a random stratified split
        r_tr, r_te = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
        )
        r_model = candidate_models(seed)[winner].fit(X.iloc[r_tr], y[r_tr])
        random_acc = float(accuracy_score(y[r_te], r_model.predict(X.iloc[r_te])))

        labels = sorted(TARGETS[target])
        report.results[target] = TargetResult(
            target=target,
            cv_scheme=scheme,
            cv_scores=cv_scores,
            best_model=winner,
            test_accuracy=float(accuracy_score(y_te, pred)),
            test_f1_macro=float(f1_score(y_te, pred, average="macro")),
            random_split_accuracy=random_acc,
            confusion=confusion_matrix(y_te, pred, labels=labels),
            labels=labels,
        )

        best[target] = candidate_models(seed)[winner].fit(X, y)
    return best, report


def sensor_importance(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> pd.Series:
    """Share of extra-trees impurity importance per sensor (sums to 1)."""
    et = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    imp = pd.Series(et.feature_importances_, index=X.columns)
    return imp.groupby(imp.index.map(sensor_of)).sum().sort_values(ascending=False)


def save_models(models: dict[str, ClassifierMixin], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": feature_names(), "models": models}, path)


class ConditionPredictor:
    """Prediction logic, independent of any web framework."""

    def __init__(self, models: dict[str, ClassifierMixin], features: list[str]):
        self.models = models
        self.features = features

    @classmethod
    def load(cls, path: str | Path) -> "ConditionPredictor":
        bundle = joblib.load(path)
        return cls(bundle["models"], bundle["features"])

    def predict(self, feature_row: dict[str, float]) -> dict[str, dict]:
        missing = [f for f in self.features if f not in feature_row]
        if missing:
            raise KeyError(f"missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        X = pd.DataFrame([[feature_row[f] for f in self.features]], columns=self.features)
        out = {}
        for target, model in self.models.items():
            label = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            out[target] = {
                "label": label,
                "description": TARGETS[target][label],
                "confidence": round(float(proba.max()), 4),
            }
        return out
