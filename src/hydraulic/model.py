"""Training, model comparison and prediction for cooler, valve and pump."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import TARGETS
from .features import feature_names


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "random_forest": make_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        ),
    }


@dataclass
class TargetResult:
    target: str
    scores: dict[str, float]          # model name -> test accuracy
    best_model: str
    confusion: np.ndarray
    labels: list[int]


@dataclass
class TrainingReport:
    results: dict[str, TargetResult] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {t: r.scores for t, r in self.results.items()}
        ).T.round(4)


def train_all(
    X: pd.DataFrame, profile: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[dict[str, Pipeline], TrainingReport]:
    """Fit every candidate per target, keep the best by test accuracy.

    The winner is refitted on all data before it is returned.
    """
    best: dict[str, Pipeline] = {}
    report = TrainingReport()
    for target in TARGETS:
        y = profile[target].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        scores, fitted = {}, {}
        for name, model in candidate_models(seed).items():
            model.fit(X_tr, y_tr)
            scores[name] = accuracy_score(y_te, model.predict(X_te))
            fitted[name] = model
        winner = max(scores, key=scores.get)
        labels = sorted(TARGETS[target])
        cm = confusion_matrix(y_te, fitted[winner].predict(X_te), labels=labels)
        report.results[target] = TargetResult(target, scores, winner, cm, labels)

        final = candidate_models(seed)[winner]
        final.fit(X, y)
        best[target] = final
    return best, report


def save_models(models: dict[str, Pipeline], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": feature_names(), "models": models}, path)


class ConditionPredictor:
    """Prediction logic, independent of any web framework."""

    def __init__(self, models: dict[str, Pipeline], features: list[str]):
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
