"""Command line entry point: load data -> features -> train -> report -> save.

    python -m hydraulic.train --data data/raw            # real UCI data
    python -m hydraulic.train --synthetic                # synthetic stand-in
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .data import load_dataset, make_synthetic  # noqa: E402
from .features import cycle_features  # noqa: E402
from .model import TrainingReport, save_models, sensor_importance, train_all  # noqa: E402


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(report: TrainingReport, out: Path) -> Path:
    table = report.cv_table()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.plot.bar(ax=ax, rot=0, width=0.8)
    lo = max(0.0, table.min().min() - 0.05)
    ax.set_ylim(lo, 1.005)
    ax.set_ylabel("Mean 5-fold CV accuracy")
    ax.set_title("Model comparison per component (purged block CV on training data)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out / "model_comparison.png")


def plot_confusions(report: TrainingReport, out: Path) -> Path:
    n = len(report.results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes = [axes] if n == 1 else axes
    for ax, (target, r) in zip(axes, report.results.items()):
        cm = r.confusion
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(r.labels)), r.labels)
        ax.set_yticks(range(len(r.labels)), r.labels)
        for i in range(len(r.labels)):
            for j in range(len(r.labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{target}: {r.best_model}\ntime-blocked test acc {r.test_accuracy:.3f}", fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
    return _save(fig, out / "confusion_matrices.png")


def plot_importance(importance: dict[str, pd.Series], out: Path, top: int = 8) -> Path:
    n = len(importance)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), sharex=True)
    axes = [axes] if n == 1 else axes
    for ax, (target, s) in zip(axes, importance.items()):
        s.head(top)[::-1].plot.barh(ax=ax, color="#3a6ea5")
        ax.set_title(target, fontsize=10)
        ax.set_xlabel("share of importance")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Which sensors drive each prediction (extra-trees importance)", fontsize=11)
    return _save(fig, out / "sensor_importance.png")


def write_results(report: TrainingReport, importance: dict[str, pd.Series],
                  meta: dict, out: Path) -> Path:
    data = {
        **meta,
        "components": {
            t: {
                "cv_scheme": r.cv_scheme,
                "best_model": r.best_model,
                "cv_scores": {m: {"mean": round(a, 4), "std": round(b, 4)}
                              for m, (a, b) in r.cv_scores.items()},
                "test_accuracy": round(r.test_accuracy, 4),
                "test_f1_macro": round(r.test_f1_macro, 4),
                "random_split_accuracy": round(r.random_split_accuracy, 4),
                "top_sensors": importance[t].head(3).round(3).to_dict(),
            }
            for t, r in report.results.items()
        },
    }
    path = out / "results.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="folder with the UCI .txt files")
    src.add_argument("--synthetic", action="store_true", help="use generated data")
    ap.add_argument("--model-out", type=Path, default=Path("models/condition_models.joblib"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    t0 = time.time()
    ds = make_synthetic() if args.synthetic else load_dataset(args.data)
    print(f"Loaded {ds.n_cycles} cycles x {len(ds.signals)} sensors ({time.time() - t0:.1f}s)")

    X = cycle_features(ds.signals)
    print(f"Features: {X.shape[1]} per cycle")

    models, report = train_all(X, ds.profile)
    print("\nCV accuracy of every candidate:\n" + report.cv_table().to_string())
    print("\nSelected models, evaluated on the time-blocked test set:\n"
          + report.summary().to_string())

    importance = {t: sensor_importance(X, ds.profile[t].to_numpy()) for t in report.results}
    print("\nTop sensors:")
    for t, s in importance.items():
        print(f"  {t:12s} " + ", ".join(f"{k} {v:.2f}" for k, v in s.head(3).items()))

    args.reports.mkdir(parents=True, exist_ok=True)
    save_models(models, args.model_out)
    outputs = [
        plot_model_comparison(report, args.reports),
        plot_confusions(report, args.reports),
        plot_importance(importance, args.reports),
        write_results(report, importance, {
            "data": "synthetic" if args.synthetic else "UCI hydraulic (2,205 cycles)",
            "n_cycles": ds.n_cycles,
            "n_features": X.shape[1],
        }, args.reports),
    ]
    print(f"\nSaved model bundle: {args.model_out}")
    for p in outputs:
        print(f"Saved: {p}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
