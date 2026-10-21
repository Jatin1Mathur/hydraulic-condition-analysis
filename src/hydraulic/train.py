"""Command line entry point: load data -> features -> train -> report -> save.

    python -m hydraulic.train --data data/raw            # real UCI data
    python -m hydraulic.train --synthetic                # synthetic stand-in
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import load_dataset, make_synthetic  # noqa: E402
from .features import cycle_features  # noqa: E402
from .model import TrainingReport, save_models, train_all  # noqa: E402


def plot_report(report: TrainingReport, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    table = report.table()
    ax = table.plot.bar(figsize=(8, 4.5), rot=0, width=0.75)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison per component")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    p = out_dir / "model_comparison.png"
    plt.tight_layout()
    plt.savefig(p, dpi=130)
    plt.close()
    paths.append(p)

    fig, axes = plt.subplots(1, len(report.results), figsize=(12, 3.8))
    for ax, (target, r) in zip(axes, report.results.items()):
        ax.imshow(r.confusion, cmap="Blues")
        ax.set_xticks(range(len(r.labels)), r.labels)
        ax.set_yticks(range(len(r.labels)), r.labels)
        for i in range(len(r.labels)):
            for j in range(len(r.labels)):
                v = r.confusion[i, j]
                dark = v > r.confusion.max() / 2
                ax.text(j, i, v, ha="center", va="center", fontsize=9,
                        color="white" if dark else "black")
        ax.set_title(f"{target} ({r.best_model})", fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
    p = out_dir / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(p, dpi=130)
    plt.close()
    paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="folder with the UCI .txt files")
    src.add_argument("--synthetic", action="store_true", help="use generated data")
    ap.add_argument("--model-out", type=Path, default=Path("models/condition_models.joblib"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    ds = make_synthetic() if args.synthetic else load_dataset(args.data)
    print(f"Loaded {ds.n_cycles} cycles x {len(ds.signals)} sensors")

    X = cycle_features(ds.signals)
    print(f"Features: {X.shape[1]} per cycle")

    models, report = train_all(X, ds.profile)
    print("\nTest accuracy:\n", report.table().to_string(), sep="")
    for t, r in report.results.items():
        print(f"  {t:7s} -> {r.best_model}")

    save_models(models, args.model_out)
    plots = plot_report(report, args.reports)
    print(f"\nSaved model bundle: {args.model_out}")
    for p in plots:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
