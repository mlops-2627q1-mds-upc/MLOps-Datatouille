"""Generate a synthetic spam/not-spam dataset for developing models independently.

The dataset mimics a tabular feature matrix: one row per message, one column per
engineered feature, and a binary target (1 = spam, 0 = ham). It is built with
``sklearn.datasets.make_classification`` so there is a real, learnable signal
mixing informative, redundant, repeated and pure-noise features, which makes it
usable for smoke-testing a training pipeline while the real data pipeline is
still being built.

Example:
    python scripts/generate_synthetic_data.py --test-size 0.2
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJ_ROOT / "data" / "processed" / "synthetic_spam.csv"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_dataset(
    n_samples: int,
    n_features: int,
    spam_fraction: float,
    n_informative: int,
    n_redundant: int,
    n_repeated: int,
    n_clusters_per_class: int,
    class_sep: float,
    flip_y: float,
    missing_rate: float,
    target_name: str,
    seed: int,
) -> pd.DataFrame:
    """Build a DataFrame with ``n_samples`` rows, ``n_features`` + 1 columns."""
    used = n_informative + n_redundant + n_repeated
    if used > n_features:
        raise ValueError(
            f"n_informative + n_redundant + n_repeated ({used}) exceeds "
            f"n_features ({n_features})"
        )

    features, target = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        weights=[1.0 - spam_fraction, spam_fraction],
        shuffle=True,
        random_state=seed,
    )

    columns = [f"feature_{i:04d}" for i in range(n_features)]
    frame = pd.DataFrame(features, columns=columns)
    frame[target_name] = target.astype(int)

    if missing_rate > 0:
        mask = np.random.default_rng(seed).random((n_samples, n_features)) < missing_rate
        frame.loc[:, columns] = frame.loc[:, columns].mask(mask)

    return frame


def write_dataset(frame: pd.DataFrame, output_path: Path, float_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format=float_format)


def summarize(frame: pd.DataFrame, target_name: str, path: Path) -> None:
    counts = frame[target_name].value_counts().sort_index()
    spam_share = counts.get(1, 0) / len(frame)
    logger.info(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Shape: {frame.shape[0]} samples x {frame.shape[1] - 1} features + '{target_name}'")
    logger.info(f"Spam (1): {counts.get(1, 0)} ({spam_share:.1%}) | Not spam (0): {counts.get(0, 0)}")
    logger.info(f"Missing values: {int(frame.isna().sum().sum())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--target-name", default="target", help="Name of the binary target column")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=1000)
    parser.add_argument("--spam-fraction", type=float, default=0.3, help="Share of label 1 rows")
    parser.add_argument("--n-informative", type=int, default=50)
    parser.add_argument("--n-redundant", type=int, default=100)
    parser.add_argument("--n-repeated", type=int, default=20)
    parser.add_argument("--n-clusters-per-class", type=int, default=2)
    parser.add_argument("--class-sep", type=float, default=1.0, help="Class separability")
    parser.add_argument("--flip-y", type=float, default=0.03, help="Label noise rate")
    parser.add_argument("--missing-rate", type=float, default=0.0, help="Fraction of feature cells to blank out")
    parser.add_argument("--float-format", default="%.4f", help="CSV float precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="If set, also write stratified <name>_train.csv / <name>_test.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = build_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        spam_fraction=args.spam_fraction,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        n_repeated=args.n_repeated,
        n_clusters_per_class=args.n_clusters_per_class,
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        missing_rate=args.missing_rate,
        target_name=args.target_name,
        seed=args.seed,
    )

    write_dataset(frame, args.output, args.float_format)
    summarize(frame, args.target_name, args.output)

    if args.test_size is not None:
        train, test = train_test_split(
            frame, test_size=args.test_size, random_state=args.seed, stratify=frame[args.target_name]
        )
        stem = args.output.with_suffix("")
        for split, name in ((train, "train"), (test, "test")):
            split_path = Path(f"{stem}_{name}.csv")
            write_dataset(split, split_path, args.float_format)
            summarize(split, args.target_name, split_path)


if __name__ == "__main__":
    main()
