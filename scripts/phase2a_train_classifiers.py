"""Train Phase 2A classifiers on cached feature banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.classifiers import train_and_evaluate_feature_set
from src.phase2a.config import FEATURES_DIR, METRICS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--feature-files", nargs="*", default=None, help="Specific .npz feature files. Defaults to every .npz in features-dir.")
    parser.add_argument("--output-prefix", default="phase2a")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.feature_files:
        feature_paths = [Path(p) for p in args.feature_files]
    else:
        feature_paths = sorted(args.features_dir.glob("*_features.npz")) + sorted(args.features_dir.glob("*_embeddings.npz"))
    if not feature_paths:
        raise FileNotFoundError(f"No feature banks found in {args.features_dir}")

    all_results = {}
    for path in feature_paths:
        feature_name = path.stem.replace("_features", "").replace("_embeddings", "")
        print(f"\n=== Training classifiers for {feature_name}: {path} ===")
        all_results[feature_name] = train_and_evaluate_feature_set(path, feature_name, output_prefix=args.output_prefix)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = METRICS_DIR / f"{args.output_prefix}_all_feature_sets_metrics.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved combined metrics: {summary_path}")


if __name__ == "__main__":
    main()

