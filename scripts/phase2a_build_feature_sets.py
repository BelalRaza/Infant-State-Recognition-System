"""Build combined Phase 2A feature banks for one-shot ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.config import FEATURES_DIR


FEATURE_FILES = {
    "ast": "ast_embeddings.npz",
    "ast_aux": "ast_aux_adapted_embeddings.npz",
    "whisper": "whisper_embeddings.npz",
    "handcrafted": "handcrafted_features.npz",
}

DEFAULT_COMBINATIONS = {
    "ablation_no_aux_no_whisper": ["ast", "handcrafted"],
    "ablation_no_aux_with_whisper": ["ast", "whisper", "handcrafted"],
    "ablation_aux_no_whisper": ["ast_aux", "handcrafted"],
    "ablation_aux_with_whisper": ["ast_aux", "whisper", "handcrafted"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.features_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for combo_name, parts in DEFAULT_COMBINATIONS.items():
        out_path = args.features_dir / f"{combo_name}_features.npz"
        if out_path.exists() and not args.force:
            print(f"exists: {out_path}")
            summary[combo_name] = {"status": "exists", "path": str(out_path), "parts": parts}
            continue

        missing = [part for part in parts if not (args.features_dir / FEATURE_FILES[part]).exists()]
        if missing:
            print(f"skip {combo_name}: missing {missing}")
            summary[combo_name] = {"status": "missing", "missing": missing, "parts": parts}
            continue

        arrays = [load_feature_bank(args.features_dir / FEATURE_FILES[part]) for part in parts]
        first = arrays[0]
        for part, arr in zip(parts[1:], arrays[1:]):
            if not np.array_equal(first["sample_ids"], arr["sample_ids"]):
                raise ValueError(f"Sample ID mismatch in {combo_name}: {part}")
            if not np.array_equal(first["y"], arr["y"]):
                raise ValueError(f"Label mismatch in {combo_name}: {part}")
            if not np.array_equal(first["splits"], arr["splits"]):
                raise ValueError(f"Split mismatch in {combo_name}: {part}")

        X = np.concatenate([arr["X"] for arr in arrays], axis=1).astype(np.float32)
        np.savez_compressed(
            out_path,
            X=X,
            y=first["y"],
            sample_ids=first["sample_ids"],
            splits=first["splits"],
            parts=np.asarray(parts, dtype=object),
        )
        print(f"wrote: {out_path} shape={X.shape}")
        summary[combo_name] = {"status": "wrote", "path": str(out_path), "parts": parts, "shape": list(X.shape)}

    summary_path = args.features_dir / "combined_feature_sets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


def load_feature_bank(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"],
        "sample_ids": data["sample_ids"].astype(str),
        "splits": data["splits"].astype(str),
    }


if __name__ == "__main__":
    main()

