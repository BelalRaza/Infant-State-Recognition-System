"""Extract Phase 2A pretrained feature banks.

Colab example:
    python scripts/phase2a_feature_bank.py --models ast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.config import DEFAULT_EMBEDDING_MODELS, FEATURE_CHECKPOINT_EVERY, FEATURES_DIR, ROOT_DIR, SUPERVISED_MANIFEST
from src.phase2a.data import load_or_create_splits, write_split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SUPERVISED_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--models", nargs="+", default=DEFAULT_EMBEDDING_MODELS, choices=["ast", "whisper"])
    parser.add_argument("--project-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=FEATURE_CHECKPOINT_EVERY)
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size for AST/Whisper embedding extraction.")
    parser.add_argument("--ast-model-name-or-path", default=None, help="Use adapted AST checkpoint directory for AST embeddings.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.phase2a.embeddings import extract_feature_bank

    manifest = load_or_create_splits(args.manifest, args.split_manifest)
    split_path = args.output_dir / "phase2a_split_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split_summary(manifest, split_path)
    paths = extract_feature_bank(
        manifest=manifest,
        model_keys=args.models,
        output_dir=args.output_dir,
        project_root=args.project_root,
        force=args.force,
        checkpoint_every=args.checkpoint_every,
        ast_model_name_or_path=args.ast_model_name_or_path,
        batch_size=args.batch_size,
    )
    print("Feature bank complete:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

