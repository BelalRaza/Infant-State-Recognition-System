"""Create a minimal Phase 2A Colab package.

The package includes code, notebooks, manifests, and only the processed audio
needed by the strict 5-class and approved auxiliary-adaptation manifests.
It intentionally excludes raw downloads, caches, old results, and duplicates.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "phase2a_colab_package.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_file(zf: zipfile.ZipFile, path: Path, arcname: Path | None = None) -> None:
    if not path.exists() or not path.is_file():
        return
    zf.write(path, arcname or path.relative_to(ROOT))


def add_tree(zf: zipfile.ZipFile, directory: Path, patterns: tuple[str, ...] = ("*",)) -> None:
    if not directory.exists():
        return
    for pattern in patterns:
        for path in directory.rglob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(ROOT))


def needed_audio_paths() -> set[Path]:
    cause = pd.read_csv(ROOT / "data_lake" / "manifests" / "cause5_authentic_manifest_v2.csv")
    unified = pd.read_csv(ROOT / "data_lake" / "manifests" / "unified_manifest_v2.csv")
    aux = unified[
        (unified["include_for_aux_pretraining"] == True)
        & (unified["dedupe_status"] == "kept")
        & (unified["canonical_label"] != "unknown_or_uncertain")
    ]
    paths = set(cause["processed_path"].astype(str)).union(set(aux["processed_path"].astype(str)))
    return {ROOT / path for path in paths}


def main() -> None:
    args = parse_args()
    audio_paths = needed_audio_paths()
    missing = sorted(str(path.relative_to(ROOT)) for path in audio_paths if not path.exists())
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required processed audio files. First missing: {missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        add_tree(zf, ROOT / "src", ("*.py",))
        add_tree(zf, ROOT / "scripts", ("*.py",))
        add_tree(zf, ROOT / "notebooks", ("Phase2A*.ipynb", "Phase2B*.ipynb"))
        add_file(zf, ROOT / "requirements-phase2a.txt")
        for manifest in [
            "cause5_authentic_manifest_v2.csv",
            "unified_manifest_v2.csv",
            "near_duplicate_report_v2.csv",
            "phase2a_split_manifest_v1.csv",
            "phase2a_split_manifest_v1.summary.json",
        ]:
            add_file(zf, ROOT / "data_lake" / "manifests" / manifest)
        for report in [
            "phase_a_v2_summary.json",
            "external_data_report_v2.md",
        ]:
            add_file(zf, ROOT / "data_lake" / "reports" / report)
        for path in sorted(audio_paths):
            zf.write(path, path.relative_to(ROOT))

        summary = {
            "required_audio_files": len(audio_paths),
            "package_purpose": "Phase 2A Colab run without raw dataset redownloads",
            "excluded": ["data_lake/raw", "duplicate processed audio", "results", "huggingface cache"],
        }
        zf.writestr("PHASE2A_COLAB_PACKAGE_SUMMARY.json", json.dumps(summary, indent=2))

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output}")
    print(f"Required processed audio files: {len(audio_paths)}")
    print(f"Zip size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

