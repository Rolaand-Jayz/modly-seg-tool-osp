"""Create and hash the shared 512px comparison fixtures from pinned source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


FIXTURES = ("chair.png", "flamingo.png", "teapot.png")
SOURCE_REPOSITORY = "https://github.com/VAST-AI-Research/TripoSR"
SOURCE_REVISION = "107cefdc244c39106fa830359024f6a2f1c78871"
FIXTURE_SIZE = (512, 512)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def prepare(source_dir: Path, destination_dir: Path, evidence_path: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in FIXTURES:
        source_path = source_dir / name
        destination_path = destination_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(f"pinned comparison fixture is missing: {source_path}")
        with Image.open(source_path) as source:
            original_size = source.size
            source_mode = source.mode
            prepared = source.convert("RGBA" if "A" in source.getbands() else "RGB")
            prepared = prepared.resize(FIXTURE_SIZE, Image.Resampling.LANCZOS)
            prepared.save(destination_path, format="PNG", optimize=False)
        records.append({
            "name": name,
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source_dimensions": list(original_size),
            "source_mode": source_mode,
            "prepared_path": str(destination_path),
            "prepared_sha256": sha256_file(destination_path),
            "prepared_dimensions": list(FIXTURE_SIZE),
            "prepared_mode": Image.open(destination_path).mode,
            "resize_filter": "Pillow LANCZOS",
        })
    evidence: dict[str, object] = {
        "schema_version": 1,
        "fixture_set_id": "triposr-upstream-chair-flamingo-teapot-512-v1",
        "upstream_repository": SOURCE_REPOSITORY,
        "upstream_revision": SOURCE_REVISION,
        "comparison_size": list(FIXTURE_SIZE),
        "image_fit_rule": "resize full frame to 512x512; do not crop or alter alpha/background",
        "fixtures": records,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.destination_dir, args.evidence), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
