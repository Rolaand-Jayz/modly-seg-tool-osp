"""Stage the exact, hash-checked geometry comparison inputs.

This uses Python's standard library so the runtime image can fetch the pinned
artifacts without installing a downloader or changing the project image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import tarfile


CANDIDATES = {
    "hunyuan-mini-turbo": {
        "source": {
            "repo": "Tencent-Hunyuan/Hunyuan3D-2",
            "revision": "f8db63096c8282cb27354314d896feba5ba6ff8a",
        },
        "weights_repo": "tencent/Hunyuan3D-2mini",
        "weights_revision": "f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6",
        "files": {
            "hunyuan3d-dit-v2-mini-turbo/config.yaml": None,
            "hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors":
                "bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4",
            "hunyuan3d-vae-v2-mini-turbo/config.yaml": None,
            "hunyuan3d-vae-v2-mini-turbo/model.fp16.safetensors":
                "5dcaca67a8da9e7079fb7b55714a572d0651ac95983bb43099913feaf6738f94",
        },
    },
    "triposr": {
        "source": {
            "repo": "VAST-AI-Research/TripoSR",
            "revision": "107cefdc244c39106fa830359024f6a2f1c78871",
        },
        "weights_repo": "stabilityai/TripoSR",
        "weights_revision": "9700b06c1641864ecbbe5eb0d89b967f3045cd5e",
        "files": {
            "config.yaml": None,
            "model.ckpt": "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
        },
    },
}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> tuple[str, int, float]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Modly-AMD-geometry-probe/1"})
    started = time.perf_counter()
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                total += len(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), total, time.perf_counter() - started


def stage(candidate: str, root: Path) -> dict[str, object]:
    spec = CANDIDATES[candidate]
    base = root / candidate
    files: dict[str, object] = {}
    for relative, expected in spec["files"].items():
        target = base / "weights" / relative
        if target.exists():
            digest = digest_file(target)
            size = target.stat().st_size
            seconds = 0.0
        else:
            url = (
                f"https://huggingface.co/{spec['weights_repo']}/resolve/"
                f"{spec['weights_revision']}/{relative}?download=true"
            )
            digest, size, seconds = download(url, target)
        if expected is not None and digest != expected:
            target.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {candidate}:{relative}")
        files[relative] = {
            "path": str(target),
            "sha256": digest,
            "bytes": size,
            "download_seconds": round(seconds, 3),
            "expected_sha256": expected,
            "verified": expected is None or digest == expected,
        }
        print(json.dumps({"candidate": candidate, "file": relative, **files[relative]}), flush=True)
    record = {
        "candidate": candidate,
        "source": spec["source"],
        "weights_repo": spec["weights_repo"],
        "weights_revision": spec["weights_revision"],
        "files": files,
    }
    source_dir = base / "source"
    source_archive = base / "source.tar.gz"
    if not source_dir.exists():
        source_url = (
            f"https://codeload.github.com/{spec['source']['repo']}/tar.gz/"
            f"{spec['source']['revision']}"
        )
        source_hash, source_size, source_seconds = download(source_url, source_archive)
        source_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source_archive, "r:gz") as archive:
            root_prefix = archive.getmembers()[0].name.split("/", 1)[0]
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("pinned source archive contains an unsafe path")
                relative = member_path.relative_to(root_prefix) if member_path.parts else Path()
                member.name = str(relative)
                if member.issym() or member.islnk():
                    raise ValueError("pinned source archive contains a link")
                archive.extract(member, source_dir, filter="data")
        record["source_archive"] = {
            "path": str(source_archive),
            "sha256": source_hash,
            "bytes": source_size,
            "download_seconds": round(source_seconds, 3),
        }
        print(json.dumps({"candidate": candidate, "source_archive_sha256": source_hash, "source_archive_bytes": source_size}), flush=True)
    else:
        record["source_path"] = str(source_dir)
    evidence = base / "download-evidence.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=sorted(CANDIDATES))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    record = stage(args.candidate, args.root.resolve())
    print(json.dumps({"staged": record["candidate"], "evidence": str(args.root.resolve() / args.candidate / "download-evidence.json")}))


if __name__ == "__main__":
    main()
