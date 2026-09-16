from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def inspect_local_dir(path: str, *, max_samples: int = 20) -> dict:
    root = Path(path).expanduser()
    if not root.exists():
        return {"ok": False, "error": f"path does not exist: {root}"}
    if not root.is_dir():
        return {"ok": False, "error": f"not a directory: {root}"}
    files = [p for p in root.rglob("*") if p.is_file()]
    samples = [str(p.relative_to(root)) for p in files[:max_samples]]
    return {
        "ok": True,
        "path": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": dir_size_bytes(root),
        "samples": samples,
    }


def write_dataset_metadata(
    folder: Path,
    *,
    owner_slug: str,
    title: str,
    licenses: list[dict] | None = None,
) -> Path:
    if "/" not in owner_slug:
        raise ValueError("dataset id must be owner/slug")
    meta = {
        "title": title,
        "id": owner_slug,
        "licenses": licenses or [{"name": "CC0-1.0"}],
    }
    path = folder / "dataset-metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def write_kernel_metadata(
    folder: Path,
    *,
    owner_slug: str,
    title: str,
    code_file: str = "script.py",
    dataset_sources: list[str] | None = None,
    enable_gpu: bool = False,
    enable_internet: bool = True,
    is_private: bool = True,
    machine_shape: str = "",
) -> Path:
    if "/" not in owner_slug:
        raise ValueError("kernel id must be owner/slug")
    meta = {
        "id": owner_slug,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": is_private,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    # When GPU is enabled, explicitly request T4 unless overridden.
    # enable_gpu alone often schedules P100 (sm_60); current Kaggle PyTorch
    # only ships sm_70+.
    if enable_gpu:
        meta["machine_shape"] = machine_shape.strip() or "NvidiaTeslaT4"
    path = folder / "kernel-metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def content_hash(folder: Path) -> str:
    """SHA-256 of sorted relative-path:size pairs (fast, not content)."""
    h = hashlib.sha256()
    entries = sorted(
        (str(p.relative_to(folder)).replace("\\", "/"), p.stat().st_size)
        for p in folder.rglob("*")
        if p.is_file() and p.name != "aiaun_manifest.json"
    )
    for name, size in entries:
        h.update(f"{name}:{size}\n".encode())
    return h.hexdigest()[:16]


def write_aiaun_manifest(
    folder: Path,
    *,
    required_paths: list[str] | None = None,
    git_sha: str = "",
    source_dir: str = "",
) -> Path:
    """Write aiaun_manifest.json into folder after a dataset push."""
    manifest = {
        "pushed_at": int(time.time()),
        "content_hash": content_hash(folder),
        "git_sha": git_sha,
        "source_dir": source_dir,
        "required_paths": required_paths or [],
    }
    path = folder / "aiaun_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def list_repo_configs(repo_root: Path, *, glob: str = "semi-mask2former/configs/**/*.yaml") -> list[str]:
    hits = sorted(repo_root.glob(glob))
    return [str(p.relative_to(repo_root)).replace("\\", "/") for p in hits if p.is_file()]
