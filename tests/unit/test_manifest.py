from __future__ import annotations

import json
import time
from pathlib import Path

from aiaun_mcp.fsutil import content_hash, write_aiaun_manifest, write_kernel_metadata


def test_content_hash_deterministic(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    h1 = content_hash(tmp_path)
    h2 = content_hash(tmp_path)
    assert h1 == h2
    assert len(h1) == 16


def test_content_hash_changes_on_new_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    h1 = content_hash(tmp_path)
    (tmp_path / "b.txt").write_text("world")
    h2 = content_hash(tmp_path)
    assert h1 != h2


def test_write_aiaun_manifest(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2")
    path = write_aiaun_manifest(
        tmp_path,
        required_paths=["data.csv", "labels.csv"],
        git_sha="abc123",
        source_dir="/local/source",
    )
    assert path.name == "aiaun_manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["git_sha"] == "abc123"
    assert "data.csv" in manifest["required_paths"]
    assert manifest["content_hash"]
    assert manifest["pushed_at"] > 0


def test_manifest_excluded_from_hash(tmp_path):
    """aiaun_manifest.json must not affect the content_hash (circular)."""
    (tmp_path / "a.txt").write_text("x")
    h1 = content_hash(tmp_path)
    write_aiaun_manifest(tmp_path)
    h2 = content_hash(tmp_path)
    assert h1 == h2


def test_machine_shape_override(tmp_path):
    write_kernel_metadata(
        tmp_path,
        owner_slug="owner/slug",
        title="T",
        enable_gpu=True,
        machine_shape="NvidiaTeslaA100",
    )
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["machine_shape"] == "NvidiaTeslaA100"


def test_machine_shape_default_t4(tmp_path):
    write_kernel_metadata(tmp_path, owner_slug="owner/slug", title="T", enable_gpu=True)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["machine_shape"] == "NvidiaTeslaT4"


def test_machine_shape_absent_when_cpu(tmp_path):
    write_kernel_metadata(tmp_path, owner_slug="owner/slug", title="T", enable_gpu=False)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert "machine_shape" not in meta


def test_kernel_push_draft_does_not_call_kaggle(tmp_path, monkeypatch):
    """run_mode=draft must write files but not push to Kaggle."""
    from aiaun_mcp.config import get_settings
    import os

    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    monkeypatch.setenv("KAGGLE_USERNAME", "namdtgk14")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok")

    push_called = []

    def fake_push(work_dir):
        push_called.append(work_dir)

    from aiaun_mcp import kaggle_ops

    monkeypatch.setattr(kaggle_ops, "build_api", lambda s: type("A", (), {"kernels_push": fake_push})())

    settings = get_settings(tmp_path)
    result = kaggle_ops.kernel_push(
        settings,
        kernel_slug="owner/test-kernel",
        script_content="print('hi')",
        dataset_slugs=[],
        title="Test",
        run_mode="draft",
    )
    assert result["run_mode"] == "draft"
    assert result["ok"] is True
    assert push_called == []  # kernels_push never called
    assert "draft" in result["note"].lower()
    assert "work_dir" in result
    script_file = Path(result["work_dir"]) / "script.py"
    assert script_file.read_text() == "print('hi')"
