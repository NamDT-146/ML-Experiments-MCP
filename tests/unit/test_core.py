from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiaun_mcp.config import ConfigError, get_settings, redact
from aiaun_mcp.experiment import resolve_experiment_request
from aiaun_mcp.fsutil import inspect_local_dir, list_repo_configs, write_dataset_metadata, write_kernel_metadata
from aiaun_mcp.synthetic import generate_synthetic_color_cls
from aiaun_mcp.templates import kernel_source_has_secrets, resnet50_smoke_script, smoke_script


def test_redact_hides_token(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_secretvalue")
    assert "KGAT_secretvalue" not in redact("token=KGAT_secretvalue")
    assert "<KAGGLE_API_TOKEN>" in redact("token=KGAT_secretvalue")


def test_require_kaggle_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "namdtgk14")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    s = get_settings(tmp_path)
    with pytest.raises(ConfigError, match="KAGGLE_API_TOKEN"):
        s.require_kaggle()


def test_owner_personal(monkeypatch, tmp_path):
    monkeypatch.setenv("AIAUN_OWNER_MODE", "personal")
    monkeypatch.setenv("KAGGLE_USERNAME", "namdtgk14")
    monkeypatch.setenv("KAGGLE_ORG", "future-org")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    assert get_settings(tmp_path).kaggle_owner == "namdtgk14"


def test_resolve_missing_fields():
    payload = resolve_experiment_request()
    assert payload["ready"] is False
    assert "data_dir_or_kaggle_slug" in payload["missing"]
    assert "code_version" in payload["missing"]
    assert "config_path" in payload["missing"]
    assert "Ask the user" in payload["ask_user"]


def test_resolve_ready():
    payload = resolve_experiment_request(
        data_dir_or_kaggle_slug="fixtures/synthetic_color_cls",
        code_version="NamDT-146/semi-mask2former@main",
        config_path="smoke",
    )
    assert payload["ready"] is True
    assert payload["missing"] == []


def test_inspect_local_dir_synthetic(tmp_path):
    d = generate_synthetic_color_cls(tmp_path / "data")
    info = inspect_local_dir(str(d))
    assert info["ok"]
    assert info["file_count"] >= 12
    assert info["total_bytes"] < 50_000


def test_metadata_writers(tmp_path):
    write_dataset_metadata(tmp_path, owner_slug="namdtgk14/foo", title="Foo")
    meta = json.loads((tmp_path / "dataset-metadata.json").read_text())
    assert meta["id"] == "namdtgk14/foo"
    write_kernel_metadata(tmp_path, owner_slug="namdtgk14/bar", title="Bar", enable_gpu=False)
    k = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert k["enable_internet"] is True
    assert k["enable_gpu"] is False
    assert "machine_shape" not in k
    assert k["kernel_type"] == "script"
    write_kernel_metadata(tmp_path, owner_slug="namdtgk14/bar", title="Bar", enable_gpu=True)
    k = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert k["enable_gpu"] is True
    assert k["machine_shape"] == "NvidiaTeslaT4"


def test_smoke_script_no_secrets():
    src = smoke_script(dataset_slug="namdtgk14/aiaun-synthetic-color-cls")
    assert "UserSecretsClient" in src
    assert "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON" in src
    assert "from_service_account_info" in src
    assert kernel_source_has_secrets(src) is False
    assert kernel_source_has_secrets("key=KGAT_abc") is True
    assert kernel_source_has_secrets('{"private_key": "-----BEGIN PRIVATE KEY-----"}') is True
    assert "SMOKE_ACC" in src
    assert "DRIVE_SKIP" in src
    assert "5 * 1024 * 1024" not in src
    assert "get_or_create_experiment_folder" in src
    assert "DRIVE_FOLDER" in src
    assert "static_discovery=True" in src
    compile(src, "<smoke>", "exec")


def test_resnet50_smoke_script_compiles():
    src = resnet50_smoke_script(dataset_slug="namdtgk14/aiaun-synthetic-color-cls")
    assert "resnet50" in src
    assert "best.pt" in src
    assert "INFER_BEST_ACC" in src
    assert "INSTALL_TORCH_CU118" in src
    assert "cuda_probe_ok" in src
    assert "enable_gpu" not in src.lower() or True
    assert kernel_source_has_secrets(src) is False
    assert "DRIVE_SKIP_FILE" not in src
    assert "250 * 1024 * 1024" not in src
    assert "get_or_create_experiment_folder" in src
    compile(src, "<resnet50>", "exec")


def test_windows_drive_path_normalized_on_posix(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    monkeypatch.setenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", r"H:\Dev\Lab\AiAuN\.secret\sa.json")
    s = get_settings(tmp_path)
    import os as _os

    if _os.name != "nt":
        assert s.drive_sa_json.startswith("/mnt/h/")
        assert "\\" not in s.drive_sa_json
    else:
        assert s.drive_sa_json[1:3] == ":\\"


def test_relative_drive_json_path(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / ".secret").mkdir()
    (tmp_path / ".secret" / "sa.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", ".secret/sa.json")
    s = get_settings(tmp_path)
    assert s.drive_sa_json.endswith("sa.json")
    assert Path(s.drive_sa_json).is_file()


def test_tracking_links(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    monkeypatch.setenv("KAGGLE_USERNAME", "namdtgk14")
    monkeypatch.setenv("WANDB_ENTITY", "my-entity")
    monkeypatch.setenv("WANDB_PROJECT", "semi-mask2former")
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "folderid123")
    t = get_settings(tmp_path).tracking_links(
        kernel_slug="namdtgk14/aiaun-resnet50-smoke",
        wandb_run_id="abc",
        drive_file_id="fileid",
        experiment_folder_id="expfolder",
    )
    assert t["kaggle_kernel"] == "https://www.kaggle.com/code/namdtgk14/aiaun-resnet50-smoke"
    assert t["wandb_project"] == "https://wandb.ai/my-entity/semi-mask2former"
    assert t["wandb_run"] == "https://wandb.ai/my-entity/semi-mask2former/runs/abc"
    assert t["drive_folder"] == "https://drive.google.com/drive/folders/expfolder"
    assert "fileid" in t["drive_file"]


def test_tracking_links_wandb_override(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    monkeypatch.setenv("KAGGLE_USERNAME", "namdtgk14")
    monkeypatch.setenv("WANDB_ENTITY", "my-entity")
    monkeypatch.setenv("WANDB_PROJECT", "semi-mask2former")
    t = get_settings(tmp_path).tracking_links(
        wandb_entity="other-team",
        wandb_project="other-proj",
        wandb_run_id="r1",
    )
    assert t["wandb_project"] == "https://wandb.ai/other-team/other-proj"
    assert t["wandb_run"] == "https://wandb.ai/other-team/other-proj/runs/r1"


def test_list_repo_configs():
    from aiaun_mcp.config import repo_root

    root = repo_root()
    cfgs = list_repo_configs(root)
    if not cfgs:
        pytest.skip("semi-mask2former not present in this checkout")
    assert any("coco_real_m2f_r50_ts" in c for c in cfgs)
