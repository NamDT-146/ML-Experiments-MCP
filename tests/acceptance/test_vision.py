from __future__ import annotations

import json

import pytest

from aiaun_mcp.config import ConfigError, get_settings
from aiaun_mcp.experiment import resolve_experiment_request
from aiaun_mcp.fsutil import inspect_local_dir
from aiaun_mcp.synthetic import generate_synthetic_color_cls
from aiaun_mcp.templates import kernel_source_has_secrets, smoke_script

live = pytest.mark.live


def test_a1_incomplete_request_asks():
    p = resolve_experiment_request(data_dir_or_kaggle_slug="x")
    assert p["ready"] is False
    assert "code_version" in p["missing"]
    assert "config_path" in p["missing"]


def test_a4_smoke_script_metadata(tmp_path):
    from aiaun_mcp.fsutil import write_kernel_metadata

    src = smoke_script(dataset_slug="namdtgk14/aiaun-synthetic-color-cls")
    assert kernel_source_has_secrets(src) is False
    write_kernel_metadata(
        tmp_path,
        owner_slug="namdtgk14/aiaun-color-smoke",
        title="smoke",
        dataset_sources=["namdtgk14/aiaun-synthetic-color-cls"],
        enable_gpu=False,
        enable_internet=True,
    )
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is True
    assert meta["enable_gpu"] is False


def test_a6_missing_token(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    with pytest.raises(ConfigError):
        get_settings(tmp_path).require_kaggle()


@live
def test_a2_a3_dataset_check_push():
    from aiaun_mcp.kaggle_ops import dataset_exists, dataset_push

    settings = get_settings()
    settings.require_kaggle()
    dest = settings.root / "fixtures" / "synthetic_color_cls"
    generate_synthetic_color_cls(dest)
    info = inspect_local_dir(str(dest))
    assert info["total_bytes"] < 5_000_000
    slug = f"{settings.kaggle_owner}/aiaun-synthetic-color-cls"
    check = dataset_exists(settings, slug)
    assert check["ok"]
    pushed = dataset_push(settings, str(dest), slug, "AiAuN synthetic color cls", force=False)
    assert pushed.get("ok"), pushed


@live
def test_a4_a5_kernel_push_status():
    from aiaun_mcp.kaggle_ops import kernel_logs, kernel_push, kernel_status

    settings = get_settings()
    slug = f"{settings.kaggle_owner}/aiaun-synthetic-color-cls"
    src = smoke_script(dataset_slug=slug)
    result = kernel_push(
        settings,
        kernel_slug=f"{settings.kaggle_owner}/aiaun-color-smoke",
        script_content=src,
        dataset_slugs=[slug],
        title="AiAuN color smoke",
        enable_gpu=False,
    )
    assert result.get("ok"), result
    st = kernel_status(settings, f"{settings.kaggle_owner}/aiaun-color-smoke")
    assert st.get("ok"), st
    logs = kernel_logs(settings, f"{settings.kaggle_owner}/aiaun-color-smoke")
    assert "ok" in logs


def test_a7_wandb_documented_not_embedded():
    src = smoke_script(dataset_slug="namdtgk14/aiaun-synthetic-color-cls")
    assert "WANDB_API_KEY" in src
    assert kernel_source_has_secrets(src) is False


@live
def test_a8_drive_folder_and_upload(tmp_path):
    from aiaun_mcp.drive_ops import folder_info, upload_file

    settings = get_settings()
    info = folder_info(settings)
    assert info.get("ok"), info
    metrics = tmp_path / "aiaun_smoke_metrics.json"
    metrics.write_text('{"smoke": true, "acc_note": "live-upload"}', encoding="utf-8")
    up = upload_file(settings, str(metrics), name="aiaun_smoke_metrics.json")
    if not up.get("ok") and "storageQuotaExceeded" in str(up.get("error", "")):
        pytest.skip(
            "SA cannot write to My Drive. Use a Shared Drive folder or "
            "python -m aiaun_mcp.drive_login after adding .secret/gdrive-oauth-client.json"
        )
    assert up.get("ok"), up
