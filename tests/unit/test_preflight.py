from __future__ import annotations

import pytest

from aiaun_mcp.experiment import preflight_experiment, resolve_experiment_request


def _settings_stub(tmp_path, **kw):
    """Minimal settings-like object for offline preflight tests."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir(exist_ok=True)

    class _S:
        wandb_entity = kw.get("wandb_entity", "ent")
        wandb_entity_resolved = kw.get("wandb_entity", "ent")
        wandb_project = kw.get("wandb_project", "proj")
        wandb_api_key = kw.get("wandb_api_key", "")
        drive_sa_json = ""
        owner_mode = "personal"
        kaggle_owner = "namdtgk14"
        kaggle_username = "namdtgk14"
        kaggle_token = kw.get("kaggle_token", "tok")

        def require_kaggle(self):
            if not self.kaggle_token:
                raise RuntimeError("no token")

        def tracking_links(self, **_):
            return {}

    return _S()


def test_preflight_missing_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok")
    monkeypatch.setenv("WANDB_API_KEY", "wkey")
    s = _settings_stub(tmp_path)
    result = preflight_experiment(settings=s)
    assert result["ok"] is False
    assert any("missing experiment fields" in i for i in result["issues"])


def test_preflight_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok")
    monkeypatch.setenv("WANDB_API_KEY", "wkey")
    s = _settings_stub(tmp_path)
    result = preflight_experiment(
        settings=s,
        data_dir_or_kaggle_slug="namdtgk14/data",
        code_version="NamDT-146/repo@main",
        config_path="configs/coco.yaml",
    )
    # May still have env warnings but no blocking issues if fields are filled
    assert result["issues"] == [] or all("missing experiment" not in i for i in result["issues"])


def test_preflight_warns_missing_env(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok")
    s = _settings_stub(tmp_path)
    result = preflight_experiment(
        settings=s,
        data_dir_or_kaggle_slug="slug",
        code_version="v1",
        config_path="cfg",
    )
    # Should warn about missing WANDB_API_KEY
    warning_text = " ".join(result["warnings"])
    assert "WANDB_API_KEY" in warning_text
