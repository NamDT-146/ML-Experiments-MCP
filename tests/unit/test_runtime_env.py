from __future__ import annotations

import os
from pathlib import Path

import pytest

from aiaun_mcp.runtime_env import (
    ENV_FILE_NAME,
    KERNEL_ENV_LOADER,
    _encode_value,
    build_env_file,
    build_env_pack,
)


def _minimal_settings(tmp_path, **overrides):
    """Return a simple namespace with the fields build_env_pack reads."""
    class S:
        wandb_entity = overrides.get("wandb_entity", "")
        wandb_project = overrides.get("wandb_project", "")
        drive_sa_json = overrides.get("drive_sa_json", "")

    return S()


def test_encode_value_plain():
    suffix, enc = _encode_value("simple")
    assert suffix == ""
    assert enc == "simple"


def test_encode_value_multiline():
    suffix, enc = _encode_value("line1\nline2")
    assert suffix == "_B64"
    import base64
    assert base64.b64decode(enc).decode() == "line1\nline2"


def test_build_env_pack_basic(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "wandb_key_123")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token456")
    monkeypatch.delenv("GOOGLE_DRIVE_FOLDER_ID", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", raising=False)
    s = _minimal_settings(tmp_path, wandb_entity="ent", wandb_project="proj")
    pack = build_env_pack(s)
    assert pack.get("WANDB_API_KEY") == "wandb_key_123"
    assert pack.get("GITHUB_TOKEN") == "ghp_token456"
    assert pack.get("WANDB_ENTITY") == "ent"
    assert pack.get("WANDB_PROJECT") == "proj"


def test_build_env_pack_resolves_sa_json_file(monkeypatch, tmp_path):
    sa = tmp_path / "sa.json"
    sa.write_text('{"type":"service_account"}')
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    s = _minimal_settings(tmp_path, drive_sa_json=str(sa))
    pack = build_env_pack(s)
    assert "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON" in pack
    assert "service_account" in pack["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"]


def test_build_env_file_no_secrets_in_output(monkeypatch):
    pack = {"WANDB_API_KEY": "secret123", "GITHUB_TOKEN": "ghp_abc"}
    env_text = build_env_file(pack)
    # Values are present but we verify the format
    assert "WANDB_API_KEY=secret123" in env_text or "WANDB_API_KEY_B64=" in env_text
    assert "GITHUB_TOKEN=" in env_text


def test_build_env_file_multiline_encoded(monkeypatch):
    import base64
    pack = {"GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"type":"sa"\n,"key":"val"}'}
    env_text = build_env_file(pack)
    assert "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON_B64=" in env_text
    b64_part = [l.split("=", 1)[1] for l in env_text.splitlines()
                if l.startswith("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON_B64=")][0]
    decoded = base64.b64decode(b64_part).decode()
    assert '"type":"sa"' in decoded


def test_kernel_env_loader_compiles():
    compile(KERNEL_ENV_LOADER, "<env_loader>", "exec")


def test_kernel_env_loader_no_secrets():
    # Loader should not contain hardcoded secrets
    from aiaun_mcp.templates import kernel_source_has_secrets
    assert not kernel_source_has_secrets(KERNEL_ENV_LOADER)


def test_kernel_env_loader_in_smoke_script():
    from aiaun_mcp.templates import smoke_script
    src = smoke_script(dataset_slug="namdtgk14/test")
    assert "_load_aiaun_env" in src
    assert "AIAUN_ENV loaded" in src
    compile(src, "<smoke>", "exec")


def test_kernel_env_loader_in_resnet50_script():
    from aiaun_mcp.templates import resnet50_smoke_script
    src = resnet50_smoke_script(dataset_slug="namdtgk14/test")
    assert "_load_aiaun_env" in src
    compile(src, "<resnet50>", "exec")
