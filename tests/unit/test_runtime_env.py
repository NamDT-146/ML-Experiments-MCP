from __future__ import annotations

import base64
from pathlib import Path

import pytest

from aiaun_mcp.runtime_env import (
    KERNEL_ENV_LOADER,
    _encode_value,
    build_env_file,
    build_env_pack,
    list_runtime_env_files,
    parse_env_file,
    resolve_env_file_path,
)


def _minimal_settings(tmp_path, **overrides):
    """Return a simple namespace with the fields build_env_pack reads."""

    class S:
        root = overrides.get("root", tmp_path)
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
    assert base64.b64decode(enc).decode() == "line1\nline2"


def test_parse_env_file(tmp_path):
    p = tmp_path / ".env.coco"
    p.write_text("WANDB_PROJECT=coco-proj\nWANDB_API_KEY=key1\n# comment\nEMPTY=\n")
    got = parse_env_file(p)
    assert got["WANDB_PROJECT"] == "coco-proj"
    assert got["WANDB_API_KEY"] == "key1"
    assert "EMPTY" not in got


def test_resolve_env_file_path_ok(tmp_path):
    p = tmp_path / ".env.x"
    p.write_text("A=1\n")
    assert resolve_env_file_path(tmp_path, ".env.x") == p.resolve()


def test_resolve_env_file_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="under repo root"):
        resolve_env_file_path(tmp_path, "../outside.env")


def test_resolve_env_file_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_env_file_path(tmp_path, ".env.missing")


def test_list_runtime_env_files(tmp_path):
    (tmp_path / ".env").write_text("A=1\n")
    (tmp_path / ".env.coco").write_text("B=2\n")
    (tmp_path / ".env.example").write_text("C=3\n")
    envs = tmp_path / "envs"
    envs.mkdir()
    (envs / "other.env").write_text("D=4\n")
    out = list_runtime_env_files(tmp_path)
    assert out["ok"] is True
    paths = [f["path"] for f in out["files"]]
    assert ".env" in paths
    assert ".env.coco" in paths
    assert "envs/other.env" in paths
    assert ".env.example" not in paths
    default = next(f for f in out["files"] if f["path"] == ".env")
    assert default["is_default"] is True
    # paths only — no secret values
    assert "A=1" not in str(out)
    assert "B=2" not in str(out)


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


def test_build_env_pack_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    alt = tmp_path / ".env.coco"
    alt.write_text(
        "WANDB_API_KEY=fromfile\n"
        "WANDB_PROJECT=coco-proj\n"
        "WANDB_ENTITY=team-x\n"
        "CUSTOM_FLAG=yes\n"
        "KAGGLE_API_TOKEN=should_not_pack\n"
    )
    s = _minimal_settings(tmp_path, wandb_entity="default-ent", wandb_project="default-proj")
    pack = build_env_pack(s, env_file=".env.coco")
    assert pack["WANDB_API_KEY"] == "fromfile"
    assert pack["WANDB_PROJECT"] == "coco-proj"
    assert pack["WANDB_ENTITY"] == "team-x"
    assert pack["CUSTOM_FLAG"] == "yes"
    assert "KAGGLE_API_TOKEN" not in pack


def test_build_env_pack_overrides_win(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    alt = tmp_path / ".env.x"
    alt.write_text("WANDB_PROJECT=fromfile\nWANDB_API_KEY=k1\n")
    s = _minimal_settings(tmp_path)
    pack = build_env_pack(
        s,
        env_file=".env.x",
        overrides={"WANDB_PROJECT": "overridden", "EXTRA": "1"},
    )
    assert pack["WANDB_PROJECT"] == "overridden"
    assert pack["EXTRA"] == "1"
    assert pack["WANDB_API_KEY"] == "k1"


def test_build_env_pack_overrides_reject_host_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    s = _minimal_settings(tmp_path, wandb_project="p", wandb_entity="e")
    pack = build_env_pack(s, overrides={"KAGGLE_API_TOKEN": "leak", "AIAUN_OWNER_MODE": "org"})
    assert "KAGGLE_API_TOKEN" not in pack
    assert "AIAUN_OWNER_MODE" not in pack


def test_build_env_pack_resolves_sa_json_file(monkeypatch, tmp_path):
    sa = tmp_path / "sa.json"
    sa.write_text('{"type":"service_account"}')
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    s = _minimal_settings(tmp_path, drive_sa_json=str(sa))
    pack = build_env_pack(s)
    assert "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON" in pack
    assert "service_account" in pack["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"]


def test_build_env_pack_sa_path_in_file(monkeypatch, tmp_path):
    sa = tmp_path / "sa.json"
    sa.write_text('{"type":"service_account","id":"x"}')
    alt = tmp_path / ".env.drive"
    alt.write_text(f"GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON={sa}\nGOOGLE_DRIVE_FOLDER_ID=folder1\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    s = _minimal_settings(tmp_path)
    pack = build_env_pack(s, env_file=".env.drive")
    assert pack["GOOGLE_DRIVE_FOLDER_ID"] == "folder1"
    assert '"type":"service_account"' in pack["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"]


def test_build_env_file_no_secrets_in_output(monkeypatch):
    pack = {"WANDB_API_KEY": "secret123", "GITHUB_TOKEN": "ghp_abc"}
    env_text = build_env_file(pack)
    assert "WANDB_API_KEY=secret123" in env_text or "WANDB_API_KEY_B64=" in env_text
    assert "GITHUB_TOKEN=" in env_text


def test_build_env_file_multiline_encoded(monkeypatch):
    pack = {"GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"type":"sa"\n,"key":"val"}'}
    env_text = build_env_file(pack)
    assert "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON_B64=" in env_text
    b64_part = [
        l.split("=", 1)[1]
        for l in env_text.splitlines()
        if l.startswith("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON_B64=")
    ][0]
    decoded = base64.b64decode(b64_part).decode()
    assert '"type":"sa"' in decoded


def test_kernel_env_loader_compiles():
    compile(KERNEL_ENV_LOADER, "<env_loader>", "exec")


def test_kernel_env_loader_no_secrets():
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
