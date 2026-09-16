"""
Build and push a tiny private Kaggle dataset containing aiaun.env so
kernel API auto-runs can authenticate without UI User Secrets.

aiaun.env format: KEY=VALUE one per line.
Multi-line values (e.g. SA JSON) are stored as KEY_B64=<base64>.

KERNEL_ENV_LOADER is a snippet inlined into kernel scripts. It runs first,
loading env vars from /kaggle/input/**/aiaun.env, then falls back to
UserSecretsClient for any key still missing.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

ENV_DATASET_SLUG_SUFFIX = "aiaun-run-env"
ENV_FILE_NAME = "aiaun.env"

# Keys pulled from os.environ (populated from .env by load_env at startup)
_ENV_KEYS = [
    "WANDB_API_KEY",
    "GITHUB_TOKEN",
    "GOOGLE_DRIVE_FOLDER_ID",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
]


def _encode_value(value: str) -> tuple[str, str]:
    """Return (key_suffix, encoded_value). Multi-line values use _B64 suffix."""
    if "\n" in value or "\r" in value:
        return "_B64", base64.b64encode(value.encode("utf-8")).decode("ascii")
    return "", value


def build_env_pack(settings) -> dict[str, str]:
    """
    Collect secret key→value pairs from settings/os.environ.
    For GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON resolves file path to JSON content.
    """
    pack: dict[str, str] = {}
    for key in _ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            pack[key] = val
    # Ensure WANDB_ENTITY/PROJECT from settings fields if not in env
    if settings.wandb_entity and "WANDB_ENTITY" not in pack:
        pack["WANDB_ENTITY"] = settings.wandb_entity
    if settings.wandb_project and "WANDB_PROJECT" not in pack:
        pack["WANDB_PROJECT"] = settings.wandb_project

    # SA JSON: prefer resolved path from settings, then inline env value
    sa_json_content = ""
    if settings.drive_sa_json and Path(settings.drive_sa_json).is_file():
        try:
            sa_json_content = Path(settings.drive_sa_json).read_text(encoding="utf-8").strip()
        except Exception:
            pass
    if not sa_json_content:
        raw = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
        if raw:
            p = Path(raw)
            if p.is_file():
                try:
                    sa_json_content = p.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            elif raw.lstrip().startswith("{"):
                sa_json_content = raw
    if sa_json_content:
        pack["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"] = sa_json_content

    return pack


def build_env_file(pack: dict[str, str]) -> str:
    """Serialize pack to aiaun.env string."""
    lines = ["# AiAuN runtime env — auto-generated; keep this dataset private"]
    for key in sorted(pack.keys()):
        val = pack[key]
        suffix, encoded = _encode_value(val)
        lines.append(f"{key}{suffix}={encoded}")
    return "\n".join(lines) + "\n"


def runtime_env_dataset_push(settings) -> dict:
    """
    Serialize .env secrets to aiaun.env and push as private dataset
    {owner}/aiaun-run-env. Returns dataset slug to attach to the next kernel.
    """
    from aiaun_mcp.config import redact
    from aiaun_mcp.fsutil import write_dataset_metadata
    from aiaun_mcp.kaggle_ops import build_api, dataset_exists

    settings.require_kaggle()
    pack = build_env_pack(settings)
    if not pack:
        return {
            "ok": False,
            "error": (
                "No secret values found in .env. "
                "Fill WANDB_API_KEY, GITHUB_TOKEN, etc. before pushing env pack."
            ),
        }

    dataset_slug = f"{settings.kaggle_owner}/{ENV_DATASET_SLUG_SUFFIX}"
    env_text = build_env_file(pack)

    with tempfile.TemporaryDirectory(prefix="aiaun_env_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / ENV_FILE_NAME).write_text(env_text, encoding="utf-8")
        write_dataset_metadata(
            tmp_path,
            owner_slug=dataset_slug,
            title="AiAuN runtime env (private — auto-generated)",
            licenses=[{"name": "other"}],
        )
        # Mark private in metadata
        meta_path = tmp_path / "dataset-metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["isPrivate"] = True
        meta_path.write_text(json.dumps(meta, indent=2))

        check = dataset_exists(settings, dataset_slug)
        api = build_api(settings)
        try:
            if check.get("exists"):
                api.dataset_create_version(
                    str(tmp_path), "aiaun env refresh", quiet=False, dir_mode="zip"
                )
                action = "version"
            else:
                try:
                    api.dataset_create_new(str(tmp_path), quiet=False, dir_mode="zip")
                    action = "create"
                except Exception:
                    api.dataset_create_version(
                        str(tmp_path), "aiaun env refresh", quiet=False, dir_mode="zip"
                    )
                    action = "version"
        except Exception as exc:
            return {"ok": False, "error": redact(str(exc)), "dataset_slug": dataset_slug}

    return {
        "ok": True,
        "action": action,
        "dataset_slug": dataset_slug,
        "keys_packed": sorted(pack.keys()),
        "url": f"https://www.kaggle.com/datasets/{dataset_slug}",
        "note": (
            "Attach this slug to kaggle_kernel_push dataset_slugs. "
            "The kernel loads aiaun.env before training. "
            "Secrets are never printed; keep this dataset private."
        ),
    }


# ---------------------------------------------------------------------------
# Kernel-side snippet — inlined in script, no aiaun_mcp import
# ---------------------------------------------------------------------------
KERNEL_ENV_LOADER = r'''
import base64 as _b64, os as _os, pathlib as _pathlib


def _load_aiaun_env() -> None:
    """Load secrets from private aiaun.env dataset; fall back to Kaggle User Secrets."""
    env_files = sorted(_pathlib.Path("/kaggle/input").glob("**/aiaun.env"))
    loaded: list[str] = []
    if env_files:
        for line in env_files[0].read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k.endswith("_B64"):
                k = k[:-4]
                try:
                    v = _b64.b64decode(v).decode("utf-8")
                except Exception:
                    pass
            if not _os.environ.get(k):
                _os.environ[k] = v
                loaded.append(k)
        print("AIAUN_ENV loaded:", loaded, flush=True)
    # Fall back to Kaggle User Secrets for any still-missing key
    _want = (
        "WANDB_API_KEY",
        "GITHUB_TOKEN",
        "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_DRIVE_FOLDER_ID",
    )
    _missing = [k for k in _want if not _os.environ.get(k)]
    if _missing:
        try:
            from kaggle_secrets import UserSecretsClient as _USC
            _us = _USC()
            for k in _missing:
                try:
                    _os.environ.setdefault(k, _us.get_secret(k))
                    print(f"AIAUN_ENV fallback User Secret: {k}", flush=True)
                except Exception:
                    pass
        except Exception:
            pass


_load_aiaun_env()
'''
