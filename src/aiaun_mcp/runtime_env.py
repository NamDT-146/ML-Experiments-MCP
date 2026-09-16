"""
Build and push a tiny private Kaggle dataset containing aiaun.env so
kernel API auto-runs can authenticate without UI User Secrets.

Supports multiple local dotenv files (default .env plus .env.* / envs/*.env).
Host MCP still boots from default .env; only the kernel pack switches source.

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
from typing import Any

from dotenv import dotenv_values

ENV_DATASET_SLUG_SUFFIX = "aiaun-run-env"
ENV_FILE_NAME = "aiaun.env"

# Keys always considered when packing from process env / settings
_ENV_KEYS = [
    "WANDB_API_KEY",
    "GITHUB_TOKEN",
    "GOOGLE_DRIVE_FOLDER_ID",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
]

# Never pack host-only credentials into the Kaggle env dataset
_HOST_ONLY_KEYS = frozenset(
    {
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_ORG",
        "AIAUN_OWNER_MODE",
    }
)

_SAFE_RESPONSE_KEYS = frozenset({"WANDB_ENTITY", "WANDB_PROJECT"})


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv file into KEY→VALUE (empty values dropped)."""
    raw = dotenv_values(path)
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k is None or v is None:
            continue
        key = str(k).strip()
        val = str(v).strip()
        if key and val:
            out[key] = val
    return out


def resolve_env_file_path(root: Path, env_file: str) -> Path:
    """
    Resolve env_file relative to repo root. Rejects empty path and path traversal.
    """
    rel = (env_file or "").strip()
    if not rel:
        raise ValueError("env_file is empty")
    root = root.resolve()
    candidate = Path(rel)
    if not candidate.is_absolute():
        candidate = (root / rel).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"env_file must be under repo root: {rel}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"env_file not found: {rel}")
    return candidate


def list_runtime_env_files(root: Path | None = None) -> dict[str, Any]:
    """
    Discover dotenv files under repo root (paths only, no values).
    Includes .env, .env.* (except .env.example), and envs/*.env.
    """
    from aiaun_mcp.config import repo_root

    root = (root or repo_root()).resolve()
    found: list[Path] = []
    default = root / ".env"
    if default.is_file():
        found.append(default)
    for p in sorted(root.glob(".env.*")):
        if not p.is_file():
            continue
        if p.name == ".env.example":
            continue
        found.append(p)
    envs_dir = root / "envs"
    if envs_dir.is_dir():
        for p in sorted(envs_dir.glob("*.env")):
            if p.is_file():
                found.append(p)

    files = []
    seen: set[str] = set()
    for p in found:
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        files.append({"path": rel, "is_default": rel == ".env"})
    return {
        "ok": True,
        "root": str(root),
        "files": files,
        "note": (
            "Pass path as env_file to runtime_env_dataset_push. "
            "If more than .env exists, ask the user which file to pack for Kaggle."
        ),
    }


def _encode_value(value: str) -> tuple[str, str]:
    """Return (key_suffix, encoded_value). Multi-line values use _B64 suffix."""
    if "\n" in value or "\r" in value:
        return "_B64", base64.b64encode(value.encode("utf-8")).decode("ascii")
    return "", value


def _resolve_sa_json_content(raw: str, settings) -> str:
    """If raw is a path to a JSON file, read it; if inline JSON, return as-is."""
    if not raw:
        return ""
    raw = raw.strip()
    p = Path(raw)
    if not p.is_absolute() and hasattr(settings, "root"):
        cand = Path(settings.root) / raw
        if cand.is_file():
            try:
                return cand.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    if raw.lstrip().startswith("{"):
        return raw
    # Try settings.drive_sa_json normalization path
    if hasattr(settings, "drive_sa_json") and settings.drive_sa_json:
        sp = Path(settings.drive_sa_json)
        if sp.is_file() and (raw == settings.drive_sa_json or Path(raw).name == sp.name):
            try:
                return sp.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    return ""


def _apply_sa_resolution(pack: dict[str, str], settings) -> None:
    """Replace GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON path values with file content."""
    raw = pack.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        if settings.drive_sa_json and Path(settings.drive_sa_json).is_file():
            try:
                pack["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"] = (
                    Path(settings.drive_sa_json).read_text(encoding="utf-8").strip()
                )
            except Exception:
                pass
        return
    content = _resolve_sa_json_content(raw, settings)
    if content:
        pack["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"] = content


def build_env_pack(
    settings,
    *,
    env_file: str = "",
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Collect secret key→value pairs for the kernel pack.

    Precedence: overrides > selected env_file > process/settings defaults.
    Host-only keys (Kaggle API token, etc.) are never packed.
    """
    pack: dict[str, str] = {}
    env_file = (env_file or "").strip()

    if env_file:
        path = resolve_env_file_path(Path(settings.root), env_file)
        file_vals = parse_env_file(path)
        for key, val in file_vals.items():
            if key in _HOST_ONLY_KEYS or key.startswith("AIAUN_"):
                continue
            pack[key] = val
    else:
        for key in _ENV_KEYS:
            val = os.environ.get(key, "").strip()
            if val:
                pack[key] = val
        if settings.wandb_entity and "WANDB_ENTITY" not in pack:
            pack["WANDB_ENTITY"] = settings.wandb_entity
        if settings.wandb_project and "WANDB_PROJECT" not in pack:
            pack["WANDB_PROJECT"] = settings.wandb_project
        raw_sa = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
        if raw_sa:
            pack["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"] = raw_sa

    if overrides:
        for key, val in overrides.items():
            if not key or val is None:
                continue
            k = str(key).strip()
            v = str(val).strip()
            if not k or not v:
                continue
            if k in _HOST_ONLY_KEYS or k.startswith("AIAUN_"):
                continue
            pack[k] = v

    _apply_sa_resolution(pack, settings)
    return pack


def build_env_file(pack: dict[str, str]) -> str:
    """Serialize pack to aiaun.env string."""
    lines = ["# AiAuN runtime env — auto-generated; keep this dataset private"]
    for key in sorted(pack.keys()):
        val = pack[key]
        suffix, encoded = _encode_value(val)
        lines.append(f"{key}{suffix}={encoded}")
    return "\n".join(lines) + "\n"


def _safe_effective(pack: dict[str, str]) -> dict[str, str]:
    return {k: pack[k] for k in _SAFE_RESPONSE_KEYS if k in pack and pack[k]}


def runtime_env_dataset_push(
    settings,
    *,
    env_file: str = "",
    overrides: dict[str, str] | None = None,
) -> dict:
    """
    Serialize secrets to aiaun.env and push as private dataset {owner}/aiaun-run-env.

    env_file: optional path under repo root (e.g. .env.coco). Empty = host .env defaults.
    overrides: optional KEY→VALUE patches applied after the file (any non-host keys).
    """
    from aiaun_mcp.config import redact
    from aiaun_mcp.fsutil import write_dataset_metadata
    from aiaun_mcp.kaggle_ops import build_api, dataset_exists

    settings.require_kaggle()
    env_file = (env_file or "").strip()
    try:
        pack = build_env_pack(settings, env_file=env_file, overrides=overrides)
    except (ValueError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}

    if not pack:
        return {
            "ok": False,
            "error": (
                "No secret values found to pack. "
                "Fill WANDB_API_KEY, GITHUB_TOKEN, etc. in the selected env file "
                "or pass overrides before pushing."
            ),
        }

    dataset_slug = f"{settings.kaggle_owner}/{ENV_DATASET_SLUG_SUFFIX}"
    env_text = build_env_file(pack)
    used_file = env_file or ".env"

    with tempfile.TemporaryDirectory(prefix="aiaun_env_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / ENV_FILE_NAME).write_text(env_text, encoding="utf-8")
        write_dataset_metadata(
            tmp_path,
            owner_slug=dataset_slug,
            title="AiAuN runtime env (private — auto-generated)",
            licenses=[{"name": "other"}],
        )
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
        "env_file": used_file,
        "keys_packed": sorted(pack.keys()),
        "effective": _safe_effective(pack),
        "url": f"https://www.kaggle.com/datasets/{dataset_slug}",
        "note": (
            "Attach this slug to kaggle_kernel_push dataset_slugs. "
            "The kernel loads aiaun.env before training. "
            "Secrets are never printed; keep this dataset private. "
            "Use effective.WANDB_* for wandb_run_lookup / tracking links."
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
