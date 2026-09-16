from __future__ import annotations

import os
import re
import time
from pathlib import Path

from aiaun_mcp.config import Settings, redact
from aiaun_mcp.fsutil import dir_size_bytes, write_dataset_metadata, write_kernel_metadata


def _patch_bearer(api, settings: Settings) -> None:
    os.environ["KAGGLE_API_TOKEN"] = settings.kaggle_token
    os.environ["KAGGLE_USERNAME"] = settings.kaggle_username or settings.kaggle_owner
    api.config_values = {
        "username": settings.kaggle_username or settings.kaggle_owner,
        "key": "unused",
    }


def build_api(settings: Settings):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    _patch_bearer(api, settings)
    try:
        api.authenticate()
    except Exception:
        # Bearer token via env may still work for kagglesdk even if authenticate wants kaggle.json
        pass
    return api


def dataset_exists(settings: Settings, dataset_slug: str) -> dict:
    settings.require_kaggle()
    if "/" not in dataset_slug:
        dataset_slug = f"{settings.kaggle_owner}/{dataset_slug}"
    owner, slug = dataset_slug.split("/", 1)
    api = build_api(settings)
    try:
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetStatusRequest

        req = ApiGetDatasetStatusRequest()
        req.owner_slug = owner
        req.dataset_slug = slug
        with api.build_kaggle_client() as client:
            resp = client.datasets.dataset_api_client.get_dataset_status(req)
        exists = bool(resp and getattr(resp, "status", None))
        return {
            "ok": True,
            "exists": exists,
            "dataset_slug": dataset_slug,
            "status": str(getattr(resp, "status", None)) if exists else None,
        }
    except Exception as exc:
        msg = redact(str(exc)).lower()
        if "404" in msg or "not found" in msg or "403" in msg or "forbidden" in msg:
            return {
                "ok": True,
                "exists": False,
                "dataset_slug": dataset_slug,
                "status": None,
                "note": redact(str(exc)),
            }
        return {"ok": False, "exists": False, "dataset_slug": dataset_slug, "error": redact(str(exc))}


def dataset_push(
    settings: Settings,
    local_path: str,
    dataset_slug: str,
    title: str,
    *,
    force: bool = False,
    version_message: str = "aiaun upload",
) -> dict:
    settings.require_kaggle()
    folder = Path(local_path).expanduser().resolve()
    if not folder.is_dir():
        return {"ok": False, "error": f"not a directory: {folder}"}
    size = dir_size_bytes(folder)
    if "/" not in dataset_slug:
        dataset_slug = f"{settings.kaggle_owner}/{dataset_slug}"
    write_dataset_metadata(folder, owner_slug=dataset_slug, title=title)
    check = dataset_exists(settings, dataset_slug)
    api = build_api(settings)
    try:
        if check.get("exists"):
            api.dataset_create_version(str(folder), version_message, quiet=False, dir_mode="zip")
            action = "version"
        else:
            try:
                api.dataset_create_new(str(folder), quiet=False, dir_mode="zip")
                action = "create"
            except Exception:
                api.dataset_create_version(str(folder), version_message, quiet=False, dir_mode="zip")
                action = "version"
        return {
            "ok": True,
            "action": action,
            "dataset_slug": dataset_slug,
            "bytes": size,
            "url": f"https://www.kaggle.com/datasets/{dataset_slug}",
            "tracking": settings.tracking_links(dataset_slug=dataset_slug),
        }
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc)), "dataset_slug": dataset_slug}


def resolve_kernel_slug(settings: Settings, requested_slug: str, title: str) -> str:
    """
    After kernels_push, Kaggle may rewrite the slug. Try to find the real one.
    Strategy: status the requested slug first; if 404/error, list recent kernels
    and match by title + recency.
    """
    api = build_api(settings)
    try:
        st = api.kernels_status(requested_slug)
        if st and getattr(st, "status", None):
            return requested_slug
    except Exception:
        pass
    # Try listing to find the rewritten slug
    owner = requested_slug.split("/")[0]
    try:
        kernels = api.kernels_list(page=1, page_size=20, user=owner, sort_by="dateCreated")
        if kernels:
            title_lower = title.lower()
            for k in kernels:
                slug_attr = getattr(k, "ref", None) or getattr(k, "id", None) or ""
                title_attr = (getattr(k, "title", None) or "").lower()
                if title_lower in title_attr or title_attr in title_lower:
                    # Strip prefix if any
                    slug_str = str(slug_attr).lstrip("/")
                    if slug_str:
                        return slug_str
    except Exception:
        pass
    return requested_slug


def kernel_push(
    settings: Settings,
    *,
    kernel_slug: str,
    script_content: str,
    dataset_slugs: list[str],
    title: str,
    enable_gpu: bool = False,
    machine_shape: str = "",
    run_mode: str = "auto",
    work_dir: Path | None = None,
) -> dict:
    """
    Push a Kaggle kernel.

    run_mode='draft': write files under .kaggle_work but do NOT call kernels_push.
    run_mode='auto' (default): push immediately.

    machine_shape overrides the accelerator (default NvidiaTeslaT4 when enable_gpu=True).
    """
    settings.require_kaggle()
    if "/" not in kernel_slug:
        kernel_slug = f"{settings.kaggle_owner}/{kernel_slug}"
    work = work_dir or (settings.root / ".kaggle_work" / kernel_slug.replace("/", "_"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "script.py").write_text(script_content, encoding="utf-8")
    write_kernel_metadata(
        work,
        owner_slug=kernel_slug,
        title=title,
        dataset_sources=dataset_slugs,
        enable_gpu=enable_gpu,
        enable_internet=True,
        machine_shape=machine_shape,
    )

    if run_mode == "draft":
        owner = kernel_slug.split("/")[0]
        ui_url = f"https://www.kaggle.com/code/{kernel_slug}"
        return {
            "ok": True,
            "run_mode": "draft",
            "kernel_slug": kernel_slug,
            "url": ui_url,
            "work_dir": str(work),
            "enable_gpu": enable_gpu,
            "machine_shape": machine_shape or ("NvidiaTeslaT4" if enable_gpu else ""),
            "note": (
                "Draft mode: files written but kernel NOT pushed. "
                "Open the Kaggle UI, attach User Secrets, then Save & Run. "
                f"Or call kaggle_kernel_push again with run_mode=auto."
            ),
            "tracking": settings.tracking_links(kernel_slug=kernel_slug),
        }

    api = build_api(settings)
    try:
        api.kernels_push(str(work))
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc)), "kernel_slug": kernel_slug}

    # Resolve the slug Kaggle actually created (may differ from requested)
    resolved_slug = resolve_kernel_slug(settings, kernel_slug, title)
    url = f"https://www.kaggle.com/code/{resolved_slug}"
    return {
        "ok": True,
        "run_mode": "auto",
        "kernel_slug": resolved_slug,
        "requested_slug": kernel_slug,
        "url": url,
        "tracking": settings.tracking_links(kernel_slug=resolved_slug),
        "work_dir": str(work),
        "enable_gpu": enable_gpu,
        "machine_shape": machine_shape or ("NvidiaTeslaT4" if enable_gpu else ""),
    }


def kernel_status(settings: Settings, kernel_slug: str) -> dict:
    settings.require_kaggle()
    if "/" not in kernel_slug:
        kernel_slug = f"{settings.kaggle_owner}/{kernel_slug}"
    api = build_api(settings)
    try:
        st = api.kernels_status(kernel_slug)
        return {
            "ok": True,
            "kernel_slug": kernel_slug,
            "status": str(st),
            "raw": redact(repr(st)),
            "tracking": settings.tracking_links(kernel_slug=kernel_slug),
        }
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc)), "kernel_slug": kernel_slug}


_LOG_PRIORITY_GLOBS = [
    "*.log",
    "artifacts/*_train.log",
    "artifacts/*.log",
]

_VENDOR_SKIP_PATTERNS = [
    "vendor/",
    "thirdparty/",
    "node_modules/",
    ".git/",
]


def _is_vendor(path: Path, root: Path) -> bool:
    rel = str(path.relative_to(root)).replace("\\", "/")
    return any(pat in rel for pat in _VENDOR_SKIP_PATTERNS)


def _priority_files(out_dir: Path) -> list[Path]:
    """Return log files in priority order: main logs first, then artifacts logs."""
    seen: set[Path] = set()
    result: list[Path] = []
    for pattern in _LOG_PRIORITY_GLOBS:
        for p in sorted(out_dir.glob(pattern)):
            if p.is_file() and p not in seen and not _is_vendor(p, out_dir):
                seen.add(p)
                result.append(p)
    # Append any other small non-vendor text files not already captured
    for p in sorted(out_dir.rglob("*")):
        if (
            p.is_file()
            and p not in seen
            and not _is_vendor(p, out_dir)
            and p.stat().st_size < 256_000
        ):
            seen.add(p)
            result.append(p)
    return result


def kernel_logs(
    settings: Settings,
    kernel_slug: str,
    *,
    max_files: int = 6,
    tail_bytes: int = 8000,
) -> dict:
    """
    Fetch kernel Output logs.

    Prioritises main *.log and artifacts/*_train.log; skips vendor trees.
    Returns at most max_files file tails, each capped to tail_bytes.
    """
    settings.require_kaggle()
    if "/" not in kernel_slug:
        kernel_slug = f"{settings.kaggle_owner}/{kernel_slug}"
    out_dir = settings.root / ".kaggle_work" / "logs" / kernel_slug.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    api = build_api(settings)
    try:
        api.kernels_output(kernel_slug, str(out_dir), quiet=True)
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc)), "kernel_slug": kernel_slug}

    priority = _priority_files(out_dir)
    chunks: list[str] = []
    for p in priority[:max_files]:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            tail = text[-tail_bytes:]
            chunks.append(f"--- {p.name} ---\n{tail}")
        except Exception:
            continue

    if not chunks:
        log_tail = "(no log files yet; Kaggle often exposes output only after complete)"
    else:
        log_tail = "\n".join(chunks)

    result = {
        "ok": True,
        "kernel_slug": kernel_slug,
        "log_dir": str(out_dir),
        "log_tail": redact(log_tail),
        "files_shown": [p.name for p in priority[:max_files]],
        "note": "kernels output is often available only after the kernel finishes.",
        "tracking": settings.tracking_links(kernel_slug=kernel_slug),
    }
    # Append metrics/predictions JSON inline if present
    for name in ("metrics.json", "predictions.json"):
        p = out_dir / "artifacts" / name
        if not p.exists():
            p = next(out_dir.rglob(name), None)
        if p and p.is_file() and p.stat().st_size < 20_000:
            try:
                result[f"--- {name} ---"] = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Classify kernel failures from log text
# ---------------------------------------------------------------------------

_FAILURE_PATTERNS: list[tuple[str, str]] = [
    # (pattern, error_class)
    (r"secret .+? unavailable|UserSecretsClient.*unavailable|WANDB_API_KEY.*unavailable", "MISSING_SECRET"),
    (r"FATAL: need T4|FATAL: need gpu|need T4 x2|cuda devices=1.*P100", "WRONG_ACCELERATOR"),
    (r"CUDA error: no kernel image", "CUDA_ARCH_INCOMPATIBLE"),
    (r"ModuleNotFoundError", "MODULE_NOT_FOUND"),
    (r"FileNotFoundError|No such file or directory", "FILE_NOT_FOUND"),
    (r"FileExistsError", "FILE_EXISTS"),
    (r"ImportError", "IMPORT_ERROR"),
    (r"ENOSPC|No space left on device", "DISK_FULL"),
    (r"ConnectionError|connection error", "NETWORK_ERROR"),
    (r"Traceback \(most recent call last\)", "TRACEBACK"),
]


def classify_kernel_failure(log_text: str) -> dict:
    """
    Scan log text and return a classified failure reason.

    Returns:
        {
            "error_class": str,      # first matched class or "UNKNOWN"
            "matched_lines": [str],  # up to 3 lines that triggered the match
            "all_classes": [str],    # all matched classes (de-duplicated)
        }
    """
    classes_found: list[str] = []
    matched_lines: list[str] = []
    for line in log_text.splitlines():
        for pattern, cls in _FAILURE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                if cls not in classes_found:
                    classes_found.append(cls)
                if len(matched_lines) < 6:
                    matched_lines.append(line.strip())
                break

    primary = classes_found[0] if classes_found else "UNKNOWN"
    return {
        "error_class": primary,
        "matched_lines": matched_lines[:6],
        "all_classes": classes_found,
        "remediation": _remediation(primary),
    }


def _remediation(cls: str) -> str:
    return {
        "MISSING_SECRET": (
            "API auto-run has no User Secrets. "
            "Use runtime_env_dataset_push to attach secrets as a private dataset, "
            "or use run_mode=draft and attach secrets via the Kaggle UI."
        ),
        "WRONG_ACCELERATOR": (
            "Kaggle assigned a P100 instead of T4. "
            "Pass machine_shape=NvidiaTeslaT4 (or NvidiaTeslaT4Highmem / NvidiaTeslaA100) "
            "to kaggle_kernel_push."
        ),
        "CUDA_ARCH_INCOMPATIBLE": (
            "The installed PyTorch does not support the assigned GPU compute capability. "
            "The kernel's _ensure_cuda_torch() fallback re-installs cu118 torch for P100."
        ),
        "MODULE_NOT_FOUND": (
            "A Python module is missing. Check that all required packages are installed "
            "in the kernel script (pip install in setup) and that code dataset paths are correct."
        ),
        "FILE_NOT_FOUND": (
            "A required file is missing under /kaggle/input or /kaggle/working. "
            "Run preflight_experiment to check required_paths in attached datasets."
        ),
        "FILE_EXISTS": (
            "A symlink or directory conflict in the working tree. "
            "Common cause: git clone leaves a dangling thirdparty symlink."
        ),
        "DISK_FULL": (
            "Kaggle /kaggle/working is full. "
            "Reduce checkpoint saves; purge work_dirs after each epoch."
        ),
        "NETWORK_ERROR": (
            "Transient network failure (often Kaggle → Google Drive). "
            "Host backup: call kaggle_kernel_output_to_drive after the kernel finishes."
        ),
        "UNKNOWN": "Review the matched_lines for manual diagnosis.",
    }.get(cls, "Review the matched_lines for manual diagnosis.")


def kernel_output_to_drive(settings: Settings, kernel_slug: str) -> dict:
    """Pull Kaggle Output with the Kaggle API, then SA-upload files to Drive (no browser)."""
    from aiaun_mcp.drive_ops import _service, ensure_run_folder, upload_file

    logs = kernel_logs(settings, kernel_slug)
    if not logs.get("ok"):
        return logs
    slug = str(logs.get("kernel_slug") or kernel_slug)
    out_dir = Path(logs["log_dir"])
    parent_id = settings.drive_folder_id
    try:
        svc = _service(settings, for_write=True)
        parent_id = ensure_run_folder(svc, settings.drive_folder_id, slug, "run")
    except Exception as exc:
        return {
            "ok": False,
            "error": redact(str(exc)),
            "kernel_slug": slug,
            "note": "could not create experiment Drive folder",
        }
    uploaded = []
    skipped = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        r = upload_file(settings, str(p), name=p.name, parent_id=parent_id)
        if r.get("ok"):
            uploaded.append({"name": p.name, "id": (r.get("file") or {}).get("id")})
        else:
            skipped.append({"name": p.name, "error": r.get("error")})
    return {
        "ok": True,
        "kernel_slug": slug,
        "log_dir": str(out_dir),
        "uploaded": uploaded,
        "skipped": skipped,
        "drive_run_folder_id": parent_id,
        "note": "Prefer kernel-side Drive upload via User Secrets. This is a host-side backup.",
        "tracking": settings.tracking_links(kernel_slug=slug, experiment_folder_id=parent_id),
    }
