from __future__ import annotations

import os
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


def kernel_push(
    settings: Settings,
    *,
    kernel_slug: str,
    script_content: str,
    dataset_slugs: list[str],
    title: str,
    enable_gpu: bool = False,
    work_dir: Path | None = None,
) -> dict:
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
    )
    api = build_api(settings)
    try:
        api.kernels_push(str(work))
        url = f"https://www.kaggle.com/code/{kernel_slug}"
        return {
            "ok": True,
            "kernel_slug": kernel_slug,
            "url": url,
            "tracking": settings.tracking_links(kernel_slug=kernel_slug),
            "work_dir": str(work),
            "enable_gpu": enable_gpu,
        }
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc)), "kernel_slug": kernel_slug}


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


def kernel_logs(settings: Settings, kernel_slug: str) -> dict:
    settings.require_kaggle()
    if "/" not in kernel_slug:
        kernel_slug = f"{settings.kaggle_owner}/{kernel_slug}"
    out_dir = settings.root / ".kaggle_work" / "logs" / kernel_slug.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    api = build_api(settings)
    try:
        api.kernels_output(kernel_slug, str(out_dir), quiet=True)
        chunks: list[str] = []
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p.stat().st_size < 512_000:
                try:
                    chunks.append(f"--- {p.name} ---\n" + p.read_text(encoding="utf-8", errors="replace")[-8000:])
                except Exception:
                    continue
        text = "\n".join(chunks) if chunks else "(no log files yet; Kaggle often exposes output only after complete)"
        return {
            "ok": True,
            "kernel_slug": kernel_slug,
            "log_dir": str(out_dir),
            "log_tail": redact(text),
            "note": "kernels output is often available only after the kernel finishes.",
            "tracking": settings.tracking_links(kernel_slug=kernel_slug),
        }
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc)), "kernel_slug": kernel_slug}


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
