from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from aiaun_mcp.config import ConfigError, get_settings
from aiaun_mcp.drive_ops import folder_info as drive_folder_info_impl
from aiaun_mcp.drive_ops import upload_file as drive_upload_impl
from aiaun_mcp.experiment import resolve_experiment_request as resolve_req
from aiaun_mcp.fsutil import inspect_local_dir as inspect_impl
from aiaun_mcp.fsutil import list_repo_configs as list_configs_impl
from aiaun_mcp.kaggle_ops import (
    dataset_exists,
    dataset_push,
    kernel_logs,
    kernel_output_to_drive,
    kernel_push,
    kernel_status,
)
from aiaun_mcp.templates import (
    GENERATE_NOTEBOOK_PROMPT,
    RUN_KAGGLE_EXPERIMENT_PROMPT,
    kernel_source_has_secrets,
    resnet50_smoke_script,
    smoke_script,
)

app = MCPServer("aiaun")


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


@app.tool()
def resolve_experiment_request(
    data_dir_or_kaggle_slug: str = "",
    code_version: str = "",
    config_path: str = "",
    github_repo: str = "",
    git_branch: str = "",
    wandb_entity: str = "",
    wandb_project: str = "",
) -> str:
    """Check required experiment fields. If missing is non-empty, ask the user; do not run other tools yet."""
    return _json(
        resolve_req(
            data_dir_or_kaggle_slug=data_dir_or_kaggle_slug,
            code_version=code_version,
            config_path=config_path,
            github_repo=github_repo,
            git_branch=git_branch,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
    )


@app.tool()
def inspect_local_dir(path: str) -> str:
    """Inspect a local data directory: file count, total bytes, sample names."""
    return _json(inspect_impl(path))


@app.tool()
def list_repo_configs() -> str:
    """List experiment YAML configs under semi-mask2former/configs."""
    settings = get_settings()
    return _json({"ok": True, "configs": list_configs_impl(settings.root)})


@app.tool()
def kaggle_dataset_check(dataset_slug: str) -> str:
    """Check whether a Kaggle dataset owner/slug already exists."""
    try:
        return _json(dataset_exists(get_settings(), dataset_slug))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def kaggle_dataset_push(
    local_path: str,
    dataset_slug: str,
    title: str,
    force: bool = False,
) -> str:
    """Upload a local folder as a Kaggle dataset. Size is limited only by the Kaggle API."""
    try:
        return _json(
            dataset_push(get_settings(), local_path, dataset_slug, title, force=force)
        )
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def kaggle_kernel_push(
    kernel_slug: str,
    script_content: str,
    dataset_slugs: str,
    title: str,
    enable_gpu: bool = False,
) -> str:
    """Push a Kaggle script kernel. dataset_slugs is a comma-separated owner/slug list."""
    if kernel_source_has_secrets(script_content):
        return _json(
            {"ok": False, "error": "refusing to push script that looks like it contains secrets"}
        )
    slugs = [s.strip() for s in dataset_slugs.split(",") if s.strip()]
    try:
        return _json(
            kernel_push(
                get_settings(),
                kernel_slug=kernel_slug,
                script_content=script_content,
                dataset_slugs=slugs,
                title=title,
                enable_gpu=enable_gpu,
            )
        )
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def kaggle_kernel_status(kernel_slug: str) -> str:
    """Poll Kaggle kernel execution status."""
    try:
        return _json(kernel_status(get_settings(), kernel_slug))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def kaggle_kernel_logs(kernel_slug: str) -> str:
    """Fetch kernel output logs (often only after complete)."""
    try:
        return _json(kernel_logs(get_settings(), kernel_slug))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def aiaun_smoke_script(dataset_slug: str = "aiaun-synthetic-color-cls") -> str:
    """Return the inlined synthetic color-class smoke kernel source (no secrets)."""
    settings = get_settings()
    if "/" not in dataset_slug:
        dataset_slug = "{}/{}".format(settings.kaggle_owner, dataset_slug)
    src = smoke_script(dataset_slug=dataset_slug)
    return _json({"ok": True, "dataset_slug": dataset_slug, "script_content": src, "kernel_slug": "aiaun-color-smoke"})


@app.tool()
def aiaun_resnet50_gpu_smoke_script(dataset_slug: str = "aiaun-synthetic-color-cls") -> str:
    """ResNet50 GPU smoke: short train, inference on best.pt, Drive upload of artifacts (no secrets in source). Push with enable_gpu=true."""
    settings = get_settings()
    if "/" not in dataset_slug:
        dataset_slug = "{}/{}".format(settings.kaggle_owner, dataset_slug)
    src = resnet50_smoke_script(dataset_slug=dataset_slug)
    return _json(
        {
            "ok": True,
            "dataset_slug": dataset_slug,
            "enable_gpu": True,
            "script_content": src,
            "tracking": settings.tracking_links(dataset_slug=dataset_slug),
            "note": "kaggle_kernel_push with enable_gpu=true (T4). Attach synthetic color dataset; Drive via User Secrets. Show tracking.kaggle_kernel / wandb_project / drive_folder.",
        }
    )


@app.tool()
def experiment_tracking_links(
    kernel_slug: str = "",
    dataset_slug: str = "",
    wandb_run_id: str = "",
    drive_file_id: str = "",
) -> str:
    """Return Kaggle / W&B / Drive URLs the user can open to track this experiment. Always show these links in the chat."""
    settings = get_settings()
    return _json(
        {
            "ok": True,
            "tracking": settings.tracking_links(
                kernel_slug=kernel_slug,
                dataset_slug=dataset_slug,
                wandb_run_id=wandb_run_id,
                drive_file_id=drive_file_id,
            ),
        }
    )


@app.tool()
def drive_folder_info() -> str:
    """Verify Google Drive service-account access to GOOGLE_DRIVE_FOLDER_ID."""
    try:
        return _json(drive_folder_info_impl(get_settings()))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def drive_upload_file(local_path: str, name: str = "") -> str:
    """Upload a file to the configured Drive folder (Shared Drive + SA; no browser)."""
    try:
        return _json(drive_upload_impl(get_settings(), local_path, name or None))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def kaggle_kernel_output_to_drive(kernel_slug: str) -> str:
    """Download Kaggle kernel Output then upload to Drive via the local service account (SSH-safe, no browser). Prefer kernel-side User Secrets upload when possible."""
    try:
        return _json(kernel_output_to_drive(get_settings(), kernel_slug))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.prompt()
def run_kaggle_experiment() -> str:
    """Workflow for running an experiment on Kaggle via this MCP."""
    return RUN_KAGGLE_EXPERIMENT_PROMPT


@app.prompt()
def generate_experiment_notebook() -> str:
    """Guide for writing a Kaggle script that clones GitHub and uses User Secrets."""
    return GENERATE_NOTEBOOK_PROMPT


def main() -> None:
    try:
        get_settings()
    except Exception as exc:
        print("aiaun_mcp settings warning: {}".format(exc), file=sys.stderr)
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
