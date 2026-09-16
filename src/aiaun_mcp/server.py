from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from aiaun_mcp.config import ConfigError, get_settings
from aiaun_mcp.drive_ops import folder_info as drive_folder_info_impl
from aiaun_mcp.drive_ops import upload_file as drive_upload_impl
from aiaun_mcp.experiment import preflight_experiment as preflight_impl
from aiaun_mcp.experiment import resolve_experiment_request as resolve_req
from aiaun_mcp.fsutil import inspect_local_dir as inspect_impl
from aiaun_mcp.fsutil import list_repo_configs as list_configs_impl
from aiaun_mcp.kaggle_ops import (
    classify_kernel_failure as classify_impl,
    dataset_exists,
    dataset_push,
    kernel_logs,
    kernel_output_to_drive,
    kernel_push,
    kernel_status,
)
from aiaun_mcp.runtime_env import list_runtime_env_files as list_env_files_impl
from aiaun_mcp.runtime_env import runtime_env_dataset_push as env_push_impl
from aiaun_mcp.templates import (
    GENERATE_NOTEBOOK_PROMPT,
    RUN_KAGGLE_EXPERIMENT_PROMPT,
    kernel_source_has_secrets,
    resnet50_smoke_script,
    smoke_script,
)
from aiaun_mcp.wandb_ops import wandb_run_lookup as wandb_lookup_impl

app = MCPServer("aiaun")


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _load_json_arg(s: str) -> Any:
    return json.loads(s)


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
def list_runtime_env_files() -> str:
    """
    List local dotenv files available to pack for Kaggle (paths only, never values).
    Includes .env, .env.* (except .env.example), and envs/*.env.
    If more than the default .env exists, ask the user which env_file to use.
    """
    try:
        return _json(list_env_files_impl(get_settings().root))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def runtime_env_dataset_push(env_file: str = "", overrides: str = "") -> str:
    """
    Pack secrets into a private Kaggle dataset (aiaun-run-env) for API auto-runs.

    env_file: optional path under repo root (e.g. .env.coco). Empty uses host .env defaults.
    overrides: optional JSON object of KEY→VALUE patches applied after the file
    (any non-host keys, e.g. {"WANDB_PROJECT":"other-proj"}).

    Call list_runtime_env_files first; ask the user which file when alternatives exist.
    Attach the returned dataset_slug to kaggle_kernel_push. Use response.effective
    WANDB_* for wandb_run_lookup / experiment_tracking_links.
    """
    parsed_overrides: dict = {}
    if overrides.strip():
        try:
            raw = _load_json_arg(overrides)
        except Exception:
            return _json({"ok": False, "error": "overrides must be a valid JSON object"})
        if not isinstance(raw, dict):
            return _json({"ok": False, "error": "overrides must be a JSON object"})
        parsed_overrides = {str(k): str(v) for k, v in raw.items() if v is not None}
    try:
        return _json(
            env_push_impl(
                get_settings(),
                env_file=env_file,
                overrides=parsed_overrides or None,
            )
        )
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def preflight_experiment(
    data_dir_or_kaggle_slug: str = "",
    code_version: str = "",
    config_path: str = "",
    github_repo: str = "",
    required_dataset_paths: str = "",
) -> str:
    """
    Pre-push sanity check: validates experiment fields, .env keys, and dataset content.

    required_dataset_paths: JSON string mapping dataset slug to list of required paths,
    e.g. '{"namdtgk14/semi-m2f-real-code": ["mask2former", "splits/voc_hetero_seed42.json"]}'.
    Leave empty to skip dataset content check.
    """
    try:
        parsed_paths: dict = {}
        if required_dataset_paths.strip():
            try:
                parsed_paths = _load_json_arg(required_dataset_paths)
            except Exception:
                return _json({"ok": False, "error": "required_dataset_paths must be valid JSON"})
        return _json(
            preflight_impl(
                settings=get_settings(),
                data_dir_or_kaggle_slug=data_dir_or_kaggle_slug,
                code_version=code_version,
                config_path=config_path,
                github_repo=github_repo,
                required_dataset_paths=parsed_paths or None,
            )
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
    machine_shape: str = "",
    run_mode: str = "auto",
) -> str:
    """
    Push a Kaggle script kernel.

    dataset_slugs: comma-separated owner/slug list. Include the aiaun-run-env slug
    here to pass secrets without UI User Secrets.

    machine_shape: accelerator override (default NvidiaTeslaT4 when enable_gpu=True).
    Other options: NvidiaTeslaT4Highmem, NvidiaTeslaA100, NvidiaL4, NvidiaH100.

    run_mode: 'auto' (push immediately) or 'draft' (write files only, return UI URL).
    Use 'draft' when the kernel needs User Secrets attached via the Kaggle UI first.
    """
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
                machine_shape=machine_shape,
                run_mode=run_mode,
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
    """
    Fetch filtered kernel output logs. Prioritises main *.log and
    artifacts/*_train.log; skips vendor trees; caps each file tail.
    Often only available after the kernel finishes.
    """
    try:
        return _json(kernel_logs(get_settings(), kernel_slug))
    except ConfigError as exc:
        return _json({"ok": False, "error": str(exc)})


@app.tool()
def classify_kernel_failure(log_text: str) -> str:
    """
    Classify a kernel ERROR from log text. Returns error_class, matched lines,
    and a remediation hint. Useful when kaggle_kernel_status returns ERROR and
    failureMessage is null.
    """
    return _json(classify_impl(log_text))


@app.tool()
def wandb_run_lookup(
    run_name: str = "",
    run_id: str = "",
    wandb_project: str = "",
    wandb_entity: str = "",
) -> str:
    """
    Look up recent W&B runs. Fill tracking.wandb_run after a kernel completes.
    Optional wandb_project / wandb_entity override .env (use effective values from
    runtime_env_dataset_push). Uses urllib only — no wandb SDK needed locally.
    """
    try:
        return _json(
            wandb_lookup_impl(
                get_settings(),
                run_name=run_name,
                run_id=run_id,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            )
        )
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
    wandb_project: str = "",
    wandb_entity: str = "",
) -> str:
    """Return Kaggle / W&B / Drive URLs the user can open to track this experiment.
    Optional wandb_project / wandb_entity override .env defaults. Always show these links."""
    settings = get_settings()
    return _json(
        {
            "ok": True,
            "tracking": settings.tracking_links(
                kernel_slug=kernel_slug,
                dataset_slug=dataset_slug,
                wandb_run_id=wandb_run_id,
                drive_file_id=drive_file_id,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
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
