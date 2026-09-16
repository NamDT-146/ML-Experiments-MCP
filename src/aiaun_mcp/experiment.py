from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_FIELDS = (
    "data_dir_or_kaggle_slug",
    "code_version",
    "config_path",
)

OPTIONAL_FIELDS = (
    "github_repo",
    "git_branch",
    "wandb_entity",
    "wandb_project",
    "enable_gpu",
)


@dataclass
class ExperimentRequest:
    data_dir_or_kaggle_slug: str = ""
    code_version: str = ""
    config_path: str = ""
    github_repo: str = ""
    git_branch: str = ""
    wandb_entity: str = ""
    wandb_project: str = ""
    enable_gpu: bool | None = None

    def missing(self) -> list[str]:
        missing: list[str] = []
        if not self.data_dir_or_kaggle_slug.strip():
            missing.append("data_dir_or_kaggle_slug")
        if not self.code_version.strip():
            missing.append("code_version")
        if not self.config_path.strip():
            missing.append("config_path")
        return missing

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_dir_or_kaggle_slug": self.data_dir_or_kaggle_slug,
            "code_version": self.code_version,
            "config_path": self.config_path,
            "github_repo": self.github_repo,
            "git_branch": self.git_branch,
            "wandb_entity": self.wandb_entity,
            "wandb_project": self.wandb_project,
            "enable_gpu": self.enable_gpu,
            "missing": self.missing(),
            "ready": not self.missing(),
        }


def resolve_experiment_request(**kwargs: Any) -> dict[str, Any]:
    req = ExperimentRequest(
        data_dir_or_kaggle_slug=str(kwargs.get("data_dir_or_kaggle_slug") or ""),
        code_version=str(kwargs.get("code_version") or ""),
        config_path=str(kwargs.get("config_path") or ""),
        github_repo=str(kwargs.get("github_repo") or ""),
        git_branch=str(kwargs.get("git_branch") or ""),
        wandb_entity=str(kwargs.get("wandb_entity") or ""),
        wandb_project=str(kwargs.get("wandb_project") or ""),
        enable_gpu=kwargs.get("enable_gpu"),
    )
    payload = req.as_dict()
    if payload["missing"]:
        payload["ask_user"] = (
            "Ask the user for: "
            + ", ".join(payload["missing"])
            + ". Do not upload data or push a kernel until these are set. "
            "If several config YAML files match the requested setting, list them and ask which one."
        )
    return payload


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

_ENV_KEYS_FOR_PUSH = [
    "KAGGLE_API_TOKEN",
    "WANDB_API_KEY",
    "GITHUB_TOKEN",
]


def preflight_experiment(
    *,
    settings,
    data_dir_or_kaggle_slug: str = "",
    code_version: str = "",
    config_path: str = "",
    github_repo: str = "",
    required_dataset_paths: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Pre-push sanity check:
    1. Required experiment fields.
    2. .env has secrets for Kaggle + W&B (or env-dataset pack will carry them).
    3. For each dataset slug in required_dataset_paths: check existence + file list.

    required_dataset_paths: {"owner/slug": ["relative/path/needed", ...], ...}
    """
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Resolver fields
    req_payload = resolve_experiment_request(
        data_dir_or_kaggle_slug=data_dir_or_kaggle_slug,
        code_version=code_version,
        config_path=config_path,
        github_repo=github_repo,
    )
    if req_payload.get("missing"):
        issues.append("missing experiment fields: " + ", ".join(req_payload["missing"]))

    # 2. Env keys
    missing_env: list[str] = []
    for key in _ENV_KEYS_FOR_PUSH:
        if not os.environ.get(key, "").strip():
            missing_env.append(key)
    if "GITHUB_TOKEN" in missing_env and not github_repo:
        missing_env.remove("GITHUB_TOKEN")  # not needed if no github_repo
    if missing_env:
        warnings.append(
            f"Keys not in .env: {missing_env}. "
            "Use runtime_env_dataset_push if they should go via env dataset, "
            "or attach them as Kaggle User Secrets via the UI (run_mode=draft)."
        )

    # 3. Dataset content check
    dataset_results: dict[str, Any] = {}
    if required_dataset_paths:
        try:
            from aiaun_mcp.kaggle_ops import build_api, dataset_exists
        except Exception:
            build_api = None  # type: ignore
            dataset_exists = None  # type: ignore

        for slug, paths in (required_dataset_paths or {}).items():
            full_slug = slug if "/" in slug else f"{settings.kaggle_owner}/{slug}"
            check = {"slug": full_slug, "exists": False, "missing_paths": [], "ok": False}
            try:
                ex = dataset_exists(settings, full_slug) if dataset_exists else {"exists": False}
                if not ex.get("exists"):
                    check["exists"] = False
                    issues.append(f"dataset not found: {full_slug}")
                    dataset_results[full_slug] = check
                    continue
                check["exists"] = True
                # Check file list via API
                api = build_api(settings)
                files_resp = api.dataset_list_files(
                    full_slug.split("/")[0],
                    full_slug.split("/")[1],
                )
                file_names: list[str] = []
                if files_resp:
                    raw_files = getattr(files_resp, "datasetFiles", []) or []
                    file_names = [getattr(f, "ref", None) or getattr(f, "name", "") for f in raw_files]
                missing = []
                for req_path in paths:
                    if not any(req_path in fn for fn in file_names):
                        missing.append(req_path)
                check["missing_paths"] = missing
                check["ok"] = not missing
                if missing:
                    issues.append(f"dataset {full_slug} missing paths: {missing}")
            except Exception as exc:
                check["error"] = str(exc)
                warnings.append(f"could not check dataset {full_slug}: {exc}")
            dataset_results[full_slug] = check

    ready = len(issues) == 0
    return {
        "ok": ready,
        "ready": ready,
        "issues": issues,
        "warnings": warnings,
        "dataset_checks": dataset_results,
        "env_keys_missing": missing_env if not ready else [],
        "note": (
            "All clear — safe to call runtime_env_dataset_push then kaggle_kernel_push."
            if ready
            else "Fix issues before pushing. Warnings are advisory."
        ),
    }
