from __future__ import annotations

from dataclasses import dataclass, field
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
