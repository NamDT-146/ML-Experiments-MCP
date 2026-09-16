from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_MODE = "personal"

_SECRET_NAMES = (
    "KAGGLE_API_TOKEN",
    "KAGGLE_KEY",
    "WANDB_API_KEY",
    "GITHUB_TOKEN",
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "docs").is_dir():
            return parent
    return here.parents[2]


def load_env(root: Path | None = None) -> Path:
    root = root or repo_root()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    return root


def redact(text: str) -> str:
    out = text
    for name in _SECRET_NAMES:
        val = os.environ.get(name, "").strip()
        if val and val in out:
            out = out.replace(val, f"<{name}>")
    return out


def _normalize_host_path(p: str, root: Path | None = None) -> str:
    """Resolve secret paths on Windows or WSL from the same .env."""
    if not p:
        return p
    root = root or repo_root()
    posix = p.replace("\\", "/")
    if os.name == "nt" and posix.startswith("/mnt/") and len(posix) > 6:
        bits = posix.split("/")
        if len(bits) >= 4 and bits[1] == "mnt" and len(bits[2]) == 1:
            p = bits[2].upper() + ":\\" + "\\".join(bits[3:])
    elif os.name != "nt" and len(p) >= 3 and p[1] == ":" and p[0].isalpha():
        rest = p[2:].replace("\\", "/").lstrip("/")
        p = f"/mnt/{p[0].lower()}/{rest}"
    path = Path(p)
    if not path.is_absolute():
        path = (root / p).resolve()
        return str(path)
    return str(Path(p))


def _first(*names: str) -> str:
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


@dataclass(frozen=True)
class Settings:
    root: Path
    owner_mode: str
    kaggle_username: str
    kaggle_org: str
    kaggle_token: str
    wandb_entity: str
    wandb_project: str
    wandb_team: str
    wandb_api_key: str
    github_token: str
    github_org: str
    drive_sa_json: str
    drive_folder_id: str
    drive_oauth_client_json: str
    drive_oauth_token_json: str
    max_upload_bytes: int = MAX_UPLOAD_BYTES

    @property
    def kaggle_owner(self) -> str:
        if self.owner_mode == "org":
            if not self.kaggle_org:
                raise ConfigError("AIAUN_OWNER_MODE=org requires KAGGLE_ORG")
            return self.kaggle_org
        if not self.kaggle_username:
            raise ConfigError("KAGGLE_USERNAME is required")
        return self.kaggle_username

    @property
    def wandb_entity_resolved(self) -> str:
        if self.owner_mode == "org":
            return self.wandb_team or self.wandb_entity
        return self.wandb_entity

    def tracking_links(
        self,
        *,
        kernel_slug: str = "",
        dataset_slug: str = "",
        drive_file_id: str = "",
        wandb_run_id: str = "",
        experiment_folder_id: str = "",
        wandb_project: str = "",
        wandb_entity: str = "",
    ) -> dict:
        """URLs the user can open: Kaggle run, W&B project/run, Drive folder/file.

        Optional wandb_project / wandb_entity override .env defaults for this call.
        """
        kaggle_kernel = ""
        if kernel_slug:
            if "/" not in kernel_slug:
                kernel_slug = f"{self.kaggle_owner}/{kernel_slug}"
            kaggle_kernel = f"https://www.kaggle.com/code/{kernel_slug}"
        kaggle_dataset = ""
        if dataset_slug:
            if "/" not in dataset_slug:
                dataset_slug = f"{self.kaggle_owner}/{dataset_slug}"
            kaggle_dataset = f"https://www.kaggle.com/datasets/{dataset_slug}"
        entity = (wandb_entity or "").strip() or self.wandb_entity_resolved
        project = (wandb_project or "").strip() or self.wandb_project
        wandb_project_url = ""
        wandb_run = ""
        if entity and project:
            wandb_project_url = f"https://wandb.ai/{entity}/{project}"
            if wandb_run_id:
                wandb_run = f"{wandb_project_url}/runs/{wandb_run_id}"
        drive_folder = ""
        folder_id = experiment_folder_id or self.drive_folder_id
        if folder_id:
            drive_folder = f"https://drive.google.com/drive/folders/{folder_id}"
        drive_file = ""
        if drive_file_id:
            drive_file = f"https://drive.google.com/file/d/{drive_file_id}/view"
        return {
            "kaggle_kernel": kaggle_kernel,
            "kaggle_kernel_versions": f"{kaggle_kernel}?scriptVersionId=latest" if kaggle_kernel else "",
            "kaggle_dataset": kaggle_dataset,
            "wandb_project": wandb_project_url,
            "wandb_run": wandb_run,
            "drive_folder": drive_folder,
            "drive_file": drive_file,
        }

    def require_kaggle(self) -> None:
        if not self.kaggle_token:
            raise ConfigError(
                "Kaggle credentials missing: set KAGGLE_API_TOKEN (or KAGGLE_KEY) in .env"
            )
        if not self.kaggle_username and self.owner_mode != "org":
            raise ConfigError("KAGGLE_USERNAME is required")

    def require_drive(self) -> None:
        if not self.drive_folder_id:
            raise ConfigError("GOOGLE_DRIVE_FOLDER_ID is required")
        sa = Path(self.drive_sa_json) if self.drive_sa_json else None
        tok = Path(self.drive_oauth_token_json) if self.drive_oauth_token_json else None
        if not ((sa and sa.is_file()) or (tok and tok.is_file())):
            raise ConfigError(
                "Drive needs GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON and/or a user OAuth token "
                "(run: python -m aiaun_mcp.drive_login)"
            )


class ConfigError(RuntimeError):
    pass


def get_settings(root: Path | None = None) -> Settings:
    root = load_env(root)
    mode = os.environ.get("AIAUN_OWNER_MODE", DEFAULT_MODE).strip().lower() or DEFAULT_MODE
    if mode not in {"personal", "org"}:
        raise ConfigError("AIAUN_OWNER_MODE must be 'personal' or 'org'")
    return Settings(
        root=root,
        owner_mode=mode,
        kaggle_username=_first("KAGGLE_USERNAME"),
        kaggle_org=_first("KAGGLE_ORG"),
        kaggle_token=_first("KAGGLE_API_TOKEN", "KAGGLE_KEY"),
        wandb_entity=_first("WANDB_ENTITY"),
        wandb_project=_first("WANDB_PROJECT"),
        wandb_team=_first("WANDB_TEAM"),
        wandb_api_key=_first("WANDB_API_KEY"),
        github_token=_first("GITHUB_TOKEN"),
        github_org=_first("GITHUB_ORG"),
        drive_sa_json=_normalize_host_path(_first("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"), root),
        drive_folder_id=_first("GOOGLE_DRIVE_FOLDER_ID"),
        drive_oauth_client_json=_normalize_host_path(
            _first("GOOGLE_DRIVE_OAUTH_CLIENT_JSON"), root
        ),
        drive_oauth_token_json=_normalize_host_path(
            _first("GOOGLE_DRIVE_OAUTH_TOKEN_JSON"), root
        )
        or str(root / ".secret" / "gdrive-token.json"),
    )
