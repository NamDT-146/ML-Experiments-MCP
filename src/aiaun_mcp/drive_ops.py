from __future__ import annotations

from pathlib import Path

from aiaun_mcp.config import Settings, redact

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_RETRIES = 5
_QUOTA_HINT = (
    "Google blocks service-account writes to My Drive (no SA quota). "
    "On Linux/SSH servers prefer a Shared Drive folder + SA (no browser). "
    "Or copy .secret/gdrive-token.json from a one-time login; runtime refresh is HTTP-only. "
    "SSH login: python -m aiaun_mcp.drive_login --mode tunnel  (ssh -L 8765:127.0.0.1:8765)"
)


def _oauth_creds(settings: Settings):
    from google.oauth2.credentials import Credentials

    path = Path(settings.drive_oauth_token_json)
    if not path.is_file():
        return None
    creds = Credentials.from_authorized_user_file(str(path), DRIVE_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request

        creds.refresh(Request())
        path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _sa_creds(settings: Settings):
    from google.oauth2 import service_account

    if not settings.drive_sa_json or not Path(settings.drive_sa_json).is_file():
        return None
    return service_account.Credentials.from_service_account_file(
        settings.drive_sa_json,
        scopes=DRIVE_SCOPES,
    )


def build_drive_service(creds):
    from googleapiclient.discovery import build

    return build(
        "drive",
        "v3",
        credentials=creds,
        cache_discovery=False,
        static_discovery=True,
    )


def _build(creds):
    return build_drive_service(creds)


def _service(settings: Settings, *, for_write: bool = False):
    settings.require_drive()
    oauth = _oauth_creds(settings)
    sa = _sa_creds(settings)
    if for_write and oauth:
        return _build(oauth)
    if sa:
        return _build(sa)
    if oauth:
        return _build(oauth)
    raise RuntimeError("no Drive credentials")


def sanitize_folder_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or "").strip())
    return (cleaned or "run")[:120]


def run_folder_name(version: str = "") -> str:
    v = (version or "").strip()
    if not v:
        return "run"
    if v.lower().startswith("v"):
        return sanitize_folder_name(v)
    return sanitize_folder_name("v" + v)


def get_or_create_experiment_folder(svc, parent_id: str, folder_name: str) -> str:
    """Find or create a Drive folder under parent_id (Shared Drive safe)."""
    name = sanitize_folder_name(folder_name)
    qname = name.replace("\\", "\\\\").replace("'", "\\'")
    query = (
        f"'{parent_id}' in parents and name='{qname}' and "
        f"mimeType='{FOLDER_MIME}' and trashed=false"
    )
    results = (
        svc.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute(num_retries=DRIVE_RETRIES)
    )
    items = results.get("files") or []
    if items:
        return str(items[0]["id"])
    created = (
        svc.files()
        .create(
            body={"name": name, "parents": [parent_id], "mimeType": FOLDER_MIME},
            fields="id,name",
            supportsAllDrives=True,
        )
        .execute(num_retries=DRIVE_RETRIES)
    )
    return str(created["id"])


def ensure_run_folder(svc, root_id: str, kernel_slug: str, version: str = "") -> str:
    slug_id = get_or_create_experiment_folder(svc, root_id, kernel_slug.replace("/", "_"))
    return get_or_create_experiment_folder(svc, slug_id, run_folder_name(version))


def folder_info(settings: Settings) -> dict:
    try:
        svc = _service(settings, for_write=False)
        meta = (
            svc.files()
            .get(
                fileId=settings.drive_folder_id,
                fields="id,name,mimeType,driveId",
                supportsAllDrives=True,
            )
            .execute(num_retries=DRIVE_RETRIES)
        )
        return {
            "ok": True,
            "folder": meta,
            "tracking": settings.tracking_links(),
        }
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc))}


def upload_file(
    settings: Settings,
    local_path: str,
    name: str | None = None,
    *,
    parent_id: str | None = None,
) -> dict:
    try:
        from googleapiclient.http import MediaFileUpload

        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            return {"ok": False, "error": f"not a file: {path}"}

        parent = parent_id or settings.drive_folder_id
        body = {
            "name": name or path.name,
            "parents": [parent],
        }
        last_error = None
        oauth = _oauth_creds(settings)
        sa = _sa_creds(settings)
        order = []
        if oauth:
            order.append(("oauth", oauth))
        if sa:
            order.append(("sa", sa))
        if not order:
            return {"ok": False, "error": "no Drive credentials"}

        for label, creds in order:
            try:
                media = MediaFileUpload(str(path), resumable=True)
                svc = _build(creds)
                created = (
                    svc.files()
                    .create(
                        body=body,
                        media_body=media,
                        fields="id,name,webViewLink",
                        supportsAllDrives=True,
                    )
                    .execute(num_retries=DRIVE_RETRIES)
                )
                return {
                    "ok": True,
                    "file": created,
                    "auth": label,
                    "parent_id": parent,
                    "tracking": settings.tracking_links(
                        drive_file_id=str(created.get("id") or ""),
                    ),
                }
            except Exception as exc:
                last_error = redact(str(exc))
                continue
        err = last_error or "upload failed"
        if "storageQuotaExceeded" in err:
            err = err + " | " + _QUOTA_HINT
        return {"ok": False, "error": err}
    except Exception as exc:
        return {"ok": False, "error": redact(str(exc))}
