"""Google user OAuth for Drive uploads. Works on SSH (no local browser)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from aiaun_mcp.config import get_settings
from aiaun_mcp.drive_ops import DRIVE_SCOPES


def is_headless() -> bool:
    """True on typical SSH/no-display Linux servers (no GUI browser)."""
    if os.environ.get("AIAUN_DRIVE_OAUTH_MODE"):
        return os.environ["AIAUN_DRIVE_OAUTH_MODE"].strip().lower() in {
            "console",
            "tunnel",
            "headless",
        }
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get(
        "WAYLAND_DISPLAY"
    ):
        return True
    return False


def _client_json(settings) -> Path:
    client = Path(settings.drive_oauth_client_json) if settings.drive_oauth_client_json else Path()
    if client.is_file():
        return client
    default = settings.root / ".secret" / "gdrive-oauth-client.json"
    if default.is_file():
        return default
    raise SystemExit(
        "Missing OAuth Desktop client JSON.\n"
        "Google Cloud → Credentials → Create OAuth client ID → Desktop app.\n"
        f"Save as {default}\n"
        "On a headless server you can instead copy an already-authorized "
        f"{settings.drive_oauth_token_json} from a machine that ran this login."
    )


def _save(settings, creds) -> Path:
    dest = Path(settings.drive_oauth_token_json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(creds.to_json(), encoding="utf-8")
    print("Wrote", dest)
    print("Copy this file to other SSH hosts; MCP refreshes it without a browser.")
    return dest


def run_oauth(*, mode: str, port: int) -> None:
    settings = get_settings()
    client = _client_json(settings)
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client), DRIVE_SCOPES)

    if mode == "auto":
        mode = "tunnel" if is_headless() else "local"

    if mode == "local":
        print("Opening local browser on port {}...".format(port))
        creds = flow.run_local_server(port=port, open_browser=True, bind_addr="127.0.0.1")
    elif mode == "tunnel":
        print(
            "Headless / SSH login (no GUI on this machine).\n"
            "From your laptop run:\n"
            f"  ssh -L {port}:127.0.0.1:{port} <user>@<this-host>\n"
            "Then open the URL printed next in the laptop browser.\n"
            "Alternatively: run this command on a laptop with a browser and copy "
            f"{settings.drive_oauth_token_json} onto the server.\n"
        )
        creds = flow.run_local_server(port=port, open_browser=False, bind_addr="127.0.0.1")
    elif mode == "console":
        print(
            "Console mode: open the URL on any device, then paste the code here.\n"
            "If Google rejects the out-of-band client, use --mode tunnel instead.\n"
        )
        creds = flow.run_console()
    else:
        raise SystemExit("mode must be auto|local|tunnel|console")

    _save(settings, creds)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Authorize Google Drive user OAuth (SSH-safe).")
    p.add_argument(
        "--mode",
        choices=("auto", "local", "tunnel", "console"),
        default="auto",
        help="auto: tunnel on SSH/no DISPLAY, else local browser. "
        "tunnel: print URL, ssh -L. console: paste code. local: open GUI.",
    )
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    run_oauth(mode=args.mode, port=args.port)


if __name__ == "__main__":
    main()
