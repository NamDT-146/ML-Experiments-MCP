"""Windows stdio shim: Cursor spawns repo-relative python.exe, then we exec WSL Python.

Cursor's MCP host is Win32. Absolute paths with a drive colon or /home/...
often fail with "The system cannot find the path specified."
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WSL = os.environ.get("AIAUN_WSL_EXE", r"C:\Windows\System32\wsl.exe")
DISTRO = os.environ.get("AIAUN_WSL_DISTRO", "Ubuntu")


def _to_wsl_path(path: Path) -> str:
    s = str(path.resolve())
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


def _linux_python() -> str:
    env = os.environ.get("AIAUN_WSL_PYTHON", "").strip()
    if env:
        return env
    marker = ROOT / ".aiaun-python"
    if marker.is_file():
        line = marker.read_text(encoding="utf-8").splitlines()[0].strip()
        if line:
            return line
    return "python3"


def main() -> int:
    cmd = [
        WSL,
        "-d",
        DISTRO,
        "--cd",
        _to_wsl_path(ROOT),
        "-e",
        _linux_python(),
        "-m",
        "aiaun_mcp",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
