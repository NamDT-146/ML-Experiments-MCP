from __future__ import annotations

from aiaun_mcp.drive_login import is_headless


def test_is_headless_ssh(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")
    monkeypatch.delenv("AIAUN_DRIVE_OAUTH_MODE", raising=False)
    assert is_headless() is True


def test_is_headless_linux_no_display(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("AIAUN_DRIVE_OAUTH_MODE", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("aiaun_mcp.drive_login.sys.platform", "linux")
    assert is_headless() is True


def test_headless_mode_override(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("AIAUN_DRIVE_OAUTH_MODE", "tunnel")
    assert is_headless() is True
