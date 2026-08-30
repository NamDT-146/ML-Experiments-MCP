from __future__ import annotations

from aiaun_mcp.config import get_settings


def test_personal_to_org_owner(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    monkeypatch.setenv("KAGGLE_USERNAME", "namdtgk14")
    monkeypatch.setenv("KAGGLE_ORG", "lab-org")
    monkeypatch.setenv("WANDB_ENTITY", "personal-entity")
    monkeypatch.setenv("WANDB_TEAM", "lab-team")
    monkeypatch.setenv("AIAUN_OWNER_MODE", "personal")
    s = get_settings(tmp_path)
    assert s.kaggle_owner == "namdtgk14"
    assert s.wandb_entity_resolved == "personal-entity"

    monkeypatch.setenv("AIAUN_OWNER_MODE", "org")
    s2 = get_settings(tmp_path)
    assert s2.kaggle_owner == "lab-org"
    assert s2.wandb_entity_resolved == "lab-team"


def test_kaggle_key_fallback(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_KEY", "legacy-key")
    monkeypatch.setenv("KAGGLE_USERNAME", "namdtgk14")
    s = get_settings(tmp_path)
    assert s.kaggle_token == "legacy-key"


def test_drive_folder_swap(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "personal-folder")
    assert get_settings(tmp_path).drive_folder_id == "personal-folder"
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "org-shared-folder")
    assert get_settings(tmp_path).drive_folder_id == "org-shared-folder"
