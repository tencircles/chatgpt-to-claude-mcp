"""
Unit tests for save_profile backup logic, _safe_resolve, and phrase persistence.
"""

import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

from chatgpt_to_claude_mcp.server import _safe_resolve, save_profile, PROFILE_PATH
from chatgpt_to_claude_mcp.extract import (
    load_banned_phrases, save_banned_phrases,
    load_signal_phrases, save_signal_phrases,
)


# ---------------------------------------------------------------------------
# _safe_resolve
# ---------------------------------------------------------------------------

def test_safe_resolve_normal(tmp_path):
    assert _safe_resolve(str(tmp_path)) == tmp_path.resolve()

def test_safe_resolve_forbidden_etc():
    assert _safe_resolve("/etc/passwd") is None

def test_safe_resolve_forbidden_ssh():
    assert _safe_resolve("~/.ssh/id_rsa") is None

def test_safe_resolve_forbidden_aws():
    assert _safe_resolve("~/.aws/credentials") is None

def test_safe_resolve_tilde_expansion(tmp_path):
    result = _safe_resolve(str(tmp_path))
    assert result is not None
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# save_profile — backup behaviour
# ---------------------------------------------------------------------------

OLD_PROFILE = "# Old Profile\n\nThis is the old content.\n"
NEW_PROFILE  = "# New Profile\n\nThis is the new content.\n"


@pytest.fixture()
def isolated_profile(tmp_path, monkeypatch):
    """Redirect PROFILE_PATH to a temp dir so we never touch ~/.config."""
    fake_global = tmp_path / "config" / "profile.md"
    fake_global.parent.mkdir()
    monkeypatch.setattr("chatgpt_to_claude_mcp.server.PROFILE_PATH", fake_global)
    return fake_global


def test_save_profile_no_existing(isolated_profile, tmp_path):
    """First save — no backup created, profile written."""
    result = save_profile(NEW_PROFILE, str(tmp_path))
    assert "error" not in result
    assert isolated_profile.read_text() == NEW_PROFILE
    assert result["global_backup"] is None
    assert result["local_backup"] is None


def test_save_profile_backup_preserves_old_content(isolated_profile, tmp_path):
    """Second save — backup has OLD content, profile has NEW content."""
    # Write initial profile (simulates a previous run)
    isolated_profile.write_text(OLD_PROFILE)

    result = save_profile(NEW_PROFILE, str(tmp_path))

    assert "error" not in result
    # New content is in profile
    assert isolated_profile.read_text() == NEW_PROFILE
    # Old content is in backup
    backup = Path(result["global_backup"])
    assert backup.exists()
    assert backup.read_text() == OLD_PROFILE


def test_save_profile_local_backup_preserves_old_content(isolated_profile, tmp_path):
    """Local extracted/profile.md backup also has OLD content."""
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    local_profile = extracted / "profile.md"
    local_profile.write_text(OLD_PROFILE)

    result = save_profile(NEW_PROFILE, str(tmp_path))

    assert local_profile.read_text() == NEW_PROFILE
    local_backup = Path(result["local_backup"])
    assert local_backup.read_text() == OLD_PROFILE


def test_save_profile_does_not_overwrite_backup_with_new_content(isolated_profile, tmp_path):
    """Regression: backup must NOT contain the new profile content."""
    isolated_profile.write_text(OLD_PROFILE)
    result = save_profile(NEW_PROFILE, str(tmp_path))

    backup = Path(result["global_backup"])
    assert backup.read_text() != NEW_PROFILE


# ---------------------------------------------------------------------------
# Phrase persistence
# ---------------------------------------------------------------------------

def test_banned_phrases_roundtrip(tmp_path):
    phrases = {"spam", "foo bar", "UPPER"}
    save_banned_phrases(tmp_path, phrases)
    loaded = load_banned_phrases(tmp_path)
    assert loaded == {"spam", "foo bar", "upper"}  # normalised to lowercase


def test_banned_phrases_empty_default(tmp_path):
    assert load_banned_phrases(tmp_path) == set()


def test_signal_phrases_roundtrip(tmp_path):
    phrases = ["my game", "my studio", "the client"]
    save_signal_phrases(tmp_path, phrases)
    loaded = load_signal_phrases(tmp_path)
    assert loaded == phrases


def test_signal_phrases_empty_default(tmp_path):
    assert load_signal_phrases(tmp_path) == []
