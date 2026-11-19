"""Frozen builds retain an explicit revision; absent or malformed receipts never fabricate one."""

import json

import pytest

from organoidtracker import paths


@pytest.mark.parametrize("dirty", [False, True])
def test_frozen_revision_records_dirty_builds(tmp_path, monkeypatch, dirty):
    monkeypatch.setattr(paths, "source_checkout_root", lambda: None)
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "paths.py"))
    revision = "a1" * 20
    (tmp_path / "_build_info.json").write_text(json.dumps({"source_revision": revision, "dirty": dirty}))
    assert paths.source_revision() == revision + ("-dirty" if dirty else "")


@pytest.mark.parametrize("contents", [None, "not JSON", "[]", '{"source_revision":"not-a-commit"}'])
def test_missing_or_invalid_build_receipt_returns_unknown(tmp_path, monkeypatch, contents):
    monkeypatch.setattr(paths, "source_checkout_root", lambda: None)
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "paths.py"))
    if contents is not None:
        (tmp_path / "_build_info.json").write_text(contents)
    assert paths.source_revision() is None


def test_source_revision_distinguishes_local_implementations(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL).decode().strip()

    git("init", "-q")
    source = tmp_path / "code.py"
    source.write_text("a = 1\n")
    git("add", "code.py")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "fixture",
    )
    revision = git("rev-parse", "HEAD")
    monkeypatch.setattr(paths, "source_checkout_root", lambda: tmp_path)
    assert paths.source_revision() == revision
    source.write_text("a = 2\n")
    dirty = paths.source_revision()
    assert dirty.startswith(revision + "-dirty-")
    source.write_text("a = 3\n")
    assert paths.source_revision() != dirty
    source.write_text("a = 1\n")
    (tmp_path / "adapter.py").write_text("implementation = 1\n")
    assert paths.source_revision().startswith(revision + "-dirty-")
