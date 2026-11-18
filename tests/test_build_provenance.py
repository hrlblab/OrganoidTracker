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
