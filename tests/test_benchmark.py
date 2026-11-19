"""Benchmark scores penalize missing masks and identity swaps, and reject unrelated videos."""

import json

import numpy as np
import pytest

from helpers import FakeTracker
from organoidtracker import cli
from organoidtracker.core.masks import PackedMask
from organoidtracker.core.tracking_result import TrackingResult
from organoidtracker.services.benchmark import compare_masks
from organoidtracker.services.session import SessionError
from organoidtracker.services.tracking_service import TrackingService
from test_services import make_session, run
from test_services import small_disc as small_disc
from test_services import small_video as small_video


def mask(left=True):
    pixels = np.zeros((8, 8), dtype=bool)
    pixels[1:4, 1:4] = left
    pixels[4:7, 4:7] = not left
    return PackedMask(pixels)


def result(data):
    return TrackingResult(data, frames_total=2, frames_done=2, frame_map=[0, 1])


def test_missing_and_extra_masks_reduce_agreement():
    reference = result({0: {1: mask()}, 1: {1: mask()}})
    candidate = result({0: {1: mask(), 9: mask(False)}, 1: {}})
    comparison = compare_masks(reference, candidate)
    assert comparison["mean_mask_iou"] == pytest.approx(1 / 3)
    assert comparison["missing_masks"] == 1 and comparison["extra_masks"] == 1
    assert comparison["max_relative_area_error"] == 1
    assert comparison["first_accepted_frame_by_object"] == {"1": 0, "9": 0}


def test_reference_identity_switches_detect_an_id_swap():
    reference = result({k: {1: mask(), 2: mask(False)} for k in (0, 1)})
    candidate = result({0: {1: mask(), 2: mask(False)}, 1: {1: mask(False), 2: mask()}})
    comparison = compare_masks(reference, candidate)
    assert comparison["reference_identity_switches"] == 2
    assert comparison["mean_mask_iou"] == 0.5
    assert compare_masks(reference, reference)["reference_identity_switches"] == 0


def test_mismatched_frame_maps_are_not_comparable():
    reference = result({0: {1: mask()}})
    candidate = result({0: {1: mask()}})
    candidate.frame_map = [0, 2]
    with pytest.raises(SessionError, match="frame maps"):
        compare_masks(reference, candidate)


def test_bench_command_writes_metrics_and_rejects_a_different_video(small_disc, small_video, tmp_path, monkeypatch):
    session = make_session(small_disc, small_video)
    reference = run(session, tmp_path / "reference", small_disc)
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session.to_document()))
    monkeypatch.setattr(TrackingService, "create", classmethod(lambda cls, spec: cls(FakeTracker(small_disc))))
    arguments = [
        "bench",
        "--session",
        str(path),
        "--out",
        str(tmp_path / "candidate"),
        "--reference",
        str(reference.output_dir),
        "--no-videos",
        "--log-level",
        "ERROR",
    ]
    assert cli.main(arguments) == 0
    data = json.loads((tmp_path / "candidate/benchmark.json").read_text())
    assert data["comparison"]["mean_mask_iou"] == 1
    assert data["measurements"]["seconds_per_frame"] >= 0
    assert data["same_annotations"] and data["same_direction"]
    manifest = json.loads((tmp_path / "candidate/run_manifest.json").read_text())
    assert "benchmark.csv" in manifest["files"] and "benchmark.json" in manifest["files"]
    changed = tmp_path / "different.mp4"
    changed.write_bytes(b"not the reference video")
    assert cli.main(arguments + ["--video", str(changed)]) == 2
