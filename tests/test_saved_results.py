"""Saved results: a run's complete experiment on disk, reloaded without a model and exported again.

The decisive check is run -> save -> reopen -> export on one hard scenario: a partial run with explicit
irregular timestamps, a non-unit calibration, two cysts with a dropped observation, an organoid without
cysts and an annotated cyst that is never tracked. Every mask comes back bit for bit, every identity,
the time axis, the calibration, the frame map, the status and the tracked frames survive, and the
exports are byte-identical to the original run's. Damaged, truncated, edited and inconsistent files
are refused with the reason; an interrupted or failed save leaves the previous result usable."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers import DiscVideo, FakeTracker, expected_growth_per_day, read_video
from organoidtracker.core.masks import PackedMask
from organoidtracker.services import saved_results
from organoidtracker.services.export_service import ExportError
from organoidtracker.services.pipeline import export_saved_result, run_session
from organoidtracker.services.run_manifest import mask_digests
from organoidtracker.services.saved_results import (
    RESULTS_NAME,
    SavedResultError,
    load_saved_result,
    write_saved_result,
)
from organoidtracker.services.session import SessionError, load_session, session_from_document
from organoidtracker.services.tracking_service import TrackingService

N = 8
TIMES = [0, 1, 3, 6, 10, 15, 21, 28]  # explicit, irregular
CALIBRATION = 1.6934
VOLATILE_SUMMARY_KEYS = ("timestamp", "output_files", "software")


@pytest.fixture(scope="module")
def small_disc():
    return DiscVideo(n_frames=N, size=128, radius=10, start=20, step=12)


@pytest.fixture(scope="module")
def small_video(small_disc, tmp_path_factory):
    return small_disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")


def hard_organoids(disc):
    cx, cy = disc.centers[N - 1]
    return [
        {
            "organoid_id": 1,
            "point": [cx - 30, cy - 30],
            "cysts": [
                {"cyst_id": 1, "bbox": list(disc.box(N - 1))},
                {"cyst_id": 2, "bbox": [cx - 15, cy - 15, cx + 20, cy + 15]},
            ],
        },
        {"organoid_id": 2, "point": [5, 5], "cysts": []},  # no cysts: still part of the population
        {"organoid_id": 3, "point": [100, 100], "cysts": [{"cyst_id": 9, "bbox": [90, 90, 120, 120]}]},  # never tracked
    ]


def make_session(disc, video, *, organoids=None, timing=None, calibration=CALIBRATION):
    return session_from_document(
        {
            "schema": "organoidtracker.session/1",
            "video": {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
            "tracking": {"model_config": "sam2_hiera_t", "device": "cpu"},
            "calibration": {"um_per_pixel": calibration},
            "timing": timing or {"frame_times_days": TIMES},
            "organoids": organoids or hard_organoids(disc),
        }
    )


def factory(disc, **fake_kwargs):
    return lambda spec: TrackingService(FakeTracker(disc, enable_reverse_tracking=spec.reverse, **fake_kwargs))


def run(session, out, disc, *, videos=False, overwrite=False, **fake_kwargs):
    return run_session(
        session, out, videos=videos, overwrite=overwrite, tracking_service_factory=factory(disc, **fake_kwargs)
    )


@pytest.fixture
def hard_run(small_disc, small_video, tmp_path):
    """Partial (frames 7, 6, 5), cyst 2 unobserved at frame 6, cyst 9 never tracked, organoid 2 empty, irregular axis."""
    session = make_session(small_disc, small_video)
    return run(
        session, tmp_path / "run", small_disc, videos=True, partial_after=3, drop={(6, 2)}, never_track={9}, grow=3
    )


def normalized_summary(path):
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if k not in VOLATILE_SUMMARY_KEYS}


def report_files(directory):
    names = ["raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"]
    names += sorted(p.relative_to(directory).as_posix() for p in (directory / "visualizations").glob("*.png"))
    return names


def assert_same_exports(original: Path, again: Path, *, videos=True):
    names = report_files(original)
    assert len([n for n in names if n.startswith("visualizations/")]) == 6
    for name in names:
        assert (again / name).read_bytes() == (original / name).read_bytes(), name
    assert normalized_summary(again / "analysis_summary.json") == normalized_summary(original / "analysis_summary.json")
    assert (again / "organoid_analysis_report.pdf").is_file()
    if videos:
        for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
            a, b = read_video(original / "videos" / name), read_video(again / "videos" / name)
            assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)), name


def read_csv(path):
    import csv

    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------------------------ round trip
def test_a_run_saves_a_result_that_reloads_bit_for_bit(hard_run):
    out = hard_run.output_dir
    manifest = json.loads(hard_run.manifest_path.read_text())
    saved = hard_run.saved_result
    assert saved is not None and saved.path == out / RESULTS_NAME and saved.masks_path.parent == out
    assert manifest["results"] == {
        "file": "results.json",
        "masks_file": saved.masks_path.name,
        "run_id": manifest["run_id"],
        "created": saved.created,
        "results_version": 2,
    }
    assert manifest["produced_by"] == "run" and manifest["source"] is None
    for name in ("results.json", saved.masks_path.name):
        assert manifest["files"][name]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()

    reloaded = load_saved_result(out)
    result = reloaded.result
    assert mask_digests(result) == manifest["masks"] == mask_digests(hard_run.result)
    assert result.status == "partial" and result.frames_done == 3 and result.frames_total == N
    assert result.tracked_frames == [7, 6, 5] and "synthetic interruption" in result.error
    assert result.direction == "reverse" and result.annotation_frame == N - 1 and result.frame_map == list(range(N))
    assert result.presence == hard_run.result.presence  # full precision, exact
    assert result.object_ids() == [1, 2] and 2 not in result[6] and 1 in result[6]
    for frame in result:
        for obj, mask in result[frame].items():
            assert isinstance(mask, PackedMask)
            assert np.array_equal(mask.numpy(), hard_run.result[frame][obj].numpy()), (frame, obj)
    # identities, time axis, calibration, video facts
    original = hard_run.saved_result.session
    assert reloaded.session.annotations.to_documents() == original.annotations.to_documents()
    assert reloaded.session.annotations.organoids_without_cysts() == [2]
    assert reloaded.session.timing == original.timing and list(reloaded.session.timing.frame_times_days) == TIMES
    assert reloaded.session.calibration.um_per_pixel == CALIBRATION
    assert reloaded.session.video.path == original.video.path and reloaded.session.video.sha256 == original.video.sha256
    assert reloaded.video == hard_run.saved_result.video
    assert reloaded.video.frame_map == tuple(range(N)) and (reloaded.video.height, reloaded.video.width) == (128, 128)
    assert reloaded.run_id == manifest["run_id"] and reloaded.results_version == 2
    assert reloaded.provenance["backend"] == "fake" and "SAM2_MIN_MASK_AREA" in reloaded.settings
    assert "python" in reloaded.environment and reloaded.software["organoidtracker"]


def test_export_reproduces_the_run_byte_for_byte_without_the_tracker(hard_run, tmp_path):
    saved = load_saved_result(hard_run.output_dir)
    again = export_saved_result(saved, tmp_path / "again", videos=True)
    assert again.status == "partial" and again.exit_code == 3 and not again.complete
    assert_same_exports(hard_run.output_dir, again.output_dir, videos=True)

    manifest = json.loads(again.manifest_path.read_text())
    original = json.loads(hard_run.manifest_path.read_text())
    assert manifest["produced_by"] == "export" and manifest["status"] == "partial" and not manifest["complete"]
    assert manifest["source"]["run_directory"] == str(hard_run.output_dir)
    assert manifest["source"]["run_created"] == saved.created and manifest["source"]["results_version"] == 2
    assert manifest["source"]["settings"] == original["settings"]
    assert manifest["run_id"] == original["run_id"]
    for key in ("masks", "tracking", "provenance", "video", "session", "analysis"):
        assert manifest[key] == original[key], key
    assert manifest["results"] == original["results"]
    # the export directory is a complete run directory: the same two result files, session and prompt record
    for name in ("results.json", original["results"]["masks_file"], "prompts.json"):
        assert (again.output_dir / name).read_bytes() == (hard_run.output_dir / name).read_bytes(), name
    assert json.loads((again.output_dir / "session.json").read_text()) == json.loads(
        (hard_run.output_dir / "session.json").read_text()
    )
    replay = load_session(again.output_dir / "session.json")
    assert replay.annotations.to_documents() == saved.session.annotations.to_documents()
    assert load_session(again.output_dir / "prompts.json").timing == saved.session.timing
    for relative, info in manifest["files"].items():
        assert hashlib.sha256((again.output_dir / relative).read_bytes()).hexdigest() == info["sha256"], relative
    assert mask_digests(load_saved_result(again.output_dir).result) == original["masks"]

    # the numbers mean what they meant: the irregular axis and the paper's growth rate on the saved masks
    rows = read_csv(again.output_dir / "raw_cyst_data.csv")
    assert sorted({float(r["Time_Days"]) for r in rows}) == [15.0, 21.0, 28.0]  # frames 5, 6, 7 of the explicit axis
    assert sorted((int(r["Cyst_ID"]), int(r["Frame"])) for r in rows) == [(1, 5), (1, 6), (1, 7), (2, 5), (2, 7)]
    areas = {int(f): objs["1"]["area"] for f, objs in original["masks"].items()}
    expected = expected_growth_per_day(areas, TIMES, CALIBRATION)
    growth = {
        int(r["Cyst_ID"]): float(r["Growth_Rate_um2_per_day"]) for r in read_csv(again.output_dir / "cyst_summary.csv")
    }
    assert growth[1] == pytest.approx(expected, abs=1e-3)
    assert float(rows[0]["Area_um2"]) == pytest.approx(areas[int(rows[0]["Frame"])] * CALIBRATION**2, abs=0.01)
    organoids = read_csv(again.output_dir / "organoid_summary.csv")
    assert [(r["Organoid_ID"], r["Total_Cysts"]) for r in organoids] == [("1", "2"), ("2", "0"), ("3", "0")]
    assert manifest["analysis"]["untracked_cysts"] == [9] and manifest["analysis"]["observed_frames"] == [5, 6, 7]
    summary = json.loads((again.output_dir / "analysis_summary.json").read_text())
    assert summary["tracking"]["status"] == "partial" and summary["experiment_info"]["frame_timestamps"] == TIMES


def test_a_tracked_frame_without_masks_stays_a_real_zero_after_reload(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video, timing={"time_lapse_days": 7.0})
    outcome = run(session, tmp_path / "run", small_disc, frames_without_masks={0}, never_track={9})
    saved = load_saved_result(outcome.output_dir)
    assert saved.result.status == "completed" and 0 in saved.result.tracked_frames and 0 not in saved.result
    again = export_saved_result(saved, tmp_path / "again", videos=False)
    assert again.exit_code == 0
    assert_same_exports(outcome.output_dir, again.output_dir, videos=False)
    manifest = json.loads(again.manifest_path.read_text())
    assert manifest["analysis"]["observed_frames"] == list(range(N))  # tracked: a zero, not a gap
    assert [int(r["Frame"]) for r in read_csv(again.output_dir / "raw_cyst_data.csv") if r["Cyst_ID"] == "1"] == list(
        range(1, N)
    )
    assert manifest["video_export"] is None and not (again.output_dir / "videos").exists()


# ------------------------------------------------------------------------------------ the video file
def test_export_without_videos_needs_no_video_and_relocation_is_checked(hard_run, tmp_path, small_video):
    saved = load_saved_result(hard_run.output_dir)
    hidden = tmp_path / "hidden.mp4"
    shutil.move(str(small_video), str(hidden))
    try:
        with pytest.raises(SessionError, match="video not found"):
            export_saved_result(saved, tmp_path / "with-videos", videos=True)
        assert not (tmp_path / "with-videos" / "run_manifest.json").exists()
        tables = export_saved_result(saved, tmp_path / "tables", videos=False)
        assert_same_exports(hard_run.output_dir, tables.output_dir, videos=False)

        other = tmp_path / "other.mp4"
        other.write_bytes(hidden.read_bytes() + b"\0")
        with pytest.raises(SessionError, match="different video"):
            export_saved_result(saved, tmp_path / "wrong", videos=True, video_path=other)

        moved = export_saved_result(saved, tmp_path / "moved", videos=True, video_path=hidden)
        assert_same_exports(hard_run.output_dir, moved.output_dir, videos=True)
        manifest = json.loads(moved.manifest_path.read_text())
        assert manifest["video"]["path"] == str(hidden) and manifest["session"]["video"]["path"] == str(hidden)
        assert json.loads((moved.output_dir / "session.json").read_text())["video"]["path"] == str(hidden)
    finally:
        shutil.move(str(hidden), str(small_video))


# ------------------------------------------------------------------------------------ damaged files
def _edit(directory: Path, change):
    path = directory / RESULTS_NAME
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _masks_file(directory: Path) -> Path:
    return directory / json.loads((directory / RESULTS_NAME).read_text())["masks"]["file"]


def _first_entry(data):
    frame = sorted(data["masks"]["index"], key=int)[0]
    obj = sorted(data["masks"]["index"][frame], key=int)[0]
    return frame, obj


DAMAGE = {
    "truncated mask file": (
        lambda d: _masks_file(d).write_bytes(_masks_file(d).read_bytes()[:-50]),
        "damaged or truncated",
    ),
    "corrupted mask file": (
        lambda d: _masks_file(d).write_bytes(
            bytes(
                b ^ 0xFF if i == len(_masks_file(d).read_bytes()) // 2 else b
                for i, b in enumerate(_masks_file(d).read_bytes())
            )
        ),
        "does not match its recorded sha256",
    ),
    "missing mask file": (lambda d: _masks_file(d).unlink(), "is missing"),
    "unreadable json": (lambda d: (d / RESULTS_NAME).write_text("{not json"), "cannot read"),
    "other schema": (lambda d: _edit(d, lambda data: data.update(schema="organoidtracker.results/9")), "schema"),
    "unknown key": (lambda d: _edit(d, lambda data: data.update(masks_file="x")), "unknown key"),
    "missing key": (lambda d: _edit(d, lambda data: data.pop("tracking")), "missing key"),
    "mask dropped from the index": (
        lambda d: _edit(d, lambda data: data["masks"]["index"][_first_entry(data)[0]].pop(_first_entry(data)[1])),
        "the index does not list",
    ),
    "edited area": (
        lambda d: _edit(
            d, lambda data: data["masks"]["index"][_first_entry(data)[0]][_first_entry(data)[1]].update(area=1)
        ),
        "has area",
    ),
    "edited mask hash": (
        lambda d: _edit(
            d, lambda data: data["masks"]["index"][_first_entry(data)[0]][_first_entry(data)[1]].update(sha256="0" * 64)
        ),
        "does not match its recorded hash",
    ),
    "partial run relabelled as completed": (
        lambda d: _edit(d, lambda data: data["tracking"].update(status="completed")),
        "a completed run must have tracked every frame",
    ),
    "tracked frames edited": (
        lambda d: _edit(d, lambda data: data["tracking"].update(tracked_frames=[7, 6])),
        "frames_done is 3 but 2 frames",
    ),
    "frame map shortened": (
        lambda d: _edit(d, lambda data: data["tracking"].update(frame_map=[0, 1, 2])),
        "frame count or frame map differs",
    ),
    "masks on an untracked frame": (
        lambda d: _edit(d, lambda data: data["masks"]["index"].update({"4": data["masks"]["index"].pop("5")})),
        "not a tracked frame",
    ),
    "object ids edited": (
        lambda d: _edit(d, lambda data: data["tracking"].update(object_ids=[1])),
        "differs from the masks' objects",
    ),
    "misspelled session key": (
        lambda d: _edit(
            d, lambda data: data["session"]["organoids"][0].update(cyst=data["session"]["organoids"][0].pop("cysts"))
        ),
        "session: .*unknown key",
    ),
    "timestamps no longer match the frames": (
        lambda d: _edit(d, lambda data: data["session"]["timing"].update(frame_times_days=[1, 2, 3])),
        "session: .*3 values but the video has 8",
    ),
    "video hash edited": (
        lambda d: _edit(d, lambda data: data["video"].update(sha256="f" * 64)),
        "different video hashes",
    ),
}


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_damaged_or_edited_results_are_refused(hard_run, tmp_path, damage):
    copy = tmp_path / "copy"
    shutil.copytree(hard_run.output_dir, copy)
    mutate, match = DAMAGE[damage]
    mutate(copy)
    with pytest.raises(SavedResultError, match=match):
        load_saved_result(copy)


def test_a_damaged_result_exports_nothing(hard_run, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(hard_run.output_dir, copy)
    DAMAGE["truncated mask file"][0](copy)
    with pytest.raises(SavedResultError):
        export_saved_result(load_saved_result(copy), tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------ saving
def _save(directory, outcome, result=None):
    saved = outcome.saved_result
    return write_saved_result(
        directory,
        run_id=saved.run_id,
        session=saved.session,
        video=saved.video,
        provenance=saved.provenance,
        result=result if result is not None else saved.result,
        settings=saved.settings,
        environment=saved.environment,
    )


def test_saving_the_same_result_again_is_idempotent_and_a_changed_result_replaces_it(hard_run, tmp_path):
    store = tmp_path / "store"
    first = _save(store, hard_run)
    again = _save(store, hard_run)
    assert again.masks_path == first.masks_path and again.masks_path.is_file()  # the bytes depend on the masks alone
    assert sorted(p.name for p in store.glob("masks-*.npz")) == [first.masks_path.name]
    changed = hard_run.result.__class__(
        {f: {o: m for o, m in objs.items() if (f, o) != (5, 1)} for f, objs in hard_run.result.items()},
        status=hard_run.result.status,
        frames_total=hard_run.result.frames_total,
        frames_done=hard_run.result.frames_done,
        error=hard_run.result.error,
        direction=hard_run.result.direction,
        annotation_frame=hard_run.result.annotation_frame,
        frame_map=hard_run.result.frame_map,
        presence=hard_run.result.presence,
        tracked_frames=hard_run.result.tracked_frames,
    )
    third = _save(store, hard_run, result=changed)
    assert third.masks_path != first.masks_path and not first.masks_path.exists()  # the stale file is gone
    assert mask_digests(load_saved_result(store).result) == mask_digests(changed)
    assert not list(store.glob(".masks-*"))


def test_an_interrupted_or_failed_save_keeps_the_previous_result(hard_run, tmp_path, monkeypatch):
    store = tmp_path / "store"
    previous = _save(store, hard_run)
    before = {p.name: p.read_bytes() for p in store.iterdir()}
    changed = hard_run.result.__class__(
        {f: {o: m for o, m in objs.items() if (f, o) != (5, 1)} for f, objs in hard_run.result.items()},
        status="partial",
        frames_total=N,
        frames_done=3,
        error="x",
        direction="reverse",
        annotation_frame=N - 1,
        frame_map=list(range(N)),
        presence=hard_run.result.presence,
        tracked_frames=[7, 6, 5],
    )

    def crash_while_writing_the_masks(handle, arrays):
        handle.write(b"partial")
        raise OSError("no space")

    monkeypatch.setattr(saved_results, "_write_npz", crash_while_writing_the_masks)
    with pytest.raises(SavedResultError, match="no space"):
        _save(store, hard_run, result=changed)
    monkeypatch.undo()
    assert {p.name: p.read_bytes() for p in store.iterdir()} == before  # nothing published, no temporary left
    assert mask_digests(load_saved_result(store).result) == mask_digests(previous.result)

    def crash_while_writing_the_document(path, text):
        raise OSError("disk full")  # the mask file is staged, results.json is not touched

    monkeypatch.setattr(saved_results, "_write_text", crash_while_writing_the_document)
    with pytest.raises(SavedResultError, match="disk full"):
        _save(store, hard_run, result=changed)
    monkeypatch.undo()
    assert {p.name: p.read_bytes() for p in store.iterdir()} == before
    assert mask_digests(load_saved_result(store).result) == mask_digests(previous.result)

    # a crash after the mask file was renamed but before results.json: the old pair still loads, and the
    # orphan is removed because the previous results.json does not name it
    import os

    original_replace = os.replace

    def crash_on_the_json_rename(src, dst):
        if str(dst).endswith(RESULTS_NAME):
            raise OSError("power cut")
        return original_replace(src, dst)

    monkeypatch.setattr(saved_results.os, "replace", crash_on_the_json_rename)
    with pytest.raises(SavedResultError, match="power cut"):
        _save(store, hard_run, result=changed)
    monkeypatch.undo()
    assert {p.name: p.read_bytes() for p in store.iterdir()} == before
    assert mask_digests(load_saved_result(store).result) == mask_digests(previous.result)
    recovered = _save(store, hard_run, result=changed)
    assert sorted(p.name for p in store.glob("masks-*.npz")) == [recovered.masks_path.name]
    assert mask_digests(load_saved_result(store).result) == mask_digests(changed)


# ------------------------------------------------------------------------------------ directories
def test_export_in_place_keeps_the_saved_result_and_replaces_the_exports(hard_run):
    out = hard_run.output_dir
    saved = load_saved_result(out)
    kept = {
        name: (out / name).read_bytes()
        for name in ("results.json", saved.masks_path.name, "session.json", "prompts.json")
    }
    (out / "organoidtracker.log").write_text("log line\n")
    with pytest.raises(ExportError, match="already holds a run"):
        export_saved_result(saved, out, videos=False)
    assert (out / "run_manifest.json").is_file()  # refused before anything changed

    again = export_saved_result(saved, out, videos=False, overwrite=True)
    assert again.output_dir == out and again.exit_code == 3
    for name, content in kept.items():
        assert (out / name).read_bytes() == content, name
    assert (out / "organoidtracker.log").read_text() == "log line\n"
    assert not (out / "videos").exists()  # the previous exports went, these were made without videos
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["produced_by"] == "export" and manifest["video_export"] is None
    assert not any(path.startswith("videos/") for path in manifest["files"])
    for path in out.rglob("*"):
        if path.is_file() and not path.name.startswith("organoidtracker.log"):
            relative = path.relative_to(out).as_posix()
            assert relative == "run_manifest.json" or relative in manifest["files"], relative
    assert mask_digests(load_saved_result(out).result) == manifest["masks"]


def test_export_into_a_directory_holding_another_run_replaces_it_completely(
    hard_run, small_disc, small_video, tmp_path
):
    other = run(
        make_session(small_disc, small_video, timing={"time_lapse_days": 7.0}, calibration=1.0),
        tmp_path / "other",
        small_disc,
        videos=True,
        never_track={9},
    )
    other_masks = other.saved_result.masks_path
    saved = load_saved_result(hard_run.output_dir)
    with pytest.raises(ExportError, match="already holds a run"):
        export_saved_result(saved, other.output_dir, videos=False)
    again = export_saved_result(saved, other.output_dir, videos=False, overwrite=True)
    manifest = json.loads(again.manifest_path.read_text())
    assert manifest["run_id"] == saved.run_id and manifest["produced_by"] == "export"
    assert not other_masks.exists() and not (other.output_dir / "videos").exists()
    assert mask_digests(load_saved_result(other.output_dir).result) == mask_digests(saved.result)
    assert json.loads((other.output_dir / "session.json").read_text())["calibration"]["um_per_pixel"] == CALIBRATION
    for path in other.output_dir.rglob("*"):
        if path.is_file() and not path.name.startswith("organoidtracker.log"):
            relative = path.relative_to(other.output_dir).as_posix()
            assert relative == "run_manifest.json" or relative in manifest["files"], relative


def test_a_failed_export_leaves_the_copied_result_but_no_manifest(hard_run, tmp_path, monkeypatch):
    from organoidtracker.analysis.organoid_report_generator import OrganoidAnalysisReportGenerator

    monkeypatch.setattr(OrganoidAnalysisReportGenerator, "_generate_enhanced_pdf_report", lambda self, *a, **k: None)
    saved = load_saved_result(hard_run.output_dir)
    with pytest.raises(ExportError, match="organoid_analysis_report.pdf"):
        export_saved_result(saved, tmp_path / "out", videos=False)
    out = tmp_path / "out"
    assert not (out / "run_manifest.json").exists()
    assert json.loads((out / "analysis_summary.json").read_text())["success"] is False
    assert mask_digests(load_saved_result(out).result) == mask_digests(saved.result)  # still re-exportable


def test_a_run_that_fails_after_tracking_keeps_its_saved_result(small_disc, small_video, tmp_path, monkeypatch):
    from organoidtracker.analysis.organoid_report_generator import OrganoidAnalysisReportGenerator

    monkeypatch.setattr(OrganoidAnalysisReportGenerator, "_generate_enhanced_pdf_report", lambda self, *a, **k: None)
    session = make_session(small_disc, small_video, timing={"time_lapse_days": 7.0})
    with pytest.raises(ExportError):
        run(session, tmp_path / "run", small_disc, never_track={9})
    saved = load_saved_result(tmp_path / "run")  # the tracking is not lost: export it once the problem is fixed
    assert saved.result.status == "completed" and saved.result.object_ids() == [1, 2]
    monkeypatch.undo()
    again = export_saved_result(saved, tmp_path / "again", videos=False)
    assert again.exit_code == 0 and (again.output_dir / "organoid_analysis_report.pdf").is_file()


# ------------------------------------------------------------------------------------ review findings
def test_a_failed_replacement_keeps_the_previous_saved_run_and_its_files(
    hard_run, small_disc, small_video, tmp_path, monkeypatch
):
    """Save Results over a saved run: the previous run, its session, prompt record and manifest stay until the
    replacement is complete (review finding: the GUI cleared them first, a failed write left nothing)."""
    from organoidtracker.services.export_service import ExportService
    from organoidtracker.services.prompt_record import build_prompt_record

    out = hard_run.output_dir
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    other = run(
        make_session(small_disc, small_video, timing={"time_lapse_days": 7.0}, calibration=1.0),
        tmp_path / "other",
        small_disc,
        never_track={9},
    )
    replacement = other.saved_result
    record = build_prompt_record(
        FakeTracker(small_disc), str(small_video), replacement.session.annotations.organoid_data(), 7.0, 1.0
    )
    exporter = ExportService(out)

    def save(**kwargs):
        return exporter.save_run(
            run_id="replacement",
            session=replacement.session,
            video=replacement.video,
            provenance=replacement.provenance,
            result=replacement.result,
            prompt_record=record,
            **kwargs,
        )

    with pytest.raises(ExportError, match="already holds a saved run"):
        save()
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before

    for seam, message in (("_write_npz", "disk full"), ("_write_text", "no space")):

        def fail(*args, _message=message, **kwargs):
            raise OSError(_message)

        monkeypatch.setattr(saved_results, seam, fail)
        with pytest.raises(SavedResultError, match=message):
            save(replace_existing=True)
        monkeypatch.undo()
        assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before, seam  # nothing changed
        assert mask_digests(load_saved_result(out).result) == mask_digests(hard_run.result)

    saved = save(replace_existing=True)
    assert saved.run_id == "replacement" and not (out / "run_manifest.json").exists()
    assert mask_digests(load_saved_result(out).result) == mask_digests(replacement.result)
    assert json.loads((out / "session.json").read_text())["calibration"]["um_per_pixel"] == 1.0
    assert json.loads((out / "prompts.json").read_text())["analysis_inputs"]["time_lapse_days"] == 7.0
    assert sorted(p.name for p in out.glob("masks-*.npz")) == [saved.masks_path.name]
    assert not list(out.glob(".*.tmp"))
    assert (out / "raw_cyst_data.csv").read_bytes() == before["raw_cyst_data.csv"]  # exports untouched


def test_a_relocated_export_reopens_and_exports_again_without_the_override(hard_run, small_video, tmp_path):
    """Export with a relocated video, then export that export without naming the video again (review finding:
    the copied results.json kept the missing path). The provenance keeps the original path."""
    saved = load_saved_result(hard_run.output_dir)
    original = saved.session.video.path
    moved = tmp_path / "elsewhere" / "well.mp4"
    moved.parent.mkdir()
    shutil.move(str(small_video), str(moved))
    try:
        first = export_saved_result(saved, tmp_path / "first", videos=True, video_path=moved)
        reopened = load_saved_result(first.output_dir)
        assert reopened.session.video.path == moved and reopened.video.path == moved
        assert reopened.provenance["video_path"] == str(original)  # where the masks came from
        assert mask_digests(reopened.result) == mask_digests(saved.result)
        assert load_session(first.output_dir / "prompts.json").video.path == moved
        assert json.loads((first.output_dir / "session.json").read_text())["video"]["path"] == str(moved)

        second = export_saved_result(reopened, tmp_path / "second", videos=True)  # no override needed
        assert second.exit_code == 3
        assert_same_exports(hard_run.output_dir, second.output_dir, videos=True)
        assert load_saved_result(second.output_dir).session.video.path == moved

        # in place: the run directory itself learns the new location
        in_place = export_saved_result(saved, hard_run.output_dir, videos=True, overwrite=True, video_path=moved)
        assert in_place.output_dir == hard_run.output_dir
        again = load_saved_result(hard_run.output_dir)
        assert again.session.video.path == moved and again.provenance["video_path"] == str(original)
        assert load_session(hard_run.output_dir / "session.json").video.path == moved
        assert load_session(hard_run.output_dir / "prompts.json").video.path == moved
        third = export_saved_result(again, tmp_path / "third", videos=True)
        assert_same_exports(hard_run.output_dir, third.output_dir, videos=True)
    finally:
        shutil.move(str(moved), str(small_video))
