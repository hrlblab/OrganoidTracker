"""The organoidtracker command line: validation, exit codes, invalid settings, and an end-to-end run."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from helpers import DiscVideo, FakeTracker
from organoidtracker import cli
from organoidtracker.services.tracking_service import TrackingService

N = 8


@pytest.fixture(scope="module")
def small_disc():
    return DiscVideo(n_frames=N, size=128, radius=10, start=20, step=12)


@pytest.fixture(scope="module")
def small_video(small_disc, tmp_path_factory):
    return small_disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")


def session_file(tmp_path, disc, video_path, **overrides):
    cx, cy = disc.centers[N - 1]
    document = {
        "schema": "organoidtracker.session/1",
        "video": {
            "path": video_path.name,
            "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "frame_indices": list(range(N)),
            "decoded_frames": N,
        },
        "tracking": {"model_config": "sam2_hiera_t", "device": "cpu"},
        "calibration": {"um_per_pixel": 1.0},
        "timing": {"time_lapse_days": 7.0},
        "organoids": [
            {"organoid_id": 1, "point": [cx - 30, cy - 30], "cysts": [{"cyst_id": 1, "bbox": list(disc.box(N - 1))}]}
        ],
    }
    document.update(overrides)
    path = video_path.parent / f"session-{tmp_path.name}.json"
    path.write_text(json.dumps(document))
    return path


def use_fake_backend(monkeypatch, disc, **fake_kwargs):
    monkeypatch.setattr(
        TrackingService,
        "create",
        classmethod(
            lambda cls, spec, registry=None: cls(FakeTracker(disc, enable_reverse_tracking=spec.reverse, **fake_kwargs))
        ),
    )


def test_validate_reports_the_session_and_its_video(tmp_path, small_disc, small_video, capsys):
    path = session_file(tmp_path, small_disc, small_video)
    assert cli.main(["validate", "--session", str(path)]) == 0
    out = capsys.readouterr().out
    assert "1 organoids, 1 cysts" in out and "valid" in out and "video sha256:" in out


def test_validate_rejects_bad_sessions_and_missing_videos(tmp_path, small_disc, small_video, capsys):
    path = session_file(tmp_path, small_disc, small_video, timing={"time_lapse_days": -2})
    assert cli.main(["validate", "--session", str(path)]) == 2
    assert "timing.time_lapse_days" in capsys.readouterr().err
    path = session_file(tmp_path, small_disc, small_video, video={"path": "nowhere.mp4"})
    assert cli.main(["validate", "--session", str(path)]) == 2
    assert "video not found" in capsys.readouterr().err
    assert cli.main(["validate", "--session", str(tmp_path / "absent.json")]) == 2


def test_run_completes_with_exit_0_and_writes_the_run(tmp_path, small_disc, small_video, monkeypatch):
    use_fake_backend(monkeypatch, small_disc)
    path = session_file(tmp_path, small_disc, small_video)
    out = tmp_path / "run"
    assert cli.main(["run", "--session", str(path), "--out", str(out), "--no-videos", "--log-level", "WARNING"]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["video_export"] is None
    assert (out / "organoidtracker.log").is_file() and (out / "raw_cyst_data.csv").is_file()
    assert not (out / "videos").exists()
    # the same directory is refused without --overwrite, accepted with it
    assert cli.main(["run", "--session", str(path), "--out", str(out), "--no-videos", "--log-level", "ERROR"]) == 2
    assert (
        cli.main(
            ["run", "--session", str(path), "--out", str(out), "--no-videos", "--overwrite", "--log-level", "ERROR"]
        )
        == 0
    )


def test_partial_run_exits_3(tmp_path, small_disc, small_video, monkeypatch):
    use_fake_backend(monkeypatch, small_disc, partial_after=2)
    path = session_file(tmp_path, small_disc, small_video)
    out = tmp_path / "run"
    assert cli.main(["run", "--session", str(path), "--out", str(out), "--no-videos", "--log-level", "ERROR"]) == 3
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "partial" and manifest["tracking"]["frames_done"] == 2


def test_run_rejects_invalid_inputs_with_exit_2(tmp_path, small_disc, small_video, monkeypatch):
    use_fake_backend(monkeypatch, small_disc)
    bad = session_file(tmp_path, small_disc, small_video, calibration={"um_per_pixel": -1})
    assert (
        cli.main(["run", "--session", str(bad), "--out", str(tmp_path / "a"), "--no-videos", "--log-level", "ERROR"])
        == 2
    )
    good = session_file(tmp_path, small_disc, small_video)
    other = tmp_path / "other.mp4"
    other.write_bytes(small_video.read_bytes() + b"\0")
    assert (
        cli.main(
            [
                "run",
                "--session",
                str(good),
                "--out",
                str(tmp_path / "b"),
                "--video",
                str(other),
                "--no-videos",
                "--log-level",
                "ERROR",
            ]
        )
        == 2
    )
    assert (
        cli.main(
            [
                "run",
                "--session",
                str(good),
                "--out",
                str(tmp_path / "c"),
                "--video",
                str(tmp_path / "none.mp4"),
                "--no-videos",
                "--log-level",
                "ERROR",
            ]
        )
        == 2
    )


def test_backend_failure_exits_1(tmp_path, small_disc, small_video, monkeypatch):
    use_fake_backend(monkeypatch, small_disc, fail_load=True)
    path = session_file(tmp_path, small_disc, small_video)
    assert (
        cli.main(["run", "--session", str(path), "--out", str(tmp_path / "run"), "--no-videos", "--log-level", "ERROR"])
        == 1
    )


def test_report_export_failure_exits_1(tmp_path, small_disc, small_video, monkeypatch):
    use_fake_backend(monkeypatch, small_disc)
    path = session_file(tmp_path, small_disc, small_video)
    out = tmp_path / "run"
    (out / "organoid_analysis_report.pdf").mkdir(parents=True)  # the PDF cannot be written here
    assert cli.main(["run", "--session", str(path), "--out", str(out), "--no-videos", "--log-level", "ERROR"]) == 1
    assert not (out / "run_manifest.json").exists()
    assert json.loads((out / "analysis_summary.json").read_text())["success"] is False


def test_overrides_reach_the_session(tmp_path, small_disc, small_video, monkeypatch):
    seen = {}

    def create(cls, spec, registry=None):
        seen["spec"] = spec
        return cls(FakeTracker(small_disc, enable_reverse_tracking=spec.reverse))

    monkeypatch.setattr(TrackingService, "create", classmethod(create))
    path = session_file(tmp_path, small_disc, small_video)
    assert (
        cli.main(
            [
                "run",
                "--session",
                str(path),
                "--out",
                str(tmp_path / "run"),
                "--no-videos",
                "--device",
                "cuda:1",
                "--model",
                "sam2_hiera_l",
                "--backend",
                "hf-sam2",
                "--checkpoint",
                str(tmp_path / "snapshot"),
                "--log-level",
                "ERROR",
            ]
        )
        == 0
    )
    assert seen["spec"].device == "cuda:1" and seen["spec"].model_config == "sam2_hiera_l"
    assert seen["spec"].backend == "hf-sam2" and seen["spec"].checkpoint_path == tmp_path / "snapshot"


def test_invalid_settings_stop_the_command_line(tmp_path, small_disc, small_video):
    bad = tmp_path / "organoidtracker.toml"
    bad.write_text("sam2_min_mask_area = 1.5\n")
    path = session_file(tmp_path, small_disc, small_video)
    env = {**os.environ, "ORGANOIDTRACKER_SETTINGS": str(bad)}
    proc = subprocess.run(
        [sys.executable, "-m", "organoidtracker.cli", "run", "--session", str(path), "--out", str(tmp_path / "run")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "Invalid settings" in proc.stderr and "expected an integer" in proc.stderr
    assert not (tmp_path / "run").exists()


def test_version_and_help():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


@pytest.mark.model
def test_end_to_end_run_with_the_real_model(tmp_path, disc, disc_video_path, device, tiny_checkpoint):
    """The CLI on the synthetic disc video with SAM 2.1 tiny: every frame's mask is the disc."""
    import math

    cx, cy = disc.centers[disc.n_frames - 1]
    document = {
        "schema": "organoidtracker.session/1",
        "video": {
            "path": str(disc_video_path),
            "frame_indices": list(range(disc.n_frames)),
            "decoded_frames": disc.n_frames,
        },
        "tracking": {"model_config": "sam2_hiera_t", "checkpoint_path": str(tiny_checkpoint), "device": device},
        "calibration": {"um_per_pixel": 1.0},
        "timing": {"time_lapse_days": 7.0},
        "organoids": [
            {
                "organoid_id": 1,
                "point": [cx - 60, cy - 60],
                "cysts": [{"cyst_id": 1, "bbox": list(disc.box(disc.n_frames - 1))}],
            }
        ],
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(document))
    out = tmp_path / "run"
    assert cli.main(["run", "--session", str(path), "--out", str(out), "--log-level", "WARNING"]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and sorted(manifest["tracking"]["tracked_frames"]) == list(
        range(disc.n_frames)
    )
    assert manifest["tracking"]["tracked_frames"][0] == disc.n_frames - 1  # visited in reverse order
    assert manifest["provenance"]["checkpoint_sha256"] and manifest["provenance"]["backend"] == "sam2-vendored"
    expected_area = math.pi * disc.radius**2
    for frame in range(disc.n_frames):
        area = manifest["masks"][str(frame)]["1"]["area"]
        assert abs(area - expected_area) / expected_area < 0.15, f"frame {frame}: area {area}"
    assert (out / "videos/multi_object_overlay.mp4").is_file() and (out / "organoid_analysis_report.pdf").is_file()
    with open(out / "raw_cyst_data.csv", newline="") as handle:
        assert len(handle.read().splitlines()) == disc.n_frames + 1


def _run_dir(tmp_path, small_disc, small_video, monkeypatch, name="run", **fake_kwargs):
    use_fake_backend(monkeypatch, small_disc, **fake_kwargs)
    path = session_file(tmp_path, small_disc, small_video)
    out = tmp_path / name
    code = cli.main(["run", "--session", str(path), "--out", str(out), "--log-level", "ERROR"])
    return out, code


NO_MODEL_SCRIPT = """
import sys
from importlib.abc import MetaPathFinder


class NoModel(MetaPathFinder):  # any import of torch or the vendored SAM 2 raises ImportError
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "sam2"):
            raise ImportError(f"{name} cannot be imported in this process")
        return None


sys.meta_path.insert(0, NoModel())
from organoidtracker.cli import main

code = main(sys.argv[1:])
loaded = sorted(
    name
    for name in sys.modules
    if name.split(".")[0] in ("torch", "sam2")
    or name.startswith(("organoidtracker.core.sam2_tracker", "organoidtracker.core.model_registry"))
)
print("MODEL_MODULES", loaded)
sys.exit(code)
"""


def test_export_reproduces_the_run_in_a_process_without_torch_or_sam2(tmp_path, small_disc, small_video, monkeypatch):
    """run -> exit the process -> export elsewhere with model loading impossible: the same exports, bit for bit."""
    run_dir, code = _run_dir(tmp_path, small_disc, small_video, monkeypatch, partial_after=5)
    assert code == 3
    out = tmp_path / "export"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            NO_MODEL_SCRIPT,
            "export",
            "--run",
            str(run_dir),
            "--out",
            str(out),
            "--log-level",
            "WARNING",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3, proc.stderr  # the saved run was partial, so is its export
    assert "MODEL_MODULES []" in proc.stdout, proc.stdout
    original = json.loads((run_dir / "run_manifest.json").read_text())
    exported = json.loads((out / "run_manifest.json").read_text())
    assert exported["produced_by"] == "export" and exported["masks"] == original["masks"]
    assert exported["tracking"] == original["tracking"] and exported["status"] == "partial"
    assert exported["environment"]["torch"] is None  # recorded truthfully: no torch in that process
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (out / name).read_bytes() == (run_dir / name).read_bytes(), name
    for figure in (run_dir / "visualizations").glob("*.png"):
        assert (out / "visualizations" / figure.name).read_bytes() == figure.read_bytes(), figure.name
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (out / "videos" / name).stat().st_size == (run_dir / "videos" / name).stat().st_size, name
    assert (out / "results.json").read_bytes() == (run_dir / "results.json").read_bytes()


def test_export_exit_codes(tmp_path, small_disc, small_video, monkeypatch):
    run_dir, code = _run_dir(tmp_path, small_disc, small_video, monkeypatch)
    assert code == 0
    # in place needs --overwrite; then the exports are replaced and the result survives
    assert cli.main(["export", "--run", str(run_dir), "--log-level", "ERROR"]) == 2
    assert cli.main(["export", "--run", str(run_dir), "--overwrite", "--no-videos", "--log-level", "ERROR"]) == 0
    assert (run_dir / "results.json").is_file() and not (run_dir / "videos").exists()
    # another directory: fresh is fine, one holding a run is refused unless overwritten
    out = tmp_path / "out"
    assert cli.main(["export", "--run", str(run_dir), "--out", str(out), "--no-videos", "--log-level", "ERROR"]) == 0
    assert cli.main(["export", "--run", str(run_dir), "--out", str(out), "--no-videos", "--log-level", "ERROR"]) == 2
    assert (
        cli.main(
            [
                "export",
                "--run",
                str(run_dir / "results.json"),
                "--out",
                str(out),
                "--no-videos",
                "--overwrite",
                "--log-level",
                "ERROR",
            ]
        )
        == 0
    )
    # the video is needed only for the videos; a different file is refused
    hidden = tmp_path / "hidden.mp4"
    os.replace(small_video, hidden)
    try:
        assert cli.main(["export", "--run", str(run_dir), "--out", str(tmp_path / "v"), "--log-level", "ERROR"]) == 2
        assert not (tmp_path / "v" / "run_manifest.json").exists()
        assert (
            cli.main(
                ["export", "--run", str(run_dir), "--out", str(tmp_path / "t"), "--no-videos", "--log-level", "ERROR"]
            )
            == 0
        )
        other = tmp_path / "other.mp4"
        other.write_bytes(hidden.read_bytes() + b"\0")
        assert (
            cli.main(
                [
                    "export",
                    "--run",
                    str(run_dir),
                    "--out",
                    str(tmp_path / "w"),
                    "--video",
                    str(other),
                    "--log-level",
                    "ERROR",
                ]
            )
            == 2
        )
        assert (
            cli.main(
                [
                    "export",
                    "--run",
                    str(run_dir),
                    "--out",
                    str(tmp_path / "r"),
                    "--video",
                    str(hidden),
                    "--log-level",
                    "ERROR",
                ]
            )
            == 0
        )
        assert (tmp_path / "r" / "videos" / "multi_object_overlay.mp4").is_file()
    finally:
        os.replace(hidden, small_video)
    # a directory without a saved result, and a damaged one
    assert (
        cli.main(["export", "--run", str(tmp_path / "nowhere"), "--out", str(tmp_path / "x"), "--log-level", "ERROR"])
        == 2
    )
    masks = next(run_dir.glob("masks-*.npz"))
    masks.write_bytes(masks.read_bytes()[:-40])
    assert (
        cli.main(["export", "--run", str(run_dir), "--out", str(tmp_path / "d"), "--no-videos", "--log-level", "ERROR"])
        == 2
    )
    assert not (tmp_path / "d" / "run_manifest.json").exists()


def test_export_failure_exits_1(tmp_path, small_disc, small_video, monkeypatch):
    run_dir, _ = _run_dir(tmp_path, small_disc, small_video, monkeypatch)
    out = tmp_path / "out"
    (out / "organoid_analysis_report.pdf").mkdir(parents=True)
    assert cli.main(["export", "--run", str(run_dir), "--out", str(out), "--no-videos", "--log-level", "ERROR"]) == 1
    assert not (out / "run_manifest.json").exists() and (out / "results.json").is_file()


def test_export_with_a_relocated_video_can_be_exported_again(tmp_path, small_disc, small_video, monkeypatch):
    """A second reopen cycle: export with --video, then export that directory without --video (review finding)."""
    run_dir, code = _run_dir(tmp_path, small_disc, small_video, monkeypatch)
    assert code == 0
    moved = tmp_path / "moved.mp4"
    os.replace(small_video, moved)
    try:
        first = tmp_path / "first"
        assert (
            cli.main(
                ["export", "--run", str(run_dir), "--out", str(first), "--video", str(moved), "--log-level", "ERROR"]
            )
            == 0
        )
        results = json.loads((first / "results.json").read_text())
        assert results["session"]["video"]["path"] == str(moved) and results["video"]["path"] == str(moved)
        assert results["provenance"]["video_path"] == str(small_video)
        second = tmp_path / "second"
        assert cli.main(["export", "--run", str(first), "--out", str(second), "--log-level", "ERROR"]) == 0
        for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
            assert (second / name).read_bytes() == (run_dir / name).read_bytes(), name
        assert (second / "videos" / "multi_object_overlay.mp4").stat().st_size == (
            run_dir / "videos" / "multi_object_overlay.mp4"
        ).stat().st_size
        # the run directory itself, relocated in place, exports again as well
        assert (
            cli.main(["export", "--run", str(run_dir), "--video", str(moved), "--overwrite", "--log-level", "ERROR"])
            == 0
        )
        assert (
            cli.main(["export", "--run", str(run_dir), "--out", str(tmp_path / "third"), "--log-level", "ERROR"]) == 0
        )
    finally:
        os.replace(moved, small_video)


def test_cli_explicit_selection_override_and_invalid_indices(monkeypatch, small_disc, small_video, tmp_path):
    use_fake_backend(monkeypatch, small_disc)
    session = session_file(tmp_path, small_disc, small_video)
    document = json.loads(session.read_text())
    document["video"].pop("frame_indices")
    document["video"].pop("decoded_frames")
    session.write_text(json.dumps(document))
    target = tmp_path / "selected"
    assert (
        cli.main(
            [
                "run",
                "--session",
                str(session),
                "--out",
                str(target),
                "--no-videos",
                "--frames",
                "1,3,5,8",
                "--decoded-frames",
                "8",
            ]
        )
        == 0
    )
    from organoidtracker.services.saved_results import load_saved_result

    assert load_saved_result(target).video.frame_map == (0, 2, 4, 7)
    assert (
        cli.main(
            [
                "run",
                "--session",
                str(session),
                "--out",
                str(tmp_path / "invalid"),
                "--frames",
                "one,two",
                "--decoded-frames",
                "8",
            ]
        )
        == 2
    )
    assert not (tmp_path / "invalid" / "results.json").exists()


def test_frame_options_cannot_move_a_confirmed_session_to_other_frames(
    monkeypatch, small_disc, small_video, tmp_path, caplog
):
    """A confirmed session's boxes and days belong to its frames: another selection is refused, not tracked."""
    use_fake_backend(monkeypatch, small_disc)
    indices, days = [0, 2, 4, 7], [1, 2, 4, 7]
    session = session_file(
        tmp_path,
        small_disc,
        small_video,
        video={
            "path": small_video.name,
            "sha256": hashlib.sha256(small_video.read_bytes()).hexdigest(),
            "frame_indices": indices,
            "decoded_frames": N,
        },
        timing={"frame_times_days": days},
    )

    def attempt(name, frames, decoded, command="run"):
        extra = ["--reference", str(tmp_path / "same")] if command == "bench" else []
        arguments = [command, "--session", str(session), "--out", str(tmp_path / name), "--no-videos", *extra]
        return cli.main([*arguments, "--frames", frames, "--decoded-frames", decoded])

    # The same number of frames, so the recorded days would fit: only the last source frame differs.
    assert attempt("moved", "1,3,5,7", "8") == 2
    assert "differ from the selection this session confirmed (frames 1, 3, 5, 8 of 8)" in caplog.text
    assert attempt("recount", "1,3,5,8", "9") == 2
    for name in ("moved", "recount"):
        assert not any((tmp_path / name).glob("*.json")), name  # nothing was tracked or saved
    # Repeating the confirmed selection is not a change.
    assert attempt("same", "1,3,5,8", "8") == 0
    from organoidtracker.services.saved_results import load_saved_result

    saved = load_saved_result(tmp_path / "same")
    assert saved.video.frame_map == tuple(indices) and saved.session.timing.frame_times_days == tuple(days)
    assert attempt("bench", "1,3,5,7", "8", command="bench") == 2


def test_validate_rejects_days_that_do_not_match_the_selected_frames(tmp_path, small_disc, small_video, capsys):
    video = {
        "path": small_video.name,
        "sha256": hashlib.sha256(small_video.read_bytes()).hexdigest(),
        "frame_indices": [0, 7],
        "decoded_frames": N,
    }
    path = session_file(tmp_path, small_disc, small_video, video=video, timing={"frame_times_days": [1, 4, 8]})
    assert cli.main(["validate", "--session", str(path)]) == 2
    captured = capsys.readouterr()
    assert "3 values but the video has 2 selected frames" in captured.err and "valid\n" not in captured.out
    # Whatever validate accepts, run accepts: the same session with one day per selected frame.
    path = session_file(tmp_path, small_disc, small_video, video=video, timing={"frame_times_days": [1, 8]})
    assert cli.main(["validate", "--session", str(path)]) == 0
    assert capsys.readouterr().out.endswith("valid\n")


def _selected_run(tmp_path, small_disc, small_video, monkeypatch, name="first"):
    """A run on source frames 1, 5 and 8 of 8, annotated on frame 8; returns its directory."""
    use_fake_backend(monkeypatch, small_disc)
    video = {
        "path": small_video.name,
        "sha256": hashlib.sha256(small_video.read_bytes()).hexdigest(),
        "frame_indices": [0, 4, 7],
        "decoded_frames": N,
    }
    session = session_file(tmp_path, small_disc, small_video, video=video)
    out = tmp_path / name
    assert cli.main(["run", "--session", str(session), "--out", str(out), "--no-videos"]) == 0
    return out


def _days_by_source_frame(directory):
    import csv

    with (directory / "raw_cyst_data.csv").open(newline="") as handle:
        return {int(row["Source_Frame"]): float(row["Time_Days"]) for row in csv.DictReader(handle)}


def test_frames_an_older_version_selected_itself_are_not_a_confirmation(
    tmp_path, small_disc, small_video, monkeypatch, caplog
):
    """A prompt record of results version 4 names frames the software chose by image similarity.

    Read as confirmed, it started a new run on them with their inferred days, and no option could
    correct it. It is now an older document: the acquisition's frames are given before tracking.
    """
    from organoidtracker.services.saved_results import load_saved_result
    from organoidtracker.services.session import load_session

    first = _selected_run(tmp_path, small_disc, small_video, monkeypatch)
    current = json.loads((first / "prompts.json").read_text())
    assert current["video"]["frame_map"] == [0, 4, 7] and current["analysis_inputs"]["time_lapse_days"] == 7.0
    legacy = tmp_path / "legacy-prompts.json"
    legacy.write_text(json.dumps({**current, "results_version": 4}))

    assert load_session(first / "prompts.json").video.frame_indices == (0, 4, 7)  # this version's record
    assert load_session(legacy).video.frame_indices is None
    assert "--frames 1,5,8 --decoded-frames 8 repeats that selection" in caplog.text
    without_version = tmp_path / "unversioned-prompts.json"
    without_version.write_text(json.dumps({k: v for k, v in current.items() if k != "results_version"}))
    assert load_session(without_version).video.frame_indices is None

    def run(name, session, *frames):
        options = ["--frames", frames[0], "--decoded-frames", "8"] if frames else []
        return cli.main(["run", "--session", str(session), "--out", str(tmp_path / name), "--no-videos", *options])

    assert run("unconfirmed", legacy) == 2 and not (tmp_path / "unconfirmed" / "results.json").exists()
    # The acquisition's eight daily frames: frame 5 is day 5, which the inferred selection called day 4.5.
    assert run("all", legacy, "1,2,3,4,5,6,7,8") == 0
    assert _days_by_source_frame(tmp_path / "all") == {index: index + 1.0 for index in range(8)}
    # Giving the recorded frames repeats the old run on purpose.
    assert run("repeat", legacy, "1,5,8") == 0
    assert _days_by_source_frame(tmp_path / "repeat") == {0: 1.0, 4: 4.5, 7: 8.0}
    # What that run wrote is confirmed: it replays without options and refuses other frames.
    replayed = tmp_path / "all" / "prompts.json"
    assert run("again", replayed) == 0
    assert load_saved_result(tmp_path / "again").video.frame_map == tuple(range(8))
    assert run("other", replayed, "1,5,8") == 2 and not (tmp_path / "other" / "results.json").exists()


def test_the_selection_is_checked_against_the_video_before_the_model_is_loaded(
    tmp_path, small_disc, small_video, monkeypatch, capsys
):
    """A session declaring nine frames for an eight-frame video passed validate; run failed after loading."""

    def no_model(cls, spec, registry=None):
        raise AssertionError("the model must not be loaded for a selection the video cannot hold")

    monkeypatch.setattr(TrackingService, "create", classmethod(no_model))
    video = {
        "path": small_video.name,
        "sha256": hashlib.sha256(small_video.read_bytes()).hexdigest(),
        "frame_indices": [0, 8],
        "decoded_frames": 9,
    }
    session = session_file(tmp_path, small_disc, small_video, video=video, timing={"frame_times_days": [1, 9]})
    assert cli.main(["validate", "--session", str(session)]) == 2
    captured = capsys.readouterr()
    assert "expects 9 decoded frames, found 8" in captured.err and "valid\n" not in captured.out
    out = tmp_path / "nine"
    assert cli.main(["run", "--session", str(session), "--out", str(out), "--no-videos"]) == 2
    assert not (out / "session.json").exists() and not (out / "results.json").exists()
