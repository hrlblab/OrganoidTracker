"""The session document: validated inputs of a run, loaded from JSON or from a prompt record."""

import json
import os
from pathlib import Path

import pytest

from organoidtracker.services.annotations import AnnotationError, AnnotationSet, CystAnnotation, OrganoidAnnotation
from organoidtracker.services.session import (
    CHECKPOINT_FAMILIES,
    MODEL_CONFIGS,
    PROMPT_RECORD_SCHEMA,
    SCHEMA,
    SessionError,
    Timing,
    load_session,
    session_from_document,
)


def document(**overrides):
    data = {
        "schema": SCHEMA,
        "video": {"path": "disc.mp4"},
        "tracking": {"model_config": "sam2_hiera_t", "device": "cpu"},
        "calibration": {"um_per_pixel": 1.6934},
        "timing": {"time_lapse_days": 6.0},
        "organoids": [
            {"organoid_id": 1, "point": [10, 10], "cysts": [{"cyst_id": 1, "bbox": [0, 0, 20, 20]}]},
            {"organoid_id": 2, "point": [30, 30], "cysts": []},
        ],
    }
    data.update(overrides)
    return data


def write(tmp_path, data, name="session.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_valid_session_loads_with_defaults_and_resolves_the_video_path(tmp_path):
    session = load_session(write(tmp_path, document()))
    assert session.video.path == tmp_path / "disc.mp4" and session.video.sha256 is None
    assert session.tracking.direction == "reverse" and session.tracking.reverse
    assert session.tracking.model_config == "sam2_hiera_t" and session.tracking.device == "cpu"
    assert session.tracking.checkpoint_family is None and session.tracking.checkpoint_path is None
    assert session.calibration.um_per_pixel == 1.6934
    assert session.timing.time_lapse_days == 6.0 and session.timing.frame_times_days is None
    assert [o.organoid_id for o in session.annotations.organoids] == [1, 2]
    assert session.annotations.cyst_ids() == [1] and session.annotations.organoids_without_cysts() == [2]
    assert session.source == tmp_path / "session.json"


def test_document_round_trip(tmp_path):
    session = load_session(write(tmp_path, document()))
    again = session_from_document(session.to_document())
    assert again.to_document() == session.to_document()
    assert again.annotations == session.annotations and again.timing == session.timing


def test_absolute_video_path_is_kept(tmp_path):
    absolute = Path(tmp_path.anchor, "data", "x.mp4")  # absolute on this platform (a drive root on Windows)
    session = session_from_document(document(video={"path": str(absolute), "sha256": "A" * 64}), base_dir=tmp_path)
    assert session.video.path == absolute and session.video.sha256 == "a" * 64


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema": "something/9"}, "schema: expected"),
        ({"extra": 1}, "unknown key"),
        ({"video": {"path": ""}}, "video.path"),
        ({"video": {"path": "x.mp4", "sha256": "abc"}}, "video.sha256"),
        ({"video": {"path": "x.mp4", "md5": "abc"}}, "unknown key"),
        ({"tracking": {"direction": "sideways"}}, "tracking.direction"),
        ({"tracking": {"model_config": "sam2_hiera_xl"}}, "tracking.model_config"),
        ({"tracking": {"checkpoint_family": "3"}}, "tracking.checkpoint_family"),
        ({"tracking": {"device": "tpu"}}, "tracking.device"),
        ({"tracking": {"gpu": "cuda"}}, "unknown key"),
        ({"calibration": {"um_per_pixel": 0}}, "calibration.um_per_pixel"),
        ({"calibration": {"um_per_px": 1.6}}, r"calibration: unknown key\(s\) \['um_per_px'\]"),
        ({"calibration": {}}, r"calibration: missing key\(s\) \['um_per_pixel'\]"),
        ({"calibration": {"um_per_pixel": "1.6"}}, "calibration.um_per_pixel"),
        ({"timing": {"time_lapse_days": -1}}, "timing.time_lapse_days"),
        ({"timing": {}}, "exactly one of"),
        ({"timing": {"time_lapse_days": 6, "frame_times_days": [1, 2]}}, "exactly one of"),
        ({"timing": {"frame_times_days": [1, 1, 2]}}, "increase strictly"),
        ({"timing": {"frame_times_days": [1, "two"]}}, r"frame_times_days\[1\]"),
        ({"timing": {"days": 6}}, "unknown key"),
        ({"organoids": "none"}, "organoids: expected a list"),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": [{"cyst_id": 1, "bbox": [5, 5, 1, 9]}]}]},
            "x1 < x2",
        ),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": [{"cyst_id": 1, "bbox": [-1, 0, 5, 5]}]}]},
            "negative",
        ),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": [{"cyst_id": 0, "bbox": [0, 0, 5, 5]}]}]},
            "positive integer",
        ),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": [{"cyst_id": 1, "bbox": [0, 0, 5]}]}]},
            "4 values",
        ),
        ({"organoids": [{"organoid_id": 1, "point": [1], "cysts": []}]}, "2 values"),
        (
            {
                "organoids": [
                    {"organoid_id": 1, "point": [1, 2], "cysts": []},
                    {"organoid_id": 1, "point": [3, 4], "cysts": []},
                ]
            },
            "used twice",
        ),
        ({"organoids": [{"organoid_id": 1, "point": [1, 2]}]}, r"organoids\[0\]: missing key\(s\) \['cysts'\]"),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cyst": [{"cyst_id": 1, "bbox": [0, 0, 5, 5]}]}]},
            r"organoids\[0\]: unknown key\(s\) \['cyst'\]",
        ),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": None}]},
            r"cysts: expected a list .* got NoneType",
        ),
        ({"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": 3}]}, r"cysts: expected a list .* got int"),
        ({"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": ["x"]}]}, r"cysts\[0\]: expected an object"),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": [{"cyst_id": 1, "box": [0, 0, 5, 5]}]}]},
            r"cysts\[0\]: unknown key\(s\) \['box'\]",
        ),
        (
            {"organoids": [{"organoid_id": 1, "point": [1, 2], "cysts": [{"cyst_id": 1}]}]},
            r"missing key\(s\) \['bbox'\]",
        ),
        (
            {
                "organoids": [
                    {"organoid_id": 1, "point": [1, 2], "cysts": [{"cyst_id": 7, "bbox": [0, 0, 5, 5]}]},
                    {"organoid_id": 2, "point": [1, 2], "cysts": [{"cyst_id": 7, "bbox": [0, 0, 5, 5]}]},
                ]
            },
            "cyst id 7 is used twice",
        ),
    ],
)
def test_invalid_documents_name_the_key(tmp_path, overrides, message):
    with pytest.raises(SessionError, match=message):
        load_session(write(tmp_path, document(**overrides)))


def test_unreadable_file_is_a_session_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(SessionError, match="cannot read session file"):
        load_session(path)
    with pytest.raises(SessionError, match="cannot read session file"):
        load_session(tmp_path / "missing.json")


def test_timing_resolution():
    uniform = Timing(time_lapse_days=12.0).resolve(7)
    assert uniform.time_lapse_days == 12.0 and uniform.frame_timestamps is None
    explicit = Timing(frame_times_days=(0, 1, 3, 7)).resolve(4)
    assert explicit.time_lapse_days == 7.0 and explicit.frame_timestamps == [0.0, 1.0, 3.0, 7.0]
    with pytest.raises(SessionError, match="4 values but the video has 5"):
        Timing(frame_times_days=(0, 1, 3, 7)).resolve(5)
    with pytest.raises(SessionError, match="no frames"):
        Timing(time_lapse_days=1.0).resolve(0)


def test_annotations_check_the_frame_size_and_keep_empty_organoids():
    annotations = AnnotationSet(
        (
            OrganoidAnnotation(1, (5, 5), (CystAnnotation(1, (0, 0, 40, 30)),)),
            OrganoidAnnotation(2, (50, 50)),
        )
    )
    annotations.check_inside(40, 30)
    with pytest.raises(AnnotationError, match="exceeds the frame size 39x30"):
        annotations.check_inside(39, 30)
    data = annotations.organoid_data()
    assert data == {
        1: {"point": (5, 5), "cysts": [{"cyst_id": 1, "bbox": (0, 0, 40, 30)}]},
        2: {"point": (50, 50), "cysts": []},
    }
    assert AnnotationSet.from_organoid_data(data) == annotations


def test_prompt_record_is_accepted_as_a_session(tmp_path):
    record = {
        "schema": PROMPT_RECORD_SCHEMA,
        "results_version": 2,
        "video": {
            "path": "well.mp4",
            "sha256": "b" * 64,
            "decoded_frames": 11,
            "unique_frames": 7,
            "frame_map": [0, 2, 3, 5, 6, 8, 9],
        },
        "tracking": {"direction": "reverse", "annotation_frame_index": 6},
        "model": {
            "model_config": "sam2_hiera_b",
            "checkpoint_family": "2.1",
            "checkpoint_path": str(Path(tmp_path.anchor, "ckpt", "x.pt")),
            "device": "cuda",
        },
        "organoids": [{"organoid_id": 1, "point": [0, 0], "cysts": [{"cyst_id": 1, "bbox": [2350, 908, 2534, 1078]}]}],
        "prompts": {"1": [{"type": "bbox", "frame_idx": 6}]},
        "analysis_inputs": {"time_lapse_days": 6.0, "conversion_factor_um_per_pixel": 1.6934},
    }
    session = load_session(write(tmp_path, record, "record.json"))
    assert session.video.path == tmp_path / "well.mp4" and session.video.sha256 == "b" * 64
    assert session.tracking.model_config == "sam2_hiera_b" and session.tracking.checkpoint_family == "2.1"
    assert (
        session.tracking.checkpoint_path == Path(tmp_path.anchor, "ckpt", "x.pt") and session.tracking.device == "cuda"
    )
    assert session.calibration.um_per_pixel == 1.6934 and session.timing.time_lapse_days == 6.0
    assert session.annotations.cyst_ids() == [1]

    record["analysis_inputs"]["time_lapse_days"] = None  # the GUI could not read the entry
    with pytest.raises(SessionError, match="exactly one of"):
        load_session(write(tmp_path, record, "record.json"))


def test_model_names_match_the_backend():
    from organoidtracker.core import sam2_tracker

    assert set(MODEL_CONFIGS) == set(sam2_tracker.SIZE_NAMES)
    assert set(CHECKPOINT_FAMILIES) == set(sam2_tracker.CHECKPOINT_FAMILIES)


def test_misspelled_cysts_key_is_rejected_not_counted_as_empty(tmp_path):
    data = document()
    data["organoids"][1]["cyst"] = [{"cyst_id": 2, "bbox": [1, 1, 10, 10]}]
    del data["organoids"][1]["cysts"]
    with pytest.raises(SessionError, match=r"organoids\[1\]: unknown key\(s\) \['cyst'\]"):
        load_session(write(tmp_path, data))


def test_relative_paths_become_absolute_when_loaded(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "weights.pt").touch()
    (inputs / "disc.mp4").touch()
    data = document(tracking={"model_config": "sam2_hiera_t", "device": "cpu", "checkpoint_path": "weights.pt"})
    write(inputs, data)
    monkeypatch.chdir(tmp_path)
    session = load_session("inputs/session.json")  # a cwd-relative session file with relative references
    assert session.video.path.is_absolute() and session.video.path == inputs / "disc.mp4"
    assert session.tracking.checkpoint_path is not None and session.tracking.checkpoint_path.is_absolute()
    assert session.tracking.checkpoint_path == inputs / "weights.pt" and session.tracking.checkpoint_path.is_file()
    assert session.with_video("other.mp4").video.path == tmp_path / "other.mp4"
    saved = session.to_document()
    assert os.path.isabs(saved["video"]["path"]) and os.path.isabs(saved["tracking"]["checkpoint_path"])
