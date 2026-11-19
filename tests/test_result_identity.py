"""A reader of the exports can tell which mask is which cyst and which video a report describes.

Guarantees: the overlay frames carry ``C<id>`` labels and the acquisition day while the mask
video stays label-free; the raw table names the source frame of every row without changing
any existing column; the PDF and the summary name the video, run and model; a labelled key
image is written, listed in the manifest and reproduced by a re-export without the video.
"""

import base64
import json
import re
import zlib
from dataclasses import replace

import numpy as np

from helpers import FakeTracker, read_video
from organoidtracker.analysis.csv_import import experiment_from_csv
from organoidtracker.core.cyst_identity import cyst_color, mask_box, render_key_image
from organoidtracker.core.masks import PackedMask
from organoidtracker.core.tracking_result import TrackingResult
from organoidtracker.io.video_output import VideoOutputGenerator
from organoidtracker.services.export_service import frame_captions
from organoidtracker.services.pipeline import export_saved_result
from organoidtracker.services.saved_results import load_saved_result
from organoidtracker.services.session import Timing, VideoReference
from test_services import make_session, read_csv, run
from test_services import small_disc as small_disc
from test_services import small_video as small_video

SELECTION = (0, 2, 3, 5, 7)  # irregular on purpose: Frame and Source_Frame differ on every row but the first
DAYS = (1.0, 2.5, 3.0, 6.0, 11.0)
LEGACY_COLUMNS = [
    "Organoid_ID",
    "Cyst_ID",
    "Frame",
    "Time_Days",
    "Area_um2",
    "Circularity",
    "Centroid_X",
    "Centroid_Y",
    "Radius_um",
]


def selected_session(disc, video, **kwargs):
    session = make_session(disc, video, **kwargs)
    return replace(
        session,
        video=VideoReference(session.video.path, session.video.sha256, SELECTION, 8),
        timing=Timing(frame_times_days=DAYS),
    )


def pdf_text(path) -> str:
    """Text of a ReportLab PDF: its ASCII85 + Flate page streams, decoded."""
    data = path.read_bytes()
    chunks = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        body = match.group(1).strip()
        try:
            chunks.append(zlib.decompress(base64.a85decode(body, adobe=True)))
        except (ValueError, zlib.error):
            continue
    return b"\n".join(chunks).decode("latin-1")


def test_overlay_frames_carry_labels_and_day_but_the_mask_video_does_not():
    size = 400
    frames = [np.full((size, size, 3), 40, np.uint8) for _ in range(3)]
    mask = np.zeros((size, size), bool)
    mask[200:260, 150:230] = True
    other = np.zeros((size, size), bool)
    other[300:340, 300:360] = True
    result = TrackingResult(status=TrackingResult.PARTIAL, frames_total=3)
    for index in (1, 2):
        result[index] = {1: PackedMask(mask), 27: PackedMask(other)}
        result.tracked_frames.append(index)
    generator = VideoOutputGenerator()
    colors = {0: (128, 128, 128), 1: cyst_color(1), 27: cyst_color(27)}
    captions = frame_captions(3, [0, 2, 3])
    assert captions == [
        "Time point 1/3 | source frame 1",
        "Time point 2/3 | source frame 3",
        "Time point 3/3 | source frame 4",
    ]
    assert frame_captions(2) == ["Time point 1/2", "Time point 2/2"]
    plain = generator._process_all_frames_optimized(frames, result, colors, 0.4, 1.0, None)
    labelled = generator._process_all_frames_optimized(frames, result, colors, 0.4, 1.0, None, captions)

    def white(image, rows, columns):
        return int((image[rows, columns] > 200).all(axis=-1).sum())  # label text on the dark (40) background

    # Labels sit just above each mask's box, on every overlay, with or without captions.
    for processed in (plain, labelled):
        overlay = processed["overlay"][2]
        assert white(overlay, slice(170, 200), slice(150, 230)) >= 8  # C1
        assert white(overlay, slice(270, 300), slice(300, 370)) >= 8  # C27
        assert white(overlay, slice(40, 140), slice(0, 400)) == 0  # and nowhere else
        # The mask video holds only the palette colours: nothing a downstream reader could mistake for a mask.
        mask_frame = processed["mask"][2]
        found = {tuple(int(v) for v in color) for color in np.unique(mask_frame.reshape(-1, 3), axis=0)}
        assert found == {(0, 0, 0), cyst_color(1), cyst_color(27)}
    # The caption is written on tracked frames only when captions are given; the untracked frame keeps its mark.
    assert white(plain["overlay"][2], slice(0, 30), slice(0, 250)) == 0
    assert white(labelled["overlay"][2], slice(0, 30), slice(0, 250)) >= 40
    untracked = labelled["overlay"][0]
    assert (untracked[2, 380] == 30).all() and (untracked[60:] == 40).all()  # the NOT TRACKED strip, no labels
    assert np.array_equal(labelled["mask"][2], plain["mask"][2])
    assert labelled["side_by_side"][2].shape == (size, 2 * size, 3)
    assert np.array_equal(labelled["side_by_side"][2][:, :size], frames[2])


def test_mask_box_and_key_image_without_a_frame():
    mask = np.zeros((300, 400), bool)
    mask[150:220, 230:310] = True
    assert mask_box(mask) == (230, 150, 80, 70) and mask_box(np.zeros((5, 5), bool)) is None
    image = render_key_image(None, {4: mask}, caption="Day 3")
    assert image.shape == (300, 400, 3) and image.dtype == np.uint8
    inside = image[200, 280]
    assert tuple(int(v) for v in inside) == tuple(int(c * 0.4) for c in cyst_color(4))
    assert (image[120:150, 230:310] > 200).all(axis=-1).sum() >= 8  # "C4" above the box
    assert (image[0:30, 0:120] > 200).all(axis=-1).sum() >= 20  # the caption, without an opaque strip
    assert (image[0:30, 300:400] == 0).all()
    wide = render_key_image(np.full((100, 3200, 3), 40, np.uint8), {1: np.ones((100, 3200), bool)})
    assert wide.shape == (50, 1600, 3)


def test_raw_table_names_the_source_frame_without_changing_existing_columns(small_disc, small_video, tmp_path):
    session = selected_session(small_disc, small_video, two_objects=True, extra_empty=True)
    outcome = run(session, tmp_path / "run", small_disc, drop={(2, 2)})
    out = outcome.output_dir
    with open(out / "raw_cyst_data.csv", newline="") as handle:
        header = handle.readline().strip().split(",")
    assert header == [*LEGACY_COLUMNS, "Source_Frame"]
    rows = read_csv(out / "raw_cyst_data.csv")
    assert len(rows) == 9  # five frames of cyst 1, four of cyst 2
    for row in rows:
        assert int(row["Source_Frame"]) == SELECTION[int(row["Frame"])]
        assert float(row["Time_Days"]) == DAYS[int(row["Frame"])]
    # The existing columns hold what they held before: the fake tracker's disc of exactly that source frame.
    fake = FakeTracker(small_disc)
    fake.frame_map = list(SELECTION)
    for row in rows:
        expected = fake.mask(int(row["Frame"]), int(row["Cyst_ID"]))
        ys, xs = np.nonzero(expected)
        assert float(row["Area_um2"]) == float(expected.sum())  # calibration 1 um per pixel
        assert abs(float(row["Centroid_X"]) - xs.mean()) < 0.6 and abs(float(row["Centroid_Y"]) - ys.mean()) < 0.6
    # A reload keeps the map, and a model-free export of the saved run writes the same table.
    experiment = experiment_from_csv(out / "raw_cyst_data.csv")
    assert experiment.source_frames == list(SELECTION)
    again = export_saved_result(load_saved_result(out), tmp_path / "copy", videos=False)
    assert (again.output_dir / "raw_cyst_data.csv").read_bytes() == (out / "raw_cyst_data.csv").read_bytes()
    second = export_saved_result(load_saved_result(again.output_dir), tmp_path / "copy2", videos=False)
    assert (second.output_dir / "raw_cyst_data.csv").read_bytes() == (out / "raw_cyst_data.csv").read_bytes()


def test_report_names_its_video_run_and_model_and_carries_a_cyst_key(small_disc, small_video, tmp_path):
    session = selected_session(small_disc, small_video, two_objects=True)
    outcome = run(session, tmp_path / "run", small_disc, videos=True)
    out = outcome.output_dir
    saved = load_saved_result(out)
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["source"]["video_name"] == "disc.mp4"
    assert summary["source"]["video_sha256"] == session.video.sha256
    assert summary["source"]["run_id"] == saved.run_id and summary["source"]["decoded_frames"] == 8
    assert "fake" in summary["source"]["model"] and "sam2_hiera_t" in summary["source"]["model"]
    text = pdf_text(out / "organoid_analysis_report.pdf")
    for expected in ("disc.mp4", saved.run_id, session.video.sha256[:12], "sam2_hiera_t", "Cyst Key"):
        assert expected in text, expected
    assert "1, 3, 4, 6, 8 of 8" in text  # the selected source frames, counted from 1 as in the window
    assert "1, 2.5, 3, 6, 11" in text  # their acquisition days
    # The key image exists, is an output the manifest accounts for, and shows both cysts in their colours.
    key = out / "visualizations" / "cyst_key.png"
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert key.is_file() and "visualizations/cyst_key.png" in manifest["files"]
    assert summary["output_files"]["cyst_key"] == str(key)
    import cv2

    image = cv2.cvtColor(cv2.imread(str(key)), cv2.COLOR_BGR2RGB)
    assert image.shape[:2] == (128, 128)
    x, y = small_disc.centers[7]
    red, green = image[y, x - 20].astype(int), image[y + 12, x + 23].astype(int)  # inside cyst 1 only, cyst 2 only
    assert red[0] > red[1] + 40 and red[0] > red[2] + 40
    assert green[1] > green[0] + 40 and green[1] > green[2] + 40
    # The overlay video uses the same colours and its frames differ from an unlabelled overlay only by text.
    overlay = read_video(out / "videos" / "multi_object_overlay.mp4")
    assert len(overlay) == 5
    pixel = overlay[4][y, x - 20].astype(int)
    assert pixel[0] > pixel[1] + 40 and pixel[0] > pixel[2] + 40
    # Exporting again in place replaces the key, with the picture, although no video is exported.
    key.write_bytes(b"stale")
    export_saved_result(saved, out, videos=False, overwrite=True)
    again = cv2.cvtColor(cv2.imread(str(key)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(again, image)
    # When the source video is gone the key is still written, as the masks alone.
    gone = replace(saved, session=saved.session.with_video(tmp_path / "missing.mp4"))
    copy = export_saved_result(gone, tmp_path / "elsewhere", videos=False)
    black = cv2.cvtColor(cv2.imread(str(copy.output_dir / "visualizations" / "cyst_key.png")), cv2.COLOR_BGR2RGB)
    assert (black[5, 100] == 0).all() and black[y, x - 20][0] > 60
    assert "disc.mp4" in pdf_text(copy.output_dir / "organoid_analysis_report.pdf")


def test_report_source_keeps_the_mark_of_uncommitted_source(small_disc, small_video):
    from organoidtracker.services.export_service import report_source
    from organoidtracker.services.video_source import VideoSource

    video = VideoSource(small_video, "ab" * 32, 8, (0, 1), 128, 128, 2.0, "reverse", 1)
    provenance = {"organoidtracker_version": "2.0.0a0", "results_version": 5, "backend": "sam2-vendored"}
    clean = report_source("run1", video, {**provenance, "source_revision": "c" * 40})
    assert clean["software"] == f"organoidtracker 2.0.0a0 revision {'c' * 12} results version 5"
    dirty = report_source("run1", video, {**provenance, "source_revision": "c" * 40 + "-dirty-0123456789abcdef"})
    assert f"revision {'c' * 12}-dirty-0123456789abcdef" in dirty["software"]
    assert report_source("run1", video, {})["software"] is None and clean["video_name"] == "disc.mp4"


def test_report_without_source_or_masks_still_exports(small_disc, small_video, tmp_path):
    """Callers that pass no source (the Tk window) and runs without any accepted mask keep working."""
    from organoidtracker.services.analysis_service import AnalysisService
    from organoidtracker.services.export_service import ExportService

    session = make_session(small_disc, small_video)
    outcome = run(session, tmp_path / "run", small_disc)
    exporter = ExportService(tmp_path / "plain")
    exporter.prepare()
    analysis = AnalysisService(exporter.generator).analyze(
        outcome.result, session.annotations, session.calibration, session.timing
    )
    summary = exporter.write_report(analysis)
    assert "source" not in summary and "cyst_key" not in summary["output_files"]
    assert not (tmp_path / "plain" / "visualizations" / "cyst_key.png").exists()
    text = pdf_text(tmp_path / "plain" / "organoid_analysis_report.pdf")
    assert "Source video" not in text and "Cyst Key" not in text and "Total Organoids" in text


def test_a_failed_key_image_does_not_fail_the_report_or_leave_a_stale_key(
    small_disc, small_video, tmp_path, monkeypatch
):
    """The key is presentation: its failure is logged, the tables are written and no old key survives."""
    from organoidtracker.services import export_service

    session = make_session(small_disc, small_video)
    out = run(session, tmp_path / "run", small_disc).output_dir
    key = out / "visualizations" / "cyst_key.png"
    assert key.is_file()

    def broken(*args, **kwargs):
        raise RuntimeError("injected key failure")

    monkeypatch.setattr(export_service, "render_key_image", broken)
    outcome = export_saved_result(load_saved_result(out), out, videos=False, overwrite=True)
    assert outcome.exit_code == 0 and not key.exists()
    manifest = json.loads((out / "run_manifest.json").read_text())
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert "visualizations/cyst_key.png" not in manifest["files"] and "cyst_key" not in summary["output_files"]
    assert summary["success"] and (out / "raw_cyst_data.csv").is_file()
    assert "Cyst Key" not in pdf_text(out / "organoid_analysis_report.pdf")
