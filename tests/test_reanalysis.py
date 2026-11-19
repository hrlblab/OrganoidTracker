"""Changed analysis inputs must survive every public saved-run consumer."""

import numpy as np
import pytest

from organoidtracker.core.video_frames import unique_frame_indices
from organoidtracker.services.pipeline import export_saved_result
from organoidtracker.services.saved_results import load_saved_result
from organoidtracker.services.session import Calibration, SessionError, Timing, load_session
from test_services import make_session, read_csv, run
from test_services import small_disc as small_disc
from test_services import small_video as small_video


def test_duplicate_selection_compares_decoded_predecessors():
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in [0, 1, 2, 5, 5]]
    assert unique_frame_indices(frames, 2) == [0, 3]
    assert unique_frame_indices([], 1) == []


def test_reanalysis_copy_preserves_scientific_inputs_and_original(small_disc, small_video, tmp_path):
    original = run(make_session(small_disc, small_video), tmp_path / "original", small_disc).saved_result
    original_bytes = original.path.read_bytes()
    calibration = Calibration(2.00123456789123)
    timing = Timing(frame_times_days=(1, 2, 3, 5, 7, 9, 10, 12))
    outcome = export_saved_result(original, tmp_path / "new", videos=False, calibration=calibration, timing=timing)
    saved = load_saved_result(outcome.output_dir)
    assert original.path.read_bytes() == original_bytes
    assert saved.results_version == original.results_version and saved.provenance == original.provenance
    assert saved.session.calibration == calibration and saved.session.timing == timing
    for name in ("session.json", "prompts.json"):
        replay = load_session(outcome.output_dir / name)
        assert replay.calibration == calibration and replay.timing == timing
    rows = read_csv(outcome.output_dir / "raw_cyst_data.csv")
    old_rows = read_csv(original.directory / "raw_cyst_data.csv")
    for row, old in zip(rows, old_rows, strict=True):
        assert float(row["Area_um2"]) == pytest.approx(float(old["Area_um2"]) * calibration.um_per_pixel**2)
    again = export_saved_result(saved, tmp_path / "again", videos=False)
    assert (again.output_dir / "raw_cyst_data.csv").read_bytes() == (
        outcome.output_dir / "raw_cyst_data.csv"
    ).read_bytes()
    with pytest.raises(SessionError, match="new export directory"):
        export_saved_result(original, original.directory, videos=False, calibration=calibration, overwrite=True)
    assert original.path.read_bytes() == original_bytes
    with pytest.raises(SessionError, match="selected frames"):
        export_saved_result(original, tmp_path / "bad", videos=False, timing=Timing(frame_times_days=(1, 2)))
    assert not (tmp_path / "bad/results.json").exists()
