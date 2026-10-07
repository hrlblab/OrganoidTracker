"""Exports carry the facts needed to reproduce and version a run."""

import re

import organoidtracker
from organoidtracker.analysis.organoid_cyst_data import ExperimentData
from organoidtracker.analysis.organoid_report_generator import OrganoidAnalysisReportGenerator
from organoidtracker.core.sam2_tracker import SAM2Tracker
from organoidtracker.paths import source_checkout_root, source_revision


def test_source_revision_is_a_commit_hash_in_a_checkout():
    revision = source_revision()
    root = source_checkout_root()
    if root is None or not (root / ".git").exists():
        assert revision is None
    else:
        assert revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision)


def test_tracker_provenance_names_software_and_mask_settings():
    tracker = SAM2Tracker(model_config="sam2_hiera_t", checkpoint_path="/nonexistent.pt", device="cpu")
    facts = tracker.provenance()
    assert facts["results_version"] == organoidtracker.RESULTS_VERSION
    assert facts["organoidtracker_version"] == organoidtracker.__version__
    assert set(facts) >= {"source_revision", "config_file", "checkpoint_sha256", "video_sha256", "direction"}
    assert set(facts["settings"]) == {
        "SAM2_USE_IMPROVED_CONFIG",
        "SAM2_IMPROVED_TRACKING",
        "SAM2_MIN_MASK_AREA",
        "SAM2_MIN_CONFIDENCE",
        "SAM2_MEMORY_FRAMES",
    }
    assert facts["collapse_duplicate_frames"] is True and facts["duplicate_threshold"] == 1.0


def test_analysis_summary_carries_results_version_and_software():
    experiment = ExperimentData(total_frames=2, time_lapse_days=1.0, conversion_factor_um_per_pixel=1.0)
    summary = OrganoidAnalysisReportGenerator()._create_analysis_summary(experiment, {}, {}, {}, None)
    assert summary["results_version"] == organoidtracker.RESULTS_VERSION
    assert summary["software"]["organoidtracker"] == organoidtracker.__version__
    assert "source_revision" in summary["software"]
