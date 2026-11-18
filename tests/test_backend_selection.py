"""Backend choices survive sessions, prompt replay and service creation without loading weights."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from organoidtracker.services.session import SessionError, TrackingSpec, session_from_document
from organoidtracker.services.tracking_service import TrackingService


def test_default_backend_does_not_change_legacy_session_documents():
    spec = TrackingSpec()
    assert spec.backend == "sam2" and "backend" not in spec.to_document()


@pytest.mark.parametrize("backend", ["hf-sam", "", None, True, 1, []])
def test_invalid_backend_is_refused(backend):
    with pytest.raises(SessionError, match="tracking.backend"):
        TrackingSpec(backend=backend)


def test_service_uses_the_selected_backend():
    seen = []
    backend = object()
    registry = SimpleNamespace(create_model_instance=lambda name, **kwargs: seen.append((name, kwargs)) or backend)
    spec = TrackingSpec(backend="hf-sam2", checkpoint_path=Path("snapshot"), device="cpu")
    service = TrackingService.create(spec, registry)
    assert service.tracker is backend
    assert seen[0][0] == "hf-sam2" and seen[0][1]["checkpoint_path"] == "snapshot"


def test_prompt_replay_retains_hf_backend_and_snapshot(tmp_path):
    record = {
        "schema": "organoidtracker.prompts/1",
        "video": {"path": "video.mp4"},
        "tracking": {"direction": "reverse"},
        "model": {
            "backend": "sam2-transformers",
            "model_config": "sam2_hiera_t",
            "checkpoint_family": "2.1",
            "checkpoint_path": "snapshot",
            "device": "cpu",
        },
        "organoids": [{"organoid_id": 1, "point": [1, 1], "cysts": []}],
        "analysis_inputs": {"time_lapse_days": 6, "conversion_factor_um_per_pixel": 1.5},
    }
    session = session_from_document(record, base_dir=tmp_path)
    assert session.tracking.backend == "hf-sam2"
    assert session.tracking.checkpoint_path == tmp_path / "snapshot"
    assert session_from_document(session.to_document()).tracking == session.tracking
