"""Reproducible backend comparisons against saved masks, not biological ground truth.

Run one benchmark per process for meaningful process-peak RSS. The normal pipeline still
writes a complete saved run and its reports. Benchmark files add timing, memory and mask
agreement; the inference memory sample is taken before report/video generation.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.tracking_result import TrackingResult
from .export_service import ExportError
from .pipeline import RunOutcome, check_video
from .run_manifest import file_inventory
from .saved_results import load_saved_result
from .session import Session, SessionError
from .tracking_service import TrackingService


def compare_masks(reference: TrackingResult, candidate: TrackingResult) -> dict[str, Any]:
    """Compare the union of frame/object pairs; missing or extra masks score zero IoU."""
    from scipy.optimize import linear_sum_assignment

    if reference.frames_total != candidate.frames_total or reference.frame_map != candidate.frame_map:
        raise SessionError("benchmark reference and candidate have different chronological frame maps")
    pairs: list[dict[str, Any]] = []
    assignments: dict[int, int] = {}
    switches = 0
    for frame in sorted(set(reference) | set(candidate)):
        ref = {int(k): v.numpy().astype(bool) for k, v in reference.get(frame, {}).items()}
        cand = {int(k): v.numpy().astype(bool) for k, v in candidate.get(frame, {}).items()}
        for obj in sorted(set(ref) | set(cand)):
            a, b = ref.get(obj), cand.get(obj)
            if a is not None and b is not None and a.shape != b.shape:
                raise SessionError("benchmark masks have different source-pixel dimensions")
            area_a = int(a.sum()) if a is not None else 0
            area_b = int(b.sum()) if b is not None else 0
            intersection = int((a & b).sum()) if a is not None and b is not None else 0
            union = area_a + area_b - intersection
            pairs.append(
                {
                    "frame": frame,
                    "object_id": obj,
                    "iou": intersection / union if union else 1.0,
                    "reference_area_pixels": area_a,
                    "candidate_area_pixels": area_b,
                    "relative_area_error": abs(area_b - area_a) / area_a if area_a else None,
                    "missing": a is not None and b is None,
                    "extra": a is None and b is not None,
                }
            )
        ref_ids, cand_ids = sorted(ref), sorted(cand)
        overlaps = np.zeros((len(ref_ids), len(cand_ids)))
        for i, rid in enumerate(ref_ids):
            for j, cid in enumerate(cand_ids):
                if ref[rid].shape != cand[cid].shape:
                    raise SessionError("benchmark masks have different source-pixel dimensions")
                union = int((ref[rid] | cand[cid]).sum())
                overlaps[i, j] = int((ref[rid] & cand[cid]).sum()) / union if union else 0
        rows, columns = linear_sum_assignment(-overlaps)
        for i, j in zip(rows, columns, strict=True):
            if overlaps[i, j] < 0.5:
                continue
            rid, cid = ref_ids[i], cand_ids[j]
            if rid in assignments and assignments[rid] != cid:
                switches += 1
            assignments[rid] = cid
    errors = [pair["relative_area_error"] for pair in pairs if pair["relative_area_error"] is not None]
    return {
        "mean_mask_iou": float(np.mean([p["iou"] for p in pairs])) if pairs else None,
        "max_relative_area_error": max(errors) if errors else None,
        "missing_masks": sum(p["missing"] for p in pairs),
        "extra_masks": sum(p["extra"] for p in pairs),
        "reference_identity_switches": switches,
        "first_accepted_frame_by_object": {
            str(obj): min(frame for frame, masks in candidate.items() if obj in masks) for obj in candidate.object_ids()
        },
        "pairs": pairs,
    }


class Benchmark:
    def __init__(self, session: Session, reference: Path | None) -> None:
        self.reference = load_saved_result(reference) if reference is not None else None
        if self.reference is not None:
            if check_video(session) != self.reference.video.sha256:
                raise SessionError("benchmark reference is from a different video")
        self.measurements: dict[str, Any] = {}

    def create_service(self, spec) -> TrackingService:
        owner = self

        class MeasuredService(TrackingService):
            def load_model(self):
                import torch

                device = getattr(self.tracker, "device", "cpu")
                self.benchmark_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
                self.benchmark_device = device
                if self.benchmark_cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                super().load_model()

            def run(self, progress=None, should_stop=None):
                import torch

                if self.benchmark_cuda:
                    torch.cuda.synchronize(self.benchmark_device)
                start = time.perf_counter()
                result = super().run(progress, should_stop)
                if self.benchmark_cuda:
                    torch.cuda.synchronize(self.benchmark_device)
                elapsed = time.perf_counter() - start
                peak_rss = None
                try:
                    import resource

                    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                        1 if sys.platform == "darwin" else 1024
                    )
                except ImportError:  # Windows: unavailable rather than a fabricated zero
                    pass
                owner.measurements.update(
                    tracking_seconds=elapsed,
                    seconds_per_frame=elapsed / result.frames_done if result.frames_done else None,
                    peak_process_rss_bytes=peak_rss,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.benchmark_device)
                    if self.benchmark_cuda
                    else None,
                )
                return result

        base = TrackingService.create(spec)
        return MeasuredService(base.tracker, base.run_id)

    def write(self, outcome: RunOutcome) -> dict[str, Any]:
        try:
            return self._write(outcome)
        except OSError as error:
            raise ExportError(f"Cannot write benchmark artifacts: {error}") from error

    def _write(self, outcome: RunOutcome) -> dict[str, Any]:
        saved = outcome.saved_result
        assert saved is not None
        metrics = compare_masks(self.reference.result, outcome.result) if self.reference else {}
        record = {
            "schema": "organoidtracker.benchmark/1",
            "status": outcome.status,
            "backend": saved.provenance.get("backend"),
            "device": saved.provenance.get("device"),
            "video_sha256": saved.video.sha256,
            "provenance": saved.provenance,
            "reference_run_id": self.reference.run_id if self.reference else None,
            "reference_provenance": self.reference.provenance if self.reference else None,
            "same_annotations": saved.session.annotations == self.reference.session.annotations
            if self.reference
            else None,
            "same_direction": saved.result.direction == self.reference.result.direction if self.reference else None,
            "measurements": dict(self.measurements),
            "pipeline_timings_s": outcome.timings_s,
            "comparison": metrics,
            "interpretation": "Agreement with saved reference masks, not accuracy against expert annotation. "
            "Identity switches count changes of one-to-one reference assignments with IoU >= 0.5. "
            "RSS is this process's lifetime peak through inference; use one case per process.",
        }
        directory = outcome.output_dir
        (directory / "benchmark.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        row = {key: record[key] for key in ("backend", "device", "video_sha256", "status")}
        row.update(self.measurements)
        row.update(
            {key: value for key, value in metrics.items() if key not in ("pairs", "first_accepted_frame_by_object")}
        )
        with (directory / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        # Include the two added artifacts in the normal run's inventory.
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = file_inventory(directory, [*manifest["files"], "benchmark.json", "benchmark.csv"])
        outcome.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return record
