# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Changes that alter scientific results (masks, measurements, time axes) are listed under
a "Scientific behavior" heading so that analyses can be attributed to a version.

## [Unreleased]

### Scientific behavior
- Reverse tracking now does what the paper describes: the prompt is placed on the last
  chronological frame and SAM2 propagates backwards in time. The released code applied the
  prompt to the first frame and tracked forward, then mirrored the output indices. Results,
  videos and analyses are now keyed chronologically. All reverse-mode measurements change; on
  a sample well with identical boxes the median per-frame area difference between the two
  behaviors was 30 percent.
- Masks are binarized uniformly at probability 0.5. The analysis previously thresholded raw
  logits at 0.5 (probability 0.62) while the videos used probability 0.5.
- Consecutive near-identical frames, which video export can repeat, are collapsed into one time
  point before tracking (`COLLAPSE_DUPLICATE_FRAMES`, `DUPLICATE_FRAME_MAD_THRESHOLD`). An
  11-frame video with 4 repeated frames yields 7 time points; the log and the prompt record
  list the decoded and unique counts.
- Days are numbered from 1, as in the paper's figures. The "Time Lapse" entry is the span from
  the first to the last frame.
- Reports no longer invent cysts. The analysis step that assumed ten tracked objects and
  "reconstructed" the missing ones is removed; the annotated cysts are compared with the objects
  actually present in the tracking results and any discrepancy is reported as a warning. A mask
  that is missing for a cyst at a frame is a missing observation and is no longer replaced by
  another object's mask from another frame. With one annotated cyst the cyst summary previously
  listed seven.
- The time axis uses the number of frames that were tracked, not the number of frames that
  happened to keep an accepted mask, so a frame without masks no longer shifts the timestamps of
  the frames after it. Partial tracking runs are analyzed as partial: `analysis_summary.json`
  carries a `tracking` block (status, frame counts, error) and `complete: false`, the PDF states
  it, and the GUI log repeats it.
- Cyst growth rates are the paper's overall growth rate, the mean over consecutive observed time
  points of the area change divided by the elapsed time in days. The cyst summary CSV, the heatmap
  and the summary JSON previously multiplied the per-frame change by the days per frame instead of
  dividing, which was only correct for frames one day apart (two-day spacing reported 20 µm²/day
  for a true 5 µm²/day). The organoid-level growth rate was already per day.
- Plots and the heatmap distinguish frames the tracker never visited (a partial run) from tracked
  frames without cysts: untracked frames are gaps (gray in the heatmap), and the plotted statistics
  use tracked frames only. The tracker records the frames it visited (`tracked_frames` in the
  summary's `tracking` block).
- These corrections are results version 2 (`results_version` in the prompt record and the
  analysis summary).

### Added
- The `organoidtracker` command line: `organoidtracker run --session S --out O` tracks, analyzes and
  exports a *session file* (video, organoid points, cyst boxes, calibration, timing) without a window,
  and `organoidtracker validate` checks a session file and its video. The prompt record written when
  tracking starts is accepted as a session file, so a GUI run can be repeated headlessly. Every run
  writes `run_manifest.json` (software, environment, settings, video facts, backend provenance,
  tracking status and frame coverage, a digest of every mask, the hash of every file) next to the
  usual exports; a run that stopped early is reported as partial in every file and exits with status 3.
- `organoidtracker.services`: GUI-free tracking, analysis and export services around the existing
  backend contract, analysis engine and video writer, shared by the command line and the desktop
  interfaces. The report generator exposes its analysis and export halves separately.
- Explicit per-frame timestamps (`frame_times_days` in a session file) as an alternative to the
  uniform time axis derived from the time-lapse span.
- `pyproject.toml` and a uv lock file (`uv.lock`, Linux and Windows). The application installs as
  the `organoidtracker` package with the `organoidtracker-tk` console script; `python
  video_tracker_gui.py` still works inside the environment. Accelerator extras `cpu` and `cuda`
  select the PyTorch build; `qt` and `hf` declare the dependencies of the upcoming PySide6 interface
  and `transformers` backends; `requirements.txt` is exported from the lock for pip users.
- Typed settings (`organoidtracker.settings.Settings`) with the defaults of the original `config.py`,
  loaded from an optional `organoidtracker.toml` (`organoidtracker.example.toml` documents every key;
  keys and value types are checked). `config.py` keeps exposing the values as module constants. An
  invalid settings file (unknown key, wrong type, unreadable) stops the application from starting,
  with the error in the console and in a dialog, instead of being replaced by the defaults. Values
  from a legacy `user_config.py` go through the same type checks.
- Logging through the standard `logging` module replaces the 350 `print` calls. Entry points write
  to the console and to a rotating log file next to the outputs
  (`data/output_videos/organoidtracker.log`); warnings and errors also appear in the GUI log panel
  (`LOG_LEVEL`, `GUI_LOG_LEVEL` in `config.py`).
- Ruff (lint and format), mypy (core package) and pytest configuration in `pyproject.toml`, a
  pre-commit configuration, and a GitHub Actions workflow that runs the checks and the synthetic
  test suite on Linux and Windows (Python 3.12 and 3.13, CPU wheels) plus the SAM 2.1 tiny model
  tests on Linux.
- `CITATION.cff` and `NOTICE` (vendored SAM 2 and its cctorch kernel; dependency licenses).
- A GUI smoke test (`pytest -m gui`) that opens and closes the main window.
- This changelog. The upstream state before maintenance began is tagged `legacy-baseline`.
- Both SAM 2 and SAM 2.1 checkpoint families, selected with `SAM2_CHECKPOINT_FAMILY` in
  `config.py` (default `2.1`; `2` is the family used for the paper's figures).
- `run_tracking` returns a `TrackingResult` whose status distinguishes completed from partial
  runs; the GUI reports a run that stopped early instead of logging success.
- A prompt and provenance record (`data/output_videos/prompts/<video>_<timestamp>.json`) with
  organoid points, cyst boxes, video hash, frame map, direction, model family, checkpoint hash
  and software versions, written when tracking starts.
- SAM2's object presence score is recorded per object and frame.
- A results version (`organoidtracker.RESULTS_VERSION`, currently 1; 0 is the untouched upstream
  code) is written into the prompt record and `analysis_summary.json`, together with the software
  version and the source commit. The tracker's provenance also records the Hydra config used and the
  settings that alter masks (quality filter thresholds, memory frames, improved-config choice).
- A pytest suite with synthetic fixtures (`python -m pytest tests`); model tests use the SAM 2.1
  tiny checkpoint and skip when it is absent.

### Changed
- The Tk application writes its prompt record through the shared services module. The record now
  stores the video and checkpoint paths as absolute paths (and, for headless runs, the explicit frame
  times), so it replays from any directory.
- Source layout: `src/organoidtracker/{core,analysis,io,gui_tk}`; the vendored upstream SAM 2 is the
  top-level `sam2` package under `src/sam2` (byte-identical to facebookresearch/sam2 at 2b90b9f) with
  its configs as package data and its license files alongside. All `sys.path` edits are gone.
- The checkpoints directory and the optional `user_config.py` are resolved from the environment
  (`ORGANOIDTRACKER_CHECKPOINTS`, `ORGANOIDTRACKER_USER_CONFIG`), the working directory, then the
  source checkout, instead of from the module file location.
- The analysis modules now actually read `config.py`; their `from ...config import` had always failed
  silently and used fallback values equal to the defaults, so output is unchanged unless `user_config.py`
  overrides them. The legacy `report_generator` keeps its own publication-style values.
- Bare `except:` clauses became `except Exception:`.
- Minimum versions: Python 3.12, torch 2.7 (RTX 50-series GPUs and current CUDA wheels).
- Masks are stored as packed bits on the CPU (2 MB per 4096x4096 mask) instead of 64 MB float32
  logits per object per frame on the GPU. Peak GPU memory on a sample well with six objects fell
  from 5.3 GB to 1.3 GB.
- The device is resolved once, so the model builder receives the device actually used on
  machines without CUDA.
- The process-wide `torch.zeros`/`torch.tensor` monkeypatch is gone; the vendored predictor
  moves inputs to the right device itself.
- `decord` is no longer required; the predictor is fed the frames the application decodes.

### Removed
- `analysis/data_reconstruction.py` (`DataReconstructionEngine`), the source of the invented cysts.
- `environment.yml` (conda); the uv lock and `requirements.txt` replace it.
- `user_config_example.py`; `organoidtracker.example.toml` replaces it. An existing `user_config.py`
  is still applied, with a deprecation warning.
- The duplicated `models/sam2/configs` tree and a stray `.backup` config; the one application-owned
  config (base-plus "improved tracking") lives in `organoidtracker/configs`.
- A data-export step in the analysis report that could never run (it referenced an undefined variable
  and a script that does not exist).
- Adaptive bounding-box tracking. Its mid-run prompts were never consulted by the running
  propagation, so it changed nothing except on a repeated run.
- The Medical-SAM2 backend (`models/medical_sam2`, `src/core/inference.py`). Its upstream has
  been dormant since 2024, its checkpoint URL returns 404, and its GPL license badge conflicted
  with this repository's Apache-2.0 license. SAM2 is the only backend.

### Fixed
- Re-plotting from the exported CSV (`scripts/csv_visualizer.py`) used frame indices as timestamps,
  which changed the day labels and the growth rates, and dropped organoids without cysts, which
  inflated the population statistics; it now rebuilds the experiment through
  `organoidtracker.analysis.csv_import.experiment_from_csv`, which keeps the exported `Time_Days`,
  the tracked frames and every organoid listed in `organoid_summary.csv`.
- The Results Viewer module had lost its indentation and could not be imported, so the
  View Results action always failed; it opens again.
- The Results Viewer's frame slider and display update called each other recursively.
- `checkpoints/download_ckpts.sh` no longer aborts on a dead Medical-SAM2 URL. It accepts a
  checkpoint family argument (`2.1` by default, `2`, or `all`), downloads into its own
  directory, skips present files, and reports failures through its exit status. The README's
  manual download commands name the files that actually exist.
- Removed a shadowed duplicate `on_window_resize` definition in the main window (no behavior
  change).
