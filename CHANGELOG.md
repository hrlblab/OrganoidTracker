# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Changes that alter scientific results (masks, measurements, time axes) are listed under
a "Scientific behavior" heading so that analyses can be attributed to a version.

## [Unreleased]

## [2.0.0a2] - 2026-11-19

No measurement changes: results version stays 5. The application is the one tagged `v2.0.0a1`; that tag
was never published as a release, because its Docker launcher failed on part of the supported hosts.

### Fixed
- Start the Docker bundle on hosts whose Docker uses the classic image store instead of the containerd
  image store. The bundle named its image by one digest; such a host
  knows the loaded image by another one, so the launcher loaded the archive and then stopped with "The
  supplied image does not match the bundle digest". The bundle now names both digests of the same image
  and the launcher accepts either.

### Added
- A `Release bundle` workflow that builds both images from a release tag and attaches the release files
  to that tag's GitHub release.

### Changed
- The README is shortened to what a new user needs: what the tool does, how to start, what a run
  contains, which version matches the paper, and the known limitations. The interface, command-line,
  measurement and development details moved unchanged to `docs/usage.md`, `docs/measurements.md` and
  `docs/development.md`.
- The README pins the code as submitted with the paper (`d748b21`), the original code's final state
  (`legacy-baseline`), the first corrected version (`494e053`) and the current prerelease.

## [2.0.0a1] - 2026-11-19

First prerelease of the maintained code base, delivered as the Linux Docker desktop preview
(`packaging/docker/README.md`). It covers everything changed since the code published with the paper
(tag `legacy-baseline`). Exports carry results version 5. The limitations that affect how results are
read are listed under "Known limitations of this prerelease" in the README. The newest changes come first.

### Run inputs stay the ones shown and recorded

No measurement changes for the same inputs: results version stays 5. The corrections stop two ways of
giving a run other inputs than the ones shown or recorded.

#### Qt interface

- Keep the calibration that is in its field when frames and days are confirmed. Confirmation wrote the
  previous value back, so a run started afterwards measured areas with the old calibration (2 µm/pixel
  entered, 1 µm/pixel used: every area a quarter of its value).
- Export reports from saved masks when the source video has moved. The window named the recorded path
  as a relocation, which made a report-only export read the missing video. Videos still need the video.
- Keep the calibration and days entered for a re-analysis when the moved video of a saved result is located.
- Export reports from saved masks when a located video has gone offline again (an unplugged disk).

#### Command line

- Refuse `--frames`/`--decoded-frames` that differ from the selection a session already confirmed
  (exit 2, no tracking result written). They moved the session's boxes and acquisition days to other source frames
  and the run still completed. Completing an older session without a selection is unchanged.
- `validate` rejects a session whose `frame_times_days` count differs from its selected frames, as `run` does.
- Read the source frames in a prompt record written before results version 5 as unconfirmed, like an
  older session file. That software selected them itself by image similarity; a new run from such a record
  used them and their inferred days without asking (source frame 5 of eight daily frames as day 4.5). The
  command line now needs `--frames`, and the window asks for confirmation. Saved results reopen and
  re-export with their recorded frames as before.
- Check a confirmed selection against the frame count in the video's index before the model is loaded.
  `validate` reported a session declaring nine frames for an eight-frame video as valid, and `run`
  failed only after loading the model (now exit 2 from both).

#### Reports and exports

- List a series of more than 30 time points in a table of its own in the PDF, 30 time points per row.
  One cell held every source frame, and past a page's height the PDF and so the whole export failed.
  Reports of up to 30 time points are unchanged.
- List the PNG written beside every SVG figure in `run_manifest.json`.

### Cyst identity, tracking direction and acquisition days

No measurement changes: results version stays 5. `raw_cyst_data.csv` gains one trailing column, so it
is no longer byte-identical to earlier exports; every earlier column is unchanged.

#### Qt interface

- Lock the tracking direction while annotations exist, refuse a session whose direction differs from
  the one its boxes were drawn under, clear the undo history when the direction changes, and stop
  remembering the direction between sessions. Boxes drawn on the last time point could previously be
  applied to the first and produce a completed run with wrong areas.
- Give a newly opened video no default span. Confirmation needs days that were entered or chosen with
  the preset; a table shows each source frame with its day before confirming.
- Treat days edited after confirmation as unconfirmed: tracking and saving wait for a new confirmation,
  which keeps annotations when the frames are unchanged. The frame label always shows the recorded day.
- Explain a click that cannot annotate (no video, unconfirmed time points, wrong frame, saved result)
  and highlight the confirm button while confirmation is needed. Errors use a distinct banner style
  and name the input that is wrong.
- Label every mask in the result view with its cyst id in one colour table shared with the videos,
  show that colour in the annotation list, and outline the mask of the selected cyst.

#### Reports and exports

- Name the source video, its hash prefix, the run, the model, the tracking software and the selected
  source frames with their days on the PDF's summary and in `analysis_summary.json`.
- Add `visualizations/cyst_key.png`, a labelled overlay embedded in the PDF. A re-export without videos
  still draws it on the picture when the source video is readable, and on black otherwise.
- Label masks with `C<id>` and write the time point and its source frame on overlay and side-by-side
  video frames, identically for every caller of the export service. The mask video is unchanged.
- Add `Source_Frame` (zero-based source-video frame) to `raw_cyst_data.csv`; CSV reloads keep it.
- Recipient guide: add the confirmation step and describe labels and the report's first page.


### Acquisition confirmation and report interpretation

#### Scientific behavior (results version 5)

- Require confirmed source frame indices and decoded count for new runs. The Qt and Tk interfaces
  preview and confirm source frames and acquisition days; the seven-day preset is explicit opt-in.
- Preserve quiet observations: image-difference thresholds no longer choose experiment time points.
- Persist selections in sessions and prompt records; preserve existing saved-result mappings.
- Invalidate Tk live results after cyst-prompt edits and reject invalid analysis parameters.
- Show missing trajectory observations as gaps and retain single observations in figures.

#### Reports and delivery

- Align heatmap rows with growth bars; include units and honor text/title/DPI settings.
- Describe accepted-mask counts and missing-mask limitations in figures and PDF warnings.
- Document each growth estimator in the exported CSV/JSON without changing its formula.
- Preserve requested original video resolution and label untracked output frames.
- Embed raster companions for SVG reports; require summary metadata for CSV replotting.
- Clear stale Tk video state after failed loads, reject drags from outside the image, and confirm
  replacement of existing Tk report/video outputs.
- Document the Docker desktop's trusted single-user host boundary.

### Missing-data semantics and saved-result safety

#### Scientific behavior (results version 4)

- Keep consecutive Qt cyst annotations attached to the selected organoid across edits and undo/redo.
- Reject incomplete indexed videos before assigning experiment times; retain OpenCV pixel decoding.
- Represent growth from fewer than two observations as missing, exclude it from growth averages and
  percentile groups, label it N/A in the heatmap, and report the number of estimable cysts and organoids.
- Retain the prompted mask when reverse tracking has only one unique time point.
- Draw every positive cyst identifier in exported videos, including identifiers above 20.

#### Saved results and delivery

- Preserve unrelated files across re-export, including files incorrectly inventoried by older versions.
- Validate explicitly relocated video content even when exporting only reports.
- Route Tk reanalysis through the shared saved-result exporter and require a copy for changed inputs.
- Distinguish dirty source runs with a local-change fingerprint.
- Open the container desktop independently of background model preparation, while validating known
  public checkpoints immediately before inference.

### Scientific behavior (results versions 2 and 3)
- Raw CSV measurements retain floating-point precision, so short or irregular time intervals,
  calibrated areas and circularity survive CSV reload without deliberate rounding. Cyst-summary
  initial and final areas now follow chronological frame order even after raw CSV rows are reordered.
  These corrections are results version 3. Tracking masks and measurement formulas are unchanged;
  older saved results can still be exported, with their original results version recorded as the source.
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
- Reproducible CPU directory-bundle recipe with a bundled CLI, Qt smoke check, dependency notices,
  package inventory and source revision in run provenance. Native Windows distribution validation
  and installer selection remain pending; NVIDIA use continues through the locked source installation.
- PySide6 desktop preview (`organoidtracker-qt`): source-pixel annotation canvas, zoom/pan,
  undo/redo, model settings, cooperative tracking/cancellation, automatic run directories,
  saved-result reopening, mask views, figures, reports and video exports. Missing source videos
  can be relocated by content hash. The Tk interface remains available during acceptance.
- Reanalysis copies accept changed calibration and timing while retaining masks and inference
  provenance. The saved result, session, prompt record and report use the same inputs, including
  after a second reopen/export; the original run remains intact.
- Optional local-snapshot Transformers SAM 2/2.1 backend (`hf-sam2`), selected in sessions or with
  `run --backend` / `--checkpoint`. The reference backend remains the default. Model size and family
  are checked, snapshot hashes enter provenance, and cancelled runs can be repeated from their prompts.
- `organoidtracker bench` writes timing, memory and mask-comparison CSV/JSON alongside a complete
  saved run. Missing masks and identity swaps are visible; comparison against an unrelated video is refused.
- The window saves every tracking run that kept masks right after tracking, without a dialog, into a new
  directory `data/output_videos/runs/<video>_<timestamp>_<run id>/` holding the files Save Results writes
  (`results.json`, the mask file, `session.json`, `prompts.json`). A run that is never exported or saved by hand
  can still be reopened with Open Results or exported again with `organoidtracker export --run`; completed,
  partial and cancelled runs keep their status. An existing directory is never used. Generating the analysis
  report keeps the saved run in step with it: its session, prompt record and result are rewritten with the time
  lapse, conversion factor and organoids the report used (the masks stay), so exporting the saved run again
  reproduces the report. If the save fails, or the time lapse or conversion factor is invalid, the results stay
  in the window, a warning says why and no partial run is left; Save Results still writes a copy into any other
  directory.
- Tracking can be cancelled. The progress dialog's Cancel button and the command line's first Ctrl-C ask
  the propagation to stop after the frame in progress (SAM 2's per-frame inference is not interruptible);
  the tracked frames keep their masks and are saved, analyzed and exported like a partial run, labelled
  `cancelled` in the saved result, the summary, the PDF and the manifest (exit status 4; a second Ctrl-C
  aborts at once). A run cancelled before any mask was kept writes only the session and the prompt record.
  Tracking again on the same loaded video reproduces an uninterrupted run. The window ignores completion and
  progress callbacks of a run that is no longer the current one.
- Every run saves its complete tracking result next to the exports, before any export is written:
  `results.json` (schema `organoidtracker.results/1`: run id and time, results version, software,
  environment, effective settings, the validated session including organoids without cysts, the video
  facts, the backend provenance, the tracking status and coverage, the presence scores and an index of
  every mask) and `masks-<digest>.npz` (every mask as packed bits, the tracker's own representation).
  `organoidtracker export --run DIR [--out DIR2]` reopens such a run without a model and writes the CSV
  tables, figures, PDF, summary and manifest again, and the videos when the video file is at hand, equal
  to the original run's; in place with `--overwrite`, or into another directory which then holds a copy
  of the result. Damaged, truncated, edited or inconsistent result files are refused with the reason; an
  interrupted or failed save keeps the previous result usable. The manifest records which saved result a
  directory holds and, for an export, which run it came from.
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
  uniform time axis derived from the time-lapse span. `analysis_summary.json` then records the whole
  axis (`experiment_info.frame_timestamps`) and re-plotting from CSV keeps it for every frame,
  including frames without measurements.
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
- A prompt and provenance record (`data/output_videos/prompts/<video>_<timestamp>_<run id>.json`) with
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
- On a Linux desktop the window tests run on a private Xvfb display that the test session starts, so a test run
  never opens windows on the screen or takes the focus from other applications; without Xvfb they are skipped
  with that reason (`OT_GUI_ON_DESKTOP=1` runs them on the desktop). CI is unchanged.
- The Tk application runs its workflow through the shared services: the backend is created, loaded
  and prompted through the tracking service, the videos and the analysis report are written by the
  export service with the same measurements and files as a headless run of the same session, and a
  report that cannot be completed (for example a PDF that cannot be written) is reported as a failure
  in the window instead of as a success.
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
- 22 settings that no code ever read, here or in the original release: `default_total_organoids`,
  `default_video_quality`, `debug_mode_enabled`, `enable_comprehensive_reports`, `figure_size`,
  `suppress_matplotlib_warnings`, `fallback_for_optional_deps`, `plot_colors`, `generate_debug_frames`,
  `auto_save_analysis_data`, `include_session_metadata`, `csv_encoding`, `json_ensure_ascii`, `pdf_page_size`,
  `max_video_frames`, `memory_optimization`, `garbage_collection_frequency`, `torch_device_override`,
  `sam2_model_cache`, `enable_model_validation`, `enable_performance_logging` and `log_analysis_details`, with
  their `config.py` constants. Setting them never changed anything (figure sizes are fixed in the plotting code,
  the device is chosen in the window or the session, every frame is read). A settings file or `user_config.py` that still sets
  one loads with a warning and the value is ignored; a later release will refuse them like any unknown key. Run
  manifests and saved results no longer list them; saved runs that do still open and export.
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
- Closing the window while Save Results was writing could stop the save between two renames; the save now finishes
  before the application exits.
- Start Tracking stayed enabled while a video was loading, so a run could share the tracker with the loader; it is
  now disabled and refused until the video has loaded, and a run whose video was replaced while it ran is discarded.
- Worker threads updated the window with `after()` calls, which Tcl/Tk 9 (bundled with the
  uv-managed interpreters) never delivers from another thread: the window never learned that the
  model had loaded or that tracking had finished. Workers now post their updates to a queue that the
  main loop drains, so the window works on Tcl/Tk 8.6 and 9 alike.
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
