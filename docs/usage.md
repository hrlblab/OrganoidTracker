# Using Organoid Tracker

The Docker desktop runs the Qt application described here; its own guide is
[packaging/docker/README.md](../packaging/docker/README.md). Installation from source is in
[development.md](development.md), and what the numbers mean is in [measurements.md](measurements.md).

## Qt desktop application

```bash
uv sync --extra qt                  # add --extra hf for the optional Transformers backend
uv run --no-sync organoidtracker-qt
```

Open a video or a saved session. Confirm the acquisition selection before annotating; until then a
click on the picture explains what is missing. The canvas then shows the last selected frame for
reverse tracking. Click an organoid, drag boxes around its cysts, then click the next organoid.
Include organoids without cysts because they contribute to the population statistics. Select an
existing organoid in the annotation list to add more cysts to it. Use the wheel to zoom, right or
middle drag to pan, and Fit to restore the view. Undo/Redo and removal operate on source-pixel
coordinates at every zoom level.

Choose the model family, size, device and weights, then check the pixel calibration and time span.
The acquisition panel lets you inspect source frames with the slider and select their one-based
numbers. **7 daily points (11-frame export)** fills source frames `1, 3, 4, 6, 7, 9, 10` and days
`1–7` for that confirmed export layout only. Check the mapping and click **Confirm frames and days**.
For other videos, select every source frame or enter your own list and dates. A newly opened video has
no default span: enter the days from the first to the last frame, or one day per frame, and check the
table of source frames and days before confirming. Clear annotations before changing frames so that
existing boxes cannot move to another observation; for the same reason the tracking direction is locked
while annotations exist and is not remembered between sessions. Days of the same frames can be corrected
with annotations in place, and the changed days must be confirmed again before tracking. The selected
source indices and actual days are saved with each session and run. Reopening old results preserves their
recorded map; old sessions without a map need confirmation before new tracking.

**Track and save run** writes
each run into its own directory under `data/output_videos/runs/` (change its parent with **Run folder**).
Masks are saved before reports, so a report failure can be recovered with **Open results**. Cancel
finishes the current frame and preserves retained masks; closing during a task waits for that task
and its save to finish. Partial and cancelled runs retain their status.

Use the frame slider to inspect originals, overlays, masks and side-by-side views. Each mask carries
its cyst label (`C1`, `C2`, …) in the colour shown in the annotation list; selecting a cyst there
outlines its mask. The Figures tab
shows the generated plots; **Open report folder** provides the PDF, CSV tables and videos. Saved
annotations stay read-only until **Edit prompts for a new run**. **Export reports to a new copy**
re-analyzes retained masks with the selected calibration/timestamps, and preserves those inputs in
every copied session and result document. The original run remains intact. Missing source videos
do not prevent mask inspection or report export without videos; **Locate source video** accepts a
relocated file only when its content hash matches.

The Qt interface is a preview while interactive Linux acceptance and native Windows CPU/GPU checks
are pending. The Tk application remains available. Run Qt regression tests offscreen with
`uv sync --extra qt --group qt-test`, then `QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -m qt`
(on Windows, set that environment variable using the shell's syntax).

An experimental CPU directory bundle can also be built using the [packaging recipe](../packaging/README.md).
Checkpoints remain external to the bundle; native Windows distribution validation is pending.

## Tk application

The original interface. It runs through the same tracking, analysis and export code.

![The Tk interface](../figures/fig2.png)

1. **Launch the Application**
   ```bash
   uv run organoidtracker-tk
   ```
   (`python video_tracker_gui.py` inside the environment does the same.) Outputs, the prompt
   records and the log file (`data/output_videos/organoidtracker.log`) are written under `data/`
   in the directory you launch from.

2. **Load Your Video**
   - Click "Load Video" and select your time-lapse organoid video
   - Supported formats: MP4

3. **Configure Tracking**
   - Select the SAM2 model size (small, base-plus or large)
   - Keep "Reverse Tracking" enabled (recommended for organoid analysis)
   - Adjust tracking thresholds and other settings in `organoidtracker.toml` if needed (see Settings below)

4. **Annotate Cysts**
   - With Reverse Tracking enabled the canvas shows the final frame of the video, where cysts are clearest
   - Left-click an organoid, then click and drag bounding boxes around its cysts; left-click again to start the next organoid
   - Each cyst is assigned a unique color
   - Confirm source frames and acquisition days before annotating. Similar-looking images are never automatically discarded.

5. **Run Analysis**
   - Click "Start Tracking" to begin automated segmentation
   - Monitor progress in the log panel
   - Processing time varies with video length and model size

6. **Review Results**
   - Generated outputs saved to `data/output_videos/`
   - Includes tracking videos, analysis reports, and raw data
   - Every tracking run that kept masks is saved automatically, right after tracking, in a directory of
     its own: `data/output_videos/runs/<video>_<timestamp>_<run id>/` with `results.json`, the mask file,
     `session.json` and `prompts.json`. It reopens with "Open Results..." and exports again with
     `organoidtracker export --run <directory> --out <new directory>`, even if it was never exported or saved by hand.
     Generating the analysis report updates it with the time lapse, conversion factor and organoids the report
     used, so that exporting it again reproduces the report; "Save Results..." writes a copy into a directory of
     your choice.

## Command line

The same tracking, analysis and exports run without a window from a *session file*: a JSON document
with the video, the organoid points and cyst boxes (source pixels on the annotation frame, the last
frame with reverse tracking), the calibration and the timing. The prompt record the application writes
when tracking starts (`data/output_videos/prompts/*.json`) is accepted as a session file, so a GUI run can
be repeated headlessly.

```json
{
  "schema": "organoidtracker.session/1",
  "video": {"path": "well.mp4", "decoded_frames": 11, "frame_indices": [0, 2, 3, 5, 6, 8, 9]},
  "tracking": {"direction": "reverse", "model_config": "sam2_hiera_b", "device": "cuda"},
  "calibration": {"um_per_pixel": 1.6934},
  "timing": {"time_lapse_days": 6.0},
  "organoids": [
    {"organoid_id": 1, "point": [1300, 700],
     "cysts": [{"cyst_id": 1, "bbox": [1200, 800, 1400, 1000]}, {"cyst_id": 2, "bbox": [1500, 820, 1650, 960]}]},
    {"organoid_id": 2, "point": [600, 2400], "cysts": []}
  ]
}
```

```bash
uv run organoidtracker validate --session session.json
uv run organoidtracker run --session session.json --out runs/well-01 [--video other/path.mp4] [--device cpu] [--model sam2_hiera_t] [--no-videos] [--video-quality original|mid|low]
uv run organoidtracker export --run runs/well-01 [--out runs/well-01-replot] [--video other/path.mp4] [--no-videos] [--video-quality original|mid|low] [--overwrite]
```

A session's `video.frame_indices` is a strictly increasing list of **zero-based** source indices;
`video.decoded_frames` records the expected decoded count. A mismatch fails before tracking.
For an older session without a selection, CLI `run`/`bench` accept
`--frames 1,3,4,6,7,9,10 --decoded-frames 11` (one-based CLI numbers). Confirm that the session's timing
describes those selected observations. A session that already records a selection refuses different
values: its boxes and days belong to its frames, so another selection is confirmed in the application.
`validate` applies the same checks as `run`: `frame_times_days` has one value per selected frame, and
the selection fits the frame count in the video's index. A prompt record written before results version 5
names frames that the older software selected itself; it is read as an older session, so a new run from it
needs `--frames` as well (the warning prints the recorded frames, which repeat the old selection).
`timing` takes either `time_lapse_days` (spread uniformly from day 1 as in the paper) or
`frame_times_days`, one explicit time per selected time point. Every key of `tracking` is optional; `--video`
relocates a moved video (its recorded `sha256`, when present, must still match). The output directory
receives the CSV tables, figures and PDF report (re-plottable with `scripts/csv_visualizer.py`), the
videos under `videos/`, the prompt record (`prompts.json`), the validated session (`session.json`), the log,
the saved tracking result (`results.json` and `masks-<digest>.npz`: every mask as packed bits, with the
organoid and cyst identities, the time axis, the calibration, the frame map, the tracking status and
coverage, the presence scores, the settings and the provenance; written right after tracking, before any
export), and `run_manifest.json`: software, environment and settings, the video facts, the backend
provenance, the tracking status and frame coverage, a digest of every mask, and the hash of every file written.

`organoidtracker export` reopens a run directory **without a model** and writes its analysis again: the
CSV tables, figures, PDF, summary and manifest, plus the videos when the video file is available (`--video`
relocates a moved file with the same content; `--no-videos` needs no video at all). In place (`--run` alone)
it needs `--overwrite` and keeps the saved result, session, prompt record and log; with `--out` another
directory receives a copy of the saved result as well, so it is a complete run directory too. The exports
equal the original run's for the same software and settings. A damaged, truncated or edited result file is
refused. Tracking again from the saved prompts is `run --session runs/well-01/session.json`; continuing an
interrupted propagation exactly is not supported (SAM 2 keeps an inference memory beyond the saved masks).

Exit status: 0 completed; 1 failed; 2 invalid input (session, settings, saved result, missing or different
video); 3 the tracking stopped early and the exports are **partial** (the manifest, the summary and the PDF
say so; `export` reports the status of the run it exports again); 4 the run was **cancelled**: the first Ctrl-C
asks the tracking to stop after the frame in progress (SAM 2's per-frame inference is not interruptible), the
tracked frames are saved, analyzed and exported with the status `cancelled`, and a second Ctrl-C aborts at once.
A run cancelled before any mask was kept writes only the session and the prompt record.

## Settings

Defaults live in the package (`organoidtracker.settings.Settings`). To change any of them, copy
[`organoidtracker.example.toml`](../organoidtracker.example.toml) to `organoidtracker.toml` in the directory you
launch from (or point `ORGANOIDTRACKER_SETTINGS` at a file) and uncomment the keys you need; keys and value
types are checked at startup. A legacy `user_config.py` is still read for now, with a deprecation warning.

## Optional Transformers backend and benchmarks

The reference SAM 2 backend remains the default. Install `uv sync --extra hf` to opt into
`hf-sam2`, which loads a **local** Hugging Face SAM 2/2.1 snapshot. Model loading never downloads
weights. For a reproducible tiny-model example, download the public snapshot separately:

```bash
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam2.1-hiera-tiny', revision='de431c4043854a71d8101e17995dfe596bf101a5', allow_patterns=['*.json', 'model.safetensors'], local_dir='checkpoints/hf/sam2.1-hiera-tiny')"
uv run --no-sync organoidtracker bench --session session.json --out reference --model sam2_hiera_t --no-videos
uv run --no-sync organoidtracker bench --session session.json --out candidate --model sam2_hiera_t --backend hf-sam2 --checkpoint checkpoints/hf/sam2.1-hiera-tiny --reference reference --no-videos
```

`tracking.backend` in a session can also select `hf-sam2`; `tracking.checkpoint_path` then names
the snapshot directory. Match the session's model size and checkpoint family to that snapshot.
The adapter uses the reference's PIL preprocessing, chronological frames, quality filters and packed
masks. The application-specific improved base-plus configuration is unsupported. This backend is
experimental; it still shares code with the vendored adapter and does not replace the vendored package.

Each benchmark produces a normal saved run plus `benchmark.json` and `benchmark.csv`. These record
tracking time, seconds per processed frame, process peak RAM (unavailable on Windows), CUDA allocated
peak memory, mask intersection-over-union (IoU), area differences and changes in reference identity
assignment. Missing and extra masks score zero IoU. The first accepted frame means the earliest
chronological frame with a retained mask, not a biological appearance time. A reference must use the
same video; annotation/direction agreement and both provenance records are included. Run one case per
process because RAM is a lifetime peak through inference. Tracking timings include each backend's
state preparation inside propagation; pipeline stage timings are also recorded. These comparisons
measure agreement with another implementation, not accuracy against expert annotations.
