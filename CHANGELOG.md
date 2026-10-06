# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Changes that alter scientific results (masks, measurements, time axes) are listed under
a "Scientific behavior" heading so that analyses can be attributed to a version.

## [Unreleased]

### Added
- This changelog. The upstream state before maintenance began is tagged `legacy-baseline`.

### Fixed
- The Results Viewer module had lost its indentation and could not be imported, so the
  View Results action always failed; it opens again.
- The Results Viewer's frame slider and display update called each other recursively.
- `checkpoints/download_ckpts.sh` no longer aborts on a dead Medical-SAM2 URL. It accepts a
  checkpoint family argument (`2.1` by default, `2`, or `all`), downloads into its own
  directory, skips present files, and reports failures through its exit status. The README's
  manual download commands name the files that actually exist.
- Removed a shadowed duplicate `on_window_resize` definition in the main window (no behavior
  change).

### Removed
- The Medical-SAM2 backend (`models/medical_sam2`, `src/core/inference.py`). Its upstream has
  been dormant since 2024, its checkpoint URL returns 404, and its GPL license badge conflicted
  with this repository's Apache-2.0 license. SAM2 is the only backend.
