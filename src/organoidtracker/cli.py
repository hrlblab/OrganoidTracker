"""The ``organoidtracker`` command line: headless tracking, analysis and export.

    organoidtracker run --session SESSION.json --out RUN_DIR [--video VIDEO] [--device cuda|cpu]
                        [--model sam2_hiera_t|s|b|l] [--no-videos] [--video-quality original|mid|low]
                        [--overwrite] [--debug] [--log-level LEVEL]
    organoidtracker validate --session SESSION.json

Exit status: 0 the run completed; 1 it failed (no masks, export error); 2 the inputs were
invalid (session file, settings file, missing or different video, bad arguments); 3 the run
stopped early and its exports are partial (they say so in every file).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INVALID = 2
EXIT_PARTIAL = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="organoidtracker", description="Organoid Tracker without a window.")
    parser.add_argument("--version", action="version", version=f"organoidtracker {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="track, analyze and export one session")
    run.add_argument("--session", required=True, type=Path, help="session file (or a prompt record)")
    run.add_argument("--out", required=True, type=Path, help="output directory for the run")
    run.add_argument("--video", type=Path, default=None, help="use this video file instead of the session's path")
    run.add_argument("--device", default=None, help="override the session's device (cuda, cuda:N or cpu)")
    run.add_argument("--model", default=None, help="override the session's model size (sam2_hiera_t, _s, _b, _l)")
    run.add_argument("--no-videos", action="store_true", help="skip the overlay, mask and side-by-side videos")
    run.add_argument("--video-quality", default="original", choices=["original", "mid", "low"])
    run.add_argument("--overwrite", action="store_true", help="allow writing into a directory that holds a run")
    run.add_argument("--debug", action="store_true", help="debug outputs of the analysis and video steps")
    run.add_argument("--log-level", default=None, help="console log level (default: the log_level setting)")

    validate = commands.add_parser("validate", help="check a session file and its video without running")
    validate.add_argument("--session", required=True, type=Path)
    validate.add_argument("--video", type=Path, default=None, help="use this video file instead of the session's path")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # PyTorch + NumPy OpenMP clash on Windows
    args = build_parser().parse_args(argv)

    from .settings import SettingsError

    try:
        from . import config
    except SettingsError as error:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)-8s %(message)s")
        logger.error(f"Invalid settings, not running: {error}")
        return EXIT_INVALID

    if args.command == "validate":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
        return _validate(args.session, args.video)
    return _run(args, config)


def _load(session_path: Path, video_override: Path | None):
    from .services.session import load_session

    session = load_session(session_path)
    if video_override is not None:
        session = session.with_video(Path(video_override))
    return session


def _validate(session_path: Path, video_override: Path | None) -> int:
    from .services.pipeline import check_video
    from .services.session import SessionError

    try:
        session = _load(session_path, video_override)
    except SessionError as error:
        print(f"invalid: {error}", file=sys.stderr)
        return EXIT_INVALID
    annotations = session.annotations
    print(f"session: {session_path}")
    print(f"video: {session.video.path}")
    print(
        f"tracking: {session.tracking.direction}, model {session.tracking.model_config}, "
        f"family {session.tracking.checkpoint_family or 'default'}, device {session.tracking.device}"
    )
    print(f"calibration: {session.calibration.um_per_pixel} um/pixel")
    print(f"timing: {session.timing.to_document()}")
    print(
        f"annotations: {len(annotations.organoids)} organoids, {len(annotations.cysts)} cysts"
        + (
            f", organoids without cysts: {annotations.organoids_without_cysts()}"
            if annotations.organoids_without_cysts()
            else ""
        )
    )
    try:
        digest = check_video(session)
    except SessionError as error:
        print(f"invalid: {error}", file=sys.stderr)
        return EXIT_INVALID
    print(f"video sha256: {digest}" + ("" if session.video.sha256 else " (not recorded in the session)"))
    print("valid")
    return EXIT_OK


def _run(args: argparse.Namespace, config) -> int:
    from dataclasses import replace

    from .logging_config import configure_logging
    from .services.annotations import AnnotationError
    from .services.export_service import ExportError
    from .services.pipeline import run_session
    from .services.session import SessionError
    from .services.tracking_service import TrackingError

    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)-8s %(message)s")
        logger.error(f"cannot create the output directory {out}: {error}")
        return EXIT_INVALID
    log_file = configure_logging(args.log_level, log_file=out / "organoidtracker.log")
    logger.info(f"organoidtracker {__version__}")
    if log_file is not None:
        logger.info(f"Log file: {log_file}")
    if config.LOADED_SETTINGS_FILE is not None:
        logger.info(f"Settings file: {config.LOADED_SETTINGS_FILE}")

    try:
        session = _load(args.session, args.video)
        if args.device or args.model:
            session = replace(
                session,
                tracking=replace(
                    session.tracking,
                    device=args.device or session.tracking.device,
                    model_config=args.model or session.tracking.model_config,
                ),
            )
    except SessionError as error:
        logger.error(f"Invalid session: {error}")
        return EXIT_INVALID

    def progress(phase: str, current: int, total: int, message: str) -> None:
        if phase == "tracking":
            logger.info(f"[{phase}] {message}")
        else:
            logger.debug(f"[{phase}] {message}")

    try:
        outcome = run_session(
            session,
            out,
            videos=not args.no_videos,
            video_quality=args.video_quality,
            overwrite=args.overwrite,
            debug=args.debug,
            progress=progress,
        )
    except (SessionError, AnnotationError) as error:
        logger.error(f"Invalid input: {error}")
        return EXIT_INVALID
    except ExportError as error:
        logger.error(f"Export failed: {error}")
        return EXIT_INVALID if "already holds a run" in str(error) else EXIT_FAILURE
    except TrackingError as error:
        logger.error(f"Tracking failed: {error}")
        return EXIT_FAILURE

    info = outcome.summary.get("experiment_info", {})
    logger.info(
        f"Tracking {outcome.result.summary()}; {info.get('total_organoids')} organoids, "
        f"{info.get('total_cysts')} cysts with trajectories; outputs in {outcome.output_dir}"
    )
    if not outcome.complete:
        logger.warning("The run is PARTIAL: exports cover the tracked frames only (exit status 3)")
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
