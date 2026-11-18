"""Own the Qt process; stop display services only after Qt has saved and exited."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .checks import check_device, check_paths


def main() -> int:
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    processes = []
    qt = None
    try:
        check_paths()
        print(check_device(os.environ["OT_DEVICE"]), flush=True)
        runtime = Path("/tmp/organoidtracker")
        runtime.mkdir(exist_ok=True)
        for name in ("ready", "report.json"):
            (runtime / name).unlink(missing_ok=True)
        # Only this supervisor owns display 99 inside the container. A previous
        # forced stop may leave files when run without the documented tmpfs.
        for name in ("/tmp/.X99-lock", "/tmp/.X11-unix/X99"):
            Path(name).unlink(missing_ok=True)
        os.environ["OT_SUPERVISOR_PID"] = str(os.getpid())

        def start(args):
            process = subprocess.Popen(args)
            processes.append(process)
            return process

        display = start(["Xvfb", ":99", "-screen", "0", "1440x1000x24", "-nolisten", "tcp"])
        for _ in range(100):
            if display.poll() is not None:
                raise RuntimeError("The virtual display failed to start.")
            if subprocess.run(["xdpyinfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("The virtual display did not become ready.")
        start(["openbox"])
        start(
            [
                "x11vnc",
                "-display",
                ":99",
                "-rfbport",
                "5900",
                "-localhost",
                "-forever",
                "-shared",
                "-nopw",
                "-noxdamage",
                "-quiet",
                "-safer",
            ]
        )
        start([sys.executable, "-m", "organoidtracker.desktop.server"])
        qt = start([sys.executable, "-m", "organoidtracker.desktop.qt"])
        requested = False
        failed = False
        while qt.poll() is None:
            if any(p.poll() is not None for p in processes[:-1]):
                stopping = failed = True
            if stopping and not requested:
                print("Stopping: waiting for Qt to finish the current operation and save retained results.", flush=True)
                qt.send_signal(signal.SIGTERM)
                requested = True
            time.sleep(0.1)
        return qt.returncode or (1 if failed else 0)
    except Exception as error:
        print(f"Desktop startup failed: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        # The normal stop loop never kills Qt. Docker's explicit stop timeout is
        # the sole last resort for a hung model; a forced kill cannot promise save.
        if qt is not None and qt.poll() is None:
            qt.send_signal(signal.SIGTERM)
            qt.wait()
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
