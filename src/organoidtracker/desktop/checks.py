"""Deployment checks: an explicitly selected GPU must perform real computation."""

import json
import os
from pathlib import Path


def check_device(mode: str) -> dict:
    import torch

    if mode not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda explicitly.")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU mode cannot access CUDA. Check the NVIDIA driver and Docker GPU runtime, or select CPU."
        )
    value = torch.arange(16, dtype=torch.float32, device=mode).reshape(4, 4)
    result = (value @ value.T).sum().item()
    if result != 3680:
        raise RuntimeError(f"{mode} computation check returned an unexpected result.")
    return {
        "mode": mode,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if mode == "cuda" else "CPU",
    }


def check_paths() -> None:
    for name in ("/videos", "/models"):
        if not Path(name).is_dir() or not os.access(name, os.R_OK | os.X_OK):
            raise RuntimeError(f"Cannot read {name}. Check the selected host directory and its permissions.")
    for name in ("/results", "/preferences"):
        directory = Path(name)
        if not directory.is_dir():
            raise RuntimeError(f"Mount a writable host directory at {name}.")
        from tempfile import TemporaryFile

        try:
            with TemporaryFile(dir=directory) as stream:
                stream.write(b"write check")
        except OSError as error:
            raise RuntimeError(f"Cannot write {name}. Check its host ownership and permissions.") from error


def main() -> None:
    print(json.dumps(check_device(os.environ.get("OT_DEVICE", "cpu"))))


if __name__ == "__main__":
    main()
