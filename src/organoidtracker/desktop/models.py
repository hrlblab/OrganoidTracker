"""Download checked public checkpoints once; inference mounts their cache read-only."""

import argparse
import fcntl
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

# Public Meta SAM 2.1 downloads, checked against the downloaded release files.
MODELS = {
    "base_plus": ("a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5", 323606802),
    "tiny": ("7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69", 156008466),
}
BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root: Path, model: str, *, offline=False) -> Path:
    expected, size = MODELS[model]
    name = f"sam2.1_hiera_{model}.pt"
    target = root / name
    root.mkdir(parents=True, exist_ok=True)
    with (root / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if target.stat().st_size != size or digest(target) != expected:
                raise ValueError(f"{name} does not match its release checksum. Move it aside or choose another cache.")
            print(f"Verified cached checkpoint: {name}", flush=True)
            return target
        if offline:
            raise FileNotFoundError(f"{name} is not cached. Prepare models once while online.")
        if shutil.disk_usage(root).free < size:
            raise OSError(f"The model cache needs at least {size // 1000000} MB free for {name}.")
        print(f"Downloading {name} ({size // 1000000} MB)…", flush=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=name + ".", suffix=".part", delete=False) as out:
                temporary = Path(out.name)
                with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if temporary.stat().st_size != size or digest(temporary) != expected:
                raise ValueError("The checkpoint download failed its release checksum. Run the launcher again.")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Checkpoint ready: {name}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS, nargs="?", default="base_plus")
    parser.add_argument("--cache", type=Path, default=Path("/models"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.cache, args.model, offline=args.offline)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Model preparation failed: {error}\n")


if __name__ == "__main__":
    main()
