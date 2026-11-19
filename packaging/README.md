# CPU directory bundle

Build on the target operating system from the locked CPU environment. This is a portable directory
prototype; it is not a Windows installer. Native Windows CPU/GPU validation remains required before
a release. The Linux prototype has been exercised outside the source checkout.

```bash
# Use a separate environment so the working NVIDIA installation is not replaced.
UV_PROJECT_ENVIRONMENT=build/packaging-cpu uv sync --locked --extra cpu --extra qt --group packaging --no-dev
build/packaging-cpu/bin/pyinstaller --distpath dist --workpath build/pyinstaller packaging/organoidtracker.spec
```

On Windows set `UV_PROJECT_ENVIRONMENT` in PowerShell and use the environment's `Scripts/pyinstaller.exe`.
Build from a clean commit for a release. The bundle's `_internal/organoidtracker/_build_info.json`
records the revision, dirty state and runtime; result provenance reads that revision. The spec gathers
Hydra YAML configs, model modules, package metadata, dependency license files and a package inventory.
It excludes checkpoints, videos, research outputs, Tk and the optional Transformers backend.
Qt's installed wheels do not contain standalone license texts. The recipe therefore also includes
the unmodified texts from the matching official Qt for Python source tag in `licenses/qt`, with
their source URLs and hashes. It includes the interpreter's license when supplied by the runtime.

Run from a writable workspace, with checkpoints outside the bundle:

```bash
/path/to/organoidtracker/organoidtracker                       # Qt application
/path/to/organoidtracker/organoidtracker --cli --version
/path/to/organoidtracker/organoidtracker --cli run --session session.json --checkpoint /path/to/sam2.1_hiera_tiny.pt --device cpu --out new-run
QT_QPA_PLATFORM=offscreen /path/to/organoidtracker/organoidtracker --smoke-test
```

The smoke check creates a window, closes it, and writes its temporary preferences under a temporary
directory. It proves Qt startup, not inference. Validate a bundle with a synthetic real-model run,
then reopen and re-export that saved run using the bundle. Also check that the collected license files
and model configuration files are present. Keep the whole directory together.

For NVIDIA acceleration, use the source installation with `uv sync --extra qt --extra cuda` and a
compatible NVIDIA driver, then `uv run --no-sync organoidtracker-qt`. A CPU-frozen application cannot
be upgraded by running pip against its bundled files; a CUDA bundle would require a separate build
and native GPU validation. The CPU spec deliberately refuses a CUDA build environment. Installer
selection and a GPU distribution format remain deferred until native Windows validation.

The optional Hugging Face backend remains available through the source installation with `--extra hf`.
The [PyInstaller usage guide](https://pyinstaller.org/en/stable/usage.html) describes directory bundles
and their platform-specific build process.
