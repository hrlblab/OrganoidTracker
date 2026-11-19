# Local Docker desktop (Linux preview)

This bundle runs the complete Organoid Tracker interface in your local browser. You select a video,
annotate organoids and cysts, track, inspect masks and export reports in the existing desktop interface.
Your videos and results stay on your computer. The browser is a view of the container's desktop;
closing its tab leaves a running job active.

This is a Linux x86-64 preview. The integration host is Ubuntu 26.04 with Docker Engine 29.1.3 and an NVIDIA RTX 5080 Laptop GPU.
A separate clean-host installation has not yet been validated. Windows and macOS are not supported
by this preview. No host Python environment, Qt installation or display-server setup is required.

## Prepare the host once

Install Docker Engine and give your account access to its daemon. Follow the official
[Docker installation instructions](https://docs.docker.com/engine/install/).
For GPU mode, install a compatible NVIDIA driver and configure the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The CUDA 13.0 image needs a driver supporting CUDA 13; this build is tested with driver 595.99.02.
The launcher checks actual GPU computation and fails if the selected GPU mode cannot run.
Choose `cpu` explicitly to use a computer without a supported GPU.

## Download

Each [release](https://github.com/hrlblab/OrganoidTracker/releases) provides these files:

- `organoidtracker-docker-<version>.tar.gz`: the launcher, this guide, the licenses, the checksums and
  the image identifiers;
- `images-cpu.tar`: the CPU image (about 0.9 GB);
- `images-cuda.tar.part-00` and `images-cuda.tar.part-01`: the NVIDIA image (about 3.5 GB) in two parts,
  because a release file must be smaller than 2 GiB.

Download the first file and the image for your computer into one folder. Then unpack the bundle, move
the image beside the launcher, join the NVIDIA parts and check the downloads:

```bash
tar -xzf organoidtracker-docker-<version>.tar.gz
mv images-*.tar* organoidtracker-docker-<version>/
cd organoidtracker-docker-<version>
cat images-cuda.tar.part-* > images-cuda.tar && rm images-cuda.tar.part-*   # NVIDIA image only
sha256sum -c --ignore-missing SHA256SUMS
```

Every listed file must report `OK`. Download a file again when it reports `FAILED`.

Keep the complete versioned bundle together. CPU and CUDA image archives are separate, so recipients
need only their selected variant. Allow storage for the image archive, Docker's unpacked image,
model weights and your results. Initial installation loads the included image; it does not build it.

## First start

Open a terminal in the bundle directory and run:

```bash
./organoidtracker
```

Enter your existing video directory, a results directory, a reusable model-cache directory and
`cuda` or `cpu`. Paths may contain spaces and Unicode. Commas and newlines are not supported.
The launcher remembers these choices and loads the included image. It opens the application while
preparing the public default SAM 2.1 Base-plus checkpoint in the background. Saved results remain
usable if preparation fails; tracking checks the selected checkpoint before inference.
The model cache survives application updates. Public model source and license:
[Meta SAM 2](https://github.com/facebookresearch/sam2).

In the interface:

1. Choose **Open video**. Your selected host video folder appears as `/videos`.
2. Confirm the acquisition time points in the **Acquisition time points** panel on the left. Annotation
   stays locked until you do, and a click on the picture says so. Check **Source frames**; for the
   eleven-frame daily export choose **7 daily points (11-frame export)**, otherwise enter the frames
   yourself. Enter either the span in days from the first to the last frame or one day per frame:
   nothing is assumed for a newly opened video. The small table shows each source frame with the day
   that will be recorded. When it is right, click **Confirm frames and days**.
3. The picture now shows the annotation frame (the last time point). Click an organoid, then drag
   boxes around its cysts. Add further organoids as needed, including organoids without cysts.
   The tracking direction is locked while annotations exist, because boxes belong to the frame
   they were drawn on. Days can still be corrected: change them and confirm again.
4. Choose **Track and save run**. Each run creates its own directory under `/results`.
5. Use the chronological slider, display modes and figures to inspect saved results. Every mask is
   labelled `C` plus its cyst number, in the colour shown beside that cyst in the annotation list;
   selecting a cyst in the list outlines its mask. The overlay video, the **Cyst Key** picture in
   the PDF and the `Cyst_ID` column of the CSV tables use the same numbers.
6. Choose **Open report folder**, then the **Open report folder** link above the desktop. The
   **Results** link also lists all saved files. PDF, CSV and video files download to your browser.
   The same files are already in your selected host results directory. The first page of the PDF
   names the video, the run, the model, the source frames and their days.

Use **Display at 1:1** to inspect the desktop without browser downscaling. Canvas zoom and right- or
middle-button dragging change the view without changing source-pixel annotations. The pinned viewer
prefers lossless ZRLE transmission. Exported videos retain the existing codec; download them if your
browser cannot play them.

The default model remains SAM 2.1 Base-plus. The image supports the reference SAM 2 backend; the
optional Transformers research backend is visibly unavailable. To cache the smaller Tiny checkpoint,
run `./organoidtracker models` and choose `tiny`, then choose Tiny in the application's Size selector.
Other compatible local SAM 2 checkpoints can be placed in the model-cache folder and selected with
**Browse weights** at `/models`. Their family and size must match the selected model. Weight files
are external to the image and read-only during inference.

## Repeat use, stop and recovery

Run `./organoidtracker` again to reconnect to the existing application or start a new one using saved
folders and preferences. Cached images and weights allow repeat use without internet access.
Only a loopback browser port is published; raw VNC is private inside the container.
Use this preview on a trusted single-user host. The desktop has no login: other users or processes
on the same host can connect and access the mounted data and results. Loopback binding and browser
origin checks restrict network exposure; they do not authenticate local users.

Use **Stop application** above the desktop or:

```bash
./organoidtracker status
./organoidtracker stop
```

Stopping asks Qt to finish the current frame, cancel the remaining tracking and save retained masks
and reports before exiting. The launcher waits up to 300 seconds. A forced kill cannot guarantee
saving an in-progress operation; reopen the most recent valid saved results. **Open results** and
**Export reports to a new copy** do not need a model. Exact continuation of SAM 2's interrupted
inference state is not supported; start a new run from saved prompts when required.

Save sessions and exports under `/results`, the persistent host results folder. `/tmp` is temporary.
Videos at `/videos` and models at `/models` are read-only. Existing native sessions/results can have
old paths; use the existing source relocation workflow and its content validation.

Stop the current application before `./organoidtracker setup` changes folders or device mode.
For an update, stop it and run the new version's launcher with the same saved configuration. It
keeps previous images and host data; running the previous bundle selects its previous image again.
The launcher refuses a second configuration writing the same results workspace.

If startup fails, the launcher shows the prerequisite failure. Check directory permissions, available
storage and Docker GPU access. `organoidtracker.log` in the results folder records application work.
A VPN occupying Docker's default private address pools may prevent Docker itself from starting;
that is a host networking issue. Choose an unused Docker address pool with the host administrator;
the launcher does not modify VPN routes, drivers, daemon settings or privileges.

## Build and provenance (maintainers)

From a clean reviewed checkout:

```bash
packaging/docker/build-bundle.sh /path/to/bundle both
```

The build uses the locked application dependencies, digest-pinned Python/uv bases, and a checksummed
noVNC 1.7.0 source archive. Debian packages are resolved at build time; the resulting image ID is the
immutable runtime artifact. Rebuilding later is not promised to reproduce its bytes. Retain the
versioned image archives and their checksums for exact distribution/rollback.

To prepare the release files from a built bundle directory (`<version>` is the application version):

```bash
cd /path/to/bundle
split -b 1900M -d images-cuda.tar images-cuda.tar.part-
tar -czf organoidtracker-docker-<version>.tar.gz --transform 's,^,organoidtracker-docker-<version>/,' \
    organoidtracker README.md LICENSE NOTICE SHA256SUMS source-revision.txt source.tar image-*.txt image-*.json
```

Upload the `.tar.gz`, `images-cpu.tar` and the `images-cuda.tar.part-*` files. `SHA256SUMS` covers the
joined archives, so the recipient's check also proves the parts were joined correctly.

The image includes application source in `/build/src`, the source revision in package build metadata,
noVNC source and licenses in `/opt/novnc`, installed dependency license metadata, Qt license texts and
a package inventory under `/opt/licenses`. `prefer-lossless.py` changes only noVNC's encoding preference
order, putting ZRLE before Tight to prevent JPEG artifacts. It checks the expected pinned source and
refuses an unexpected upgrade. Upstream copyrights and licenses remain intact. The image contains no
model weights, private videos, user preferences or research outputs; `.dockerignore` is an allow-list.

The host launcher uses Bash and Docker plus standard Linux utilities. `ORGANOIDTRACKER_HOME` selects
an alternate configuration directory; `ORGANOIDTRACKER_NO_BROWSER=1` suppresses browser opening for
integration checks. These are optional, not part of routine recipient setup.

### Model preparation and saved reports

The launcher opens the desktop while it prepares the default checkpoint in the background.
Progress and failures are recorded in `model-preparation.log` beside the launcher settings.
Opening saved results and exporting reports require no checkpoint or network connection.
Tracking verifies the selected public checkpoint before loading it. If it is missing or
corrupt, move corrupt files aside and run `./organoidtracker models` to prepare the cache.
