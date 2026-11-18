#!/usr/bin/env bash
# Maintainer-only: build from a clean checkout; recipients use organoidtracker.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
[[ -z $(git status --porcelain --untracked-files=normal) ]] || { printf 'Commit the reviewed changes before building a delivery bundle.\n' >&2; exit 1; }
revision=$(git rev-parse HEAD)
destination=${1:?Usage: build-bundle.sh OUTPUT_DIRECTORY [cpu|cuda|both]}
selection=${2:-both}
case "$selection" in cpu|cuda) modes=("$selection");; both) modes=(cpu cuda);; *) exit 2;; esac
mkdir -p -- "$destination"
destination=$(realpath -- "$destination")
for mode in "${modes[@]}"; do
    tag="organoidtracker:${revision:0:12}-$mode"
    docker build --build-arg "ACCELERATOR=$mode" --build-arg "SOURCE_REVISION=$revision" \
        --build-arg SOURCE_DIRTY=false -f packaging/docker/Dockerfile -t "$tag" .
    docker image inspect --format '{{.Id}}' "$tag" > "$destination/image-$mode.txt"
    docker image inspect "$tag" > "$destination/image-$mode.json"
    docker save -o "$destination/images-$mode.tar" "$tag"
done
cp packaging/docker/organoidtracker packaging/docker/README.md LICENSE NOTICE "$destination/"
printf '%s\n' "$revision" > "$destination/source-revision.txt"
# Keep source available with the image, including the pinned noVNC patch.
git archive -o "$destination/source.tar" "$revision"
(cd "$destination" && sha256sum ./*.tar > SHA256SUMS)
printf 'Local bundle prepared at %s. Publication is a separate action.\n' "$destination"
