#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Download SAM 2 model checkpoints into this directory.
#
# Usage:  bash download_ckpts.sh [2.1|2|all]
#   2.1   SAM 2.1 checkpoints (default; the family the application selects by default)
#   2     original SAM 2 checkpoints (the family used for the paper's figures)
#   all   both families
#
# Existing non-empty files are skipped; delete a file to download it again.

set -u
FAMILY="${1:-2.1}"
cd "$(dirname "$0")" || exit 1

case "$FAMILY" in
    2|2.1|all) ;;
    *) echo "Usage: bash download_ckpts.sh [2.1|2|all]"; exit 2 ;;
esac

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824"
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

status=0
download() {
    local url="$1"
    local file
    file="$(basename "$url")"
    if [ -s "$file" ]; then
        echo "$file already exists, skipping."
        return
    fi
    echo "Downloading $file ..."
    $CMD "$url" || { echo "Failed to download $url"; status=1; }
}

if [ "$FAMILY" = "2" ] || [ "$FAMILY" = "all" ]; then
    for name in sam2_hiera_tiny sam2_hiera_small sam2_hiera_base_plus sam2_hiera_large; do
        download "${SAM2_BASE_URL}/${name}.pt"
    done
fi

if [ "$FAMILY" = "2.1" ] || [ "$FAMILY" = "all" ]; then
    for name in sam2.1_hiera_tiny sam2.1_hiera_small sam2.1_hiera_base_plus sam2.1_hiera_large; do
        download "${SAM2p1_BASE_URL}/${name}.pt"
    done
fi

if [ "$status" -eq 0 ]; then
    echo "All requested checkpoints are present."
else
    echo "Some checkpoints failed to download."
fi
exit "$status"
