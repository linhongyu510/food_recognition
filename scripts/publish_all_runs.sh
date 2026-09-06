#!/usr/bin/env bash
# Publish every benchmarked checkpoint to the Hugging Face Hub.
#
# Covers all nine cells of the resolution x architecture grid plus both
# Food-101 models, so every number in the README's benchmark tables has a
# downloadable checkpoint behind it.
#
# Usage:
#     hf auth login                       # needs a WRITE token
#     scripts/publish_all_runs.sh RUNS_ROOT [HF_USERNAME]
#
# RUNS_ROOT is the directory holding the bench/, abl/ and f101/ run trees.
# Runs are not in version control (see .gitignore), so their location has to be
# passed in rather than assumed.
#
# Add --dry-run after the username to render every card without uploading.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,16p' "$0"
    exit 2
fi

ROOT=${1%/}
USER=${2:-hylin16}
shift $(( $# > 2 ? 2 : $# ))
EXTRA=("$@")            # e.g. --dry-run

cd "$(dirname "$0")/.."
PREFIX=food-recognition
COMMIT=$(git rev-parse --short HEAD)

PUBLISHED=0
SKIPPED=0

publish () {            # publish <run-subpath> <repo-suffix> <note>
    local run="$ROOT/$1"
    if [[ ! -f "$run/checkpoints/best.pt" ]]; then
        echo "SKIP $2 — no checkpoint at $run" >&2
        SKIPPED=$(( SKIPPED + 1 ))
        return 0
    fi
    echo "=========== $2  ($3) ==========="
    python scripts/publish_to_hf.py \
        --run "$run" \
        --repo-id "$USER/$PREFIX-$2" \
        --commit "$COMMIT" \
        "${EXTRA[@]+"${EXTRA[@]}"}"
    PUBLISHED=$(( PUBLISHED + 1 ))
}

# --- Food-11: all nine grid cells, ordered by accuracy ------------------------
# The 224px B0 repo has no resolution suffix: it was published before the grid
# existed and is linked from elsewhere. It is the 224px cell.
publish abl/runs/abl_b3_380            food11-effnet-b3-cbam-380 "95.45% — best"
publish abl/runs/abl_b0_300            food11-effnet-b0-cbam-300 "95.15% — best per minute"
publish abl/runs/abl_b3_300            food11-effnet-b3-cbam-300 "95.15%"
publish bench/runs/b4_food11_380_full  food11-effnet-b4-cbam-380 "95.00%"
publish abl/runs/abl_b0_380            food11-effnet-b0-cbam-380 "94.85%"
publish abl/runs/abl_b4_300            food11-effnet-b4-cbam-300 "94.85%"
publish bench/runs/b4_food11_224_full  food11-effnet-b4-cbam-224 "94.24%"
publish bench/runs/bench_effnet_cbam   food11-effnet-b0-cbam     "93.64% — 224px"
publish abl/runs/abl_b3_224            food11-effnet-b3-cbam-224 "93.03%"

# --- Food-101: both cards default to license: other plus the dataset terms ---
publish f101/runs/b4_food101_224       food101-effnet-b4-cbam    "89.11%"
publish f101/runs/bench_food101        food101-effnet-b0-cbam    "88.70%"

echo
echo "Published $PUBLISHED of 11 (9 Food-11 grid cells + 2 Food-101 models)."
if [[ $SKIPPED -gt 0 ]]; then
    echo "Skipped $SKIPPED — check that RUNS_ROOT points at the run trees." >&2
    exit 1
fi
