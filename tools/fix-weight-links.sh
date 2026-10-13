#!/usr/bin/env bash
# Repair an HF cache repo dir (hub/models--org--name) so a container can load it offline:
#  1. blobs that are symlinks into tt's shared hub/blobs store -> hardlinks (a container that mounts
#     only the repo dir sees real files; no extra disk)
#  2. a missing refs/main when exactly one snapshot exists (tt downloads pinned revisions without it,
#     and code calling snapshot_download(repo) offline then fails with LocalEntryNotFoundError)
set -euo pipefail
for d in "$@"; do
  [ -d "$d" ] || { echo "$d: missing"; continue; }
  n=0
  for b in "$d"/blobs/*; do
    [ -L "$b" ] || continue
    t=$(readlink -f "$b"); [ -f "$t" ] || { echo "dangling: $b"; continue; }
    ln -f "$t" "$b.tmp.$$" && mv -f "$b.tmp.$$" "$b"; n=$((n+1))
  done
  ref=""
  if [ ! -f "$d/refs/main" ]; then
    snaps=("$d"/snapshots/*)
    if [ ${#snaps[@]} -eq 1 ] && [ -d "${snaps[0]}" ]; then
      mkdir -p "$d/refs" && basename "${snaps[0]}" | tr -d '\n' > "$d/refs/main" && ref=" refs/main written"
    fi
  fi
  echo "$d: $n blob symlinks -> hardlinks$ref"
done
