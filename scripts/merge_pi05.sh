#!/usr/bin/env bash
set -e
cd /data/rpent/checkpoints
N=$(ls pi05_parts | wc -l)
echo "parts: $N"
cat pi05_parts/p* > pi05-libero-130/model.safetensors
SZ=$(stat -c%s pi05-libero-130/model.safetensors)
echo "merged size: $SZ (expect 7473091464)"
[ "$SZ" -eq 7473091464 ] && echo MERGE-OK || echo MERGE-SIZE-MISMATCH
