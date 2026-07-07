#!/usr/bin/env bash
# E4 — positive transfer under low resource (the revision's key claim).
#
# For each IMDB training fraction, compares:
#   alone: the low-resource domain trained by itself
#   joint: the same subset (paired per seed) trained WITH mnist+speechcommands
#
# The claim to test: joint > alone on the TEXT column as the fraction shrinks.
# Read the `text` per-domain column of the paired CSVs; significance by
# paired t-test over seeds (subsets are seed-matched across arms).
#
# Full-resource (1.0) is the control: paper's E1 parity result.
set -e
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-30}
VERSION=${VERSION:-gaau}

for frac in 0.01 0.1 1.0; do
  echo "=== E4 imdb@${frac} alone ==="
  python experiments/run.py --dataset "imdb@${frac}" \
    --version "$VERSION" --epochs "$EPOCHS" \
    --out "results/e4_${VERSION}_imdb${frac}_alone.csv"
  echo "=== E4 imdb@${frac} joint (mnist+sc) ==="
  python experiments/run.py --dataset "mnist,speechcommands,imdb@${frac}" \
    --version "$VERSION" --epochs "$EPOCHS" \
    --out "results/e4_${VERSION}_imdb${frac}_joint.csv"
done
