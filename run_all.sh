#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh - unattended overnight run of the whole training pipeline.
#
#   Launch and walk away:
#       nohup bash run_all.sh > logs/overnight.log 2>&1 &
#       tail -f logs/overnight.log        # watch progress
#
# Every step is timed and logged. If one model fails the rest still run.
# Tune the three knobs below: smaller = faster. CPU-only, so keep them modest.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"

PY=/home/isha/miniconda3/envs/env-ann/bin/python   # env-ann (has cv2.data + TF)
PER_CLASS=2500              # images per class in build_dataset (CPU-friendly)
EPOCHS_FROZEN=6            # stage-1 epochs (early stopping may end sooner)
EPOCHS_FINETUNE=4          # stage-2 epochs
BUILD_DATASET=0            # dataset_final/ already built 2026-07-24 (per-class 2500); set 1 to rebuild

mkdir -p logs
step () { echo; echo "===== $(date '+%H:%M:%S')  $* ====="; }

echo "### overnight run started $(date) on $(uname -n)"
"$PY" -c "import tensorflow as tf; print('TF', tf.__version__)"

if [ "$BUILD_DATASET" -eq 1 ]; then
  step "build_dataset (--per-class $PER_CLASS)"
  time "$PY" build_dataset.py --per-class "$PER_CLASS"
fi

for M in mobilenetv2 mobilenetv3 baseline; do
  step "train $M"
  time "$PY" train_classifier.py --model "$M" \
        --epochs-frozen "$EPOCHS_FROZEN" --epochs-finetune "$EPOCHS_FINETUNE" \
        || echo "!!! $M training failed, continuing"
done

step "train_embedding (open-set)"
time "$PY" train_embedding.py || echo "!!! embedding failed, continuing"

for M in mobilenetv2 mobilenetv3 baseline; do
  step "evaluate $M"
  "$PY" evaluate.py --model "$M" || echo "!!! evaluate $M failed"
done
step "comparison table"
"$PY" evaluate.py --compare

step "quantize mobilenetv2 (float32 + INT8)"
time "$PY" quantize.py --model mobilenetv2 || echo "!!! quantize failed"

echo; echo "### overnight run finished $(date)"
echo "### results in results/ , models in models/ , this log in logs/overnight.log"
