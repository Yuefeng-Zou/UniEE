#!/usr/bin/env bash
# Phase 1 retrain with the fixed data pipeline:
#   * Full-duration NoXi (w2vbert2 sr fix → +60% frames per session)
#   * Whisper-large-v3 audio backbone (DAPA paper's choice, displacing W2vBERT2
#     as the dominant audio signal)
#   * Same backbone params, same eval protocol vs official baseline
#
# Targets:
#   noxi_val    CCC ≥ 0.855  (DAPA SOTA on val)
#   noxi_j_val  CCC ≥ 0.722  (DAPA SOTA on val)

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME=${EXP_NAME:-phase1_whisper_noxi_noxij}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v3}
FEATURE_STATS=${FEATURE_STATS:-/ossfs/workspace/mm26_stats/feature_stats_phase1_whisper.npz}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-300}    # +50% over prev: full-len NoXi has more windows
WINDOW_LEN=${WINDOW_LEN:-512}
TRAIN_STRIDE=${TRAIN_STRIDE:-64}
LR=${LR:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
OUTPUT_DIR=multimediate26/output/${EXP_NAME}_seed${SEED}

mkdir -p /ossfs/workspace/run_logs/${EXP_NAME}_seed${SEED}
LOG=/ossfs/workspace/run_logs/${EXP_NAME}_seed${SEED}/train.log

echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR"
echo "  features=$FEATURES"
echo "  log → $LOG"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests multimediate26/manifests/noxi_train.jsonl,multimediate26/manifests/noxi_j_train.jsonl \
    --val-manifests   multimediate26/manifests/noxi_val.jsonl,multimediate26/manifests/noxi_j_val.jsonl \
    --npz-root        "$NPZ_ROOT" \
    --feature-stats   "$FEATURE_STATS" \
    --output-dir      "$OUTPUT_DIR" \
    --seed            "$SEED" \
    --epochs          "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --batch-size      "$BATCH" \
    --window-len      "$WINDOW_LEN" \
    --train-stride    "$TRAIN_STRIDE" \
    --max-partners    1 \
    --num-workers     "$NUM_WORKERS" \
    --lr              "$LR" \
    2>&1 | tee "$LOG"
