#!/usr/bin/env bash
# UniEE Phase 2: add MPIIGI and activate multi-partner pooling.

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME=${EXP_NAME:-paper_phase2_11feat}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v4}
FEATURE_STATS=${FEATURE_STATS:-multimediate26/experiments/_feature_stats/feature_stats_v4_whisper_full.npz}
INIT_FROM=${INIT_FROM:-multimediate26/output/paper_phase1_11feat_seed${SEED}/best.pt}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-300}
WINDOW_LEN=${WINDOW_LEN:-512}
TRAIN_STRIDE=${TRAIN_STRIDE:-64}
NUM_WORKERS=${NUM_WORKERS:-2}
OUTPUT_DIR=multimediate26/output/${EXP_NAME}_seed${SEED}

mkdir -p "$OUTPUT_DIR"
LOG=${OUTPUT_DIR}/train.log

echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR"
echo "  init-from=$INIT_FROM"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests multimediate26/manifests/noxi_train.jsonl,multimediate26/manifests/noxi_j_train.jsonl,multimediate26/manifests/mpiigi_train.jsonl \
    --val-manifests   multimediate26/manifests/noxi_val.jsonl,multimediate26/manifests/noxi_j_val.jsonl,multimediate26/manifests/mpiigi_val_held.jsonl \
    --npz-root        "$NPZ_ROOT" \
    --feature-stats   "$FEATURE_STATS" \
    --output-dir      "$OUTPUT_DIR" \
    --seed            "$SEED" \
    --epochs          "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --batch-size      "$BATCH" \
    --window-len      "$WINDOW_LEN" \
    --train-stride    "$TRAIN_STRIDE" \
    --max-partners    3 \
    --num-workers     "$NUM_WORKERS" \
    --init-from       "$INIT_FROM" \
    --use-layerwise-lr \
    --use-group-fusion \
    2>&1 | tee "$LOG"
