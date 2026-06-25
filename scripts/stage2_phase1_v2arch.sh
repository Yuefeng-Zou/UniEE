#!/usr/bin/env bash
# Phase 1 v2arch: NoXi + NoXi-J regression pre-train with:
#   - ModalityGroupFusion (replaces concat+Linear)
#   - HierarchicalDomainPrompt (coarse+fine)
#   - MultiPartnerPooling (attention-based, though 1 partner in Phase 1)
#   - Layer-wise LR from base.yaml lr_groups
#
# Targets:
#   noxi_val    CCC ≥ 0.855
#   noxi_j_val  CCC ≥ 0.680

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME=${EXP_NAME:-phase1_v2arch_whisper}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v3}
FEATURE_STATS=${FEATURE_STATS:-experiments/_feature_stats/feature_stats_phase1_whisper.npz}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-300}
WINDOW_LEN=${WINDOW_LEN:-512}
TRAIN_STRIDE=${TRAIN_STRIDE:-64}
NUM_WORKERS=${NUM_WORKERS:-2}
OUTPUT_DIR=multimediate26/output/${EXP_NAME}_seed${SEED}

mkdir -p "$OUTPUT_DIR"
LOG=${OUTPUT_DIR}/train.log

echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR"

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
    --use-layerwise-lr \
    --use-group-fusion \
    2>&1 | tee "$LOG"
