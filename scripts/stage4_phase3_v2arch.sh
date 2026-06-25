#!/usr/bin/env bash
# Phase 3 v2arch: 5-domain joint with PInSoRo + bridge + ordinal
# Init from Phase 2 v2arch best.pt per seed
#
# Targets:
#   maintain regression CCC + pinsoro kappa ≥ 0.30

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME=${EXP_NAME:-phase3_v2arch_whisper}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v3}
FEATURE_STATS=${FEATURE_STATS:-experiments/_feature_stats/feature_stats_phase3_whisper.npz}
INIT_FROM=${INIT_FROM:-multimediate26/output/phase2_v2arch_whisper_seed${SEED}/best.pt}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-400}
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
    --train-manifests multimediate26/manifests/noxi_train.jsonl,multimediate26/manifests/noxi_j_train.jsonl,multimediate26/manifests/mpiigi_train.jsonl,multimediate26/manifests/pinsoro_cc_train.jsonl,multimediate26/manifests/pinsoro_cr_train.jsonl \
    --val-manifests   multimediate26/manifests/noxi_val.jsonl,multimediate26/manifests/noxi_j_val.jsonl,multimediate26/manifests/mpiigi_val_held.jsonl,multimediate26/manifests/pinsoro_cc_val.jsonl,multimediate26/manifests/pinsoro_cr_val.jsonl \
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
    --lr              2e-5 \
    --init-from       "$INIT_FROM" \
    --enable-bridge \
    --loss-weight bridge_ccc=0.3 \
    --loss-weight ordinal=0.1 \
    --use-layerwise-lr \
    --use-group-fusion \
    2>&1 | tee "$LOG"
