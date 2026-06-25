#!/usr/bin/env bash
# Phase 1 pre-training: NoXi + NoXi-J continuous-only.
# Per TechPlan v3 §6.2 — Phase 1 trains the MD-DAPA backbone on the two
# best-labelled regression-only domains. PInSoRo is held for Phase 2.

set -euo pipefail
cd "$(dirname "$0")/../.."   # → project root

EXP_NAME=${EXP_NAME:-phase1_noxi_noxij}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,xlmr,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v3}
FEATURE_STATS=${FEATURE_STATS:-multimediate26/data_processed/npz_v3/__feature_stats.npz}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-200}
WINDOW_LEN=${WINDOW_LEN:-512}
TRAIN_STRIDE=${TRAIN_STRIDE:-64}     # 8x overlap (USTC-IAT'25 winner)
LR=${LR:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
RESUME=${RESUME:-}    # optional: path to last.pt to resume from
OUTPUT_DIR=multimediate26/output/${EXP_NAME}_seed${SEED}

mkdir -p multimediate26/run_logs/${EXP_NAME}_seed${SEED}
LOG=multimediate26/run_logs/${EXP_NAME}_seed${SEED}/train.log

echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR"
echo "  features=$FEATURES"
echo "  log → $LOG"
echo "  num_workers=$NUM_WORKERS  resume=${RESUME:-<none>}"

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
    RESUME_ARGS=(--resume "$RESUME")
fi

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
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "$LOG"
