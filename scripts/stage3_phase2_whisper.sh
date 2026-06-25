#!/usr/bin/env bash
# Phase 2 joint training (whisper edition):
#   * NoXi + NoXi-J + MPIIGI continuous-only
#   * Initialized from Phase 1 whisper best.pt (3 seeds × seed-matched init)
#   * 9-feat preset (8 from Phase 1 + whisper) — same as Phase 1
#   * max_partners=3 (MPIIGI 4-person sessions)
#
# Targets: noxi ≥0.855, noxi_j ≥0.722, mpiigi ≥0.67 (all DAPA SOTA val).

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME=${EXP_NAME:-phase2_whisper_noxi_noxij_mpiigi}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v3}
FEATURE_STATS=${FEATURE_STATS:-/ossfs/workspace/mm26_stats/feature_stats_phase2_whisper.npz}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-300}
WINDOW_LEN=${WINDOW_LEN:-512}
TRAIN_STRIDE=${TRAIN_STRIDE:-64}
LR=${LR:-3e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_PARTNERS=${MAX_PARTNERS:-3}
OUTPUT_DIR=multimediate26/output/${EXP_NAME}_seed${SEED}
# RESUME=auto → use OUTPUT_DIR/last.pt if it exists. Empty (default) → init-from.
RESUME=${RESUME:-}
INIT_FROM=${INIT_FROM:-multimediate26/output/phase1_whisper_noxi_noxij_seed${SEED}/best.pt}

mkdir -p /ossfs/workspace/run_logs/${EXP_NAME}_seed${SEED}
LOG=/ossfs/workspace/run_logs/${EXP_NAME}_seed${SEED}/train.log

if [[ "$RESUME" == "auto" ]]; then
    RESUME="$OUTPUT_DIR/last.pt"
fi

# Build optional resume vs init-from arg (mutually exclusive in trainer).
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    INIT_ARG="--resume $RESUME"
    echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR (RESUME from $RESUME)"
else
    INIT_ARG="--init-from $INIT_FROM"
    echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR (INIT_FROM $INIT_FROM)"
fi
echo "  features=$FEATURES"
echo "  max_partners=$MAX_PARTNERS"
echo "  log → $LOG"

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
    --max-partners    "$MAX_PARTNERS" \
    --num-workers     "$NUM_WORKERS" \
    --lr              "$LR" \
    $INIT_ARG \
    2>&1 | tee -a "$LOG"
