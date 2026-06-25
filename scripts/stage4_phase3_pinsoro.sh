#!/usr/bin/env bash
# Phase 3 — 5-domain joint training with bridge + ordinal contrastive.
#   * Adds PInSoRo cc + cr to the train set (classification → CE + bridge)
#   * Enables bridge_ccc loss (PInSoRo gets pseudo-cont supervision so the
#     regression head sees gradient on classification frames too)
#   * Enables ordinal contrastive (all 5 domains; pulls features of similar
#     engagement frames together regardless of which domain emitted them)
#   * Initialized from Phase 2 best.pt (same seed)
#
# Targets:
#   noxi/noxi_j/mpiigi unchanged from Phase 2 (and ideally a bit higher
#                       from ordinal regularization)
#   pinsoro_cc kappa  ≥ 0.30  (officially Combined Kappa = mean of 4 cols)
#   pinsoro_cr kappa  ≥ 0.30

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME=${EXP_NAME:-phase3_5domain_whisper}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v3}
FEATURE_STATS=${FEATURE_STATS:-/ossfs/workspace/mm26_stats/feature_stats_phase3_whisper.npz}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-400}    # +33% over Phase 2 — 2 more domains
WINDOW_LEN=${WINDOW_LEN:-512}
TRAIN_STRIDE=${TRAIN_STRIDE:-64}
LR=${LR:-2e-5}                              # even lower — backbone is hot
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_PARTNERS=${MAX_PARTNERS:-3}
INIT_FROM=${INIT_FROM:-multimediate26/output/phase2_whisper_noxi_noxij_mpiigi_seed${SEED}/best.pt}
OUTPUT_DIR=multimediate26/output/${EXP_NAME}_seed${SEED}
# RESUME=auto → use OUTPUT_DIR/last.pt if it exists. Empty (default) → init-from.
RESUME=${RESUME:-}

mkdir -p /ossfs/workspace/run_logs/${EXP_NAME}_seed${SEED}
LOG=/ossfs/workspace/run_logs/${EXP_NAME}_seed${SEED}/train.log

if [[ "$RESUME" == "auto" ]]; then
    RESUME="$OUTPUT_DIR/last.pt"
fi
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    INIT_ARG="--resume $RESUME"
    echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR (RESUME from $RESUME)"
else
    INIT_ARG="--init-from $INIT_FROM"
    echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR (INIT_FROM $INIT_FROM)"
fi
echo "  features=$FEATURES"
echo "  max_partners=$MAX_PARTNERS"
echo "  bridge_ccc=0.3 ordinal=0.1"
echo "  log → $LOG"

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
    --max-partners    "$MAX_PARTNERS" \
    --num-workers     "$NUM_WORKERS" \
    --lr              "$LR" \
    --enable-bridge \
    $INIT_ARG \
    --loss-weight     bridge_ccc=0.3 \
    --loss-weight     ordinal=0.1 \
    2>&1 | tee -a "$LOG"
