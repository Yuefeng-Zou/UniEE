#!/usr/bin/env bash
# Phase 2 joint training: NoXi + NoXi-J + MPIIGI continuous-only.
# Per TechPlan v3 §6.2 — adds the MPIIGI val sessions (as quasi-train) so the
# 3-of-4 leaderboard regression domains all receive direct supervision. PInSoRo
# is held for Phase 3 (needs bridge + ordinal contrastive).
#
# Targets (must beat last year's SOTA):
#   noxi      val CCC ≥ 0.855  (DAPA 2025)
#   noxi_j    val CCC ≥ 0.722  (DAPA 2025)
#   mpiigi    val CCC ≥ 0.670  (DAPA 2025 test, baseline 0.4539)
#   combined  val CCC ≥ 0.735

set -euo pipefail
cd "$(dirname "$0")/../.."   # → project root

EXP_NAME=${EXP_NAME:-phase2_noxi_noxij_mpiigi}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,xlmr,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v3}
FEATURE_STATS=${FEATURE_STATS:-/ossfs/workspace/mm26_stats/feature_stats_phase2.npz}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-300}    # +50% over Phase 1 — more domain mix
WINDOW_LEN=${WINDOW_LEN:-512}
TRAIN_STRIDE=${TRAIN_STRIDE:-64}            # 8× overlap
LR=${LR:-3e-5}                              # lower than Phase 1 5e-5: warm start
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_PARTNERS=${MAX_PARTNERS:-3}             # mpiigi 4-person → 3 partners
INIT_FROM=${INIT_FROM:-multimediate26/output/phase1_noxi_noxij_seed2/best.pt}
OUTPUT_DIR=multimediate26/output/${EXP_NAME}_seed${SEED}

mkdir -p multimediate26/run_logs/${EXP_NAME}_seed${SEED}
LOG=multimediate26/run_logs/${EXP_NAME}_seed${SEED}/train.log

echo "GPU=$GPU EXP=$EXP_NAME SEED=$SEED → $OUTPUT_DIR"
echo "  features=$FEATURES"
echo "  max_partners=$MAX_PARTNERS  init_from=$INIT_FROM"
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
    --init-from       "$INIT_FROM" \
    2>&1 | tee "$LOG"
