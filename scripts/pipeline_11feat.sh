#!/usr/bin/env bash
# 11-feat pipeline: Phase 2 bis + Phase 3 bis
# Run AFTER Phase 1 bis (11-feat) completes.
#
# Usage:
#   GPU=0 SEED=0 bash multimediate26/scripts/pipeline_11feat.sh phase2
#   GPU=0 SEED=0 bash multimediate26/scripts/pipeline_11feat.sh phase3

set -euo pipefail
cd "$(dirname "$0")/../.."

PHASE=${1:?usage: pipeline_11feat.sh phase2|phase3}
SEED=${SEED:-0}
GPU=${GPU:-0}
FEAT="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip"
NPZ=multimediate26/data_processed/npz_v4
STATS=/ossfs/workspace/mm26_stats/feature_stats_v4_whisper_full.npz

case "$PHASE" in
    phase2)
        EXP_NAME=phase2_11feat \
        FEATURES="$FEAT" NPZ_ROOT="$NPZ" FEATURE_STATS="$STATS" \
        INIT_FROM=multimediate26/output/phase1_11feat_seed${SEED}/best.pt \
        SEED=$SEED GPU=$GPU EPOCHS=30 LR=3e-5 MAX_PARTNERS=3 \
        bash multimediate26/scripts/stage3_phase2_whisper.sh
        ;;
    phase3)
        EXP_NAME=phase3_11feat \
        FEATURES="$FEAT" NPZ_ROOT="$NPZ" FEATURE_STATS="$STATS" \
        INIT_FROM=multimediate26/output/phase2_11feat_seed${SEED}/best.pt \
        SEED=$SEED GPU=$GPU EPOCHS=30 LR=2e-5 MAX_PARTNERS=3 \
        bash multimediate26/scripts/stage4_phase3_pinsoro.sh
        ;;
    *)
        echo "unknown phase: $PHASE"; exit 1 ;;
esac
