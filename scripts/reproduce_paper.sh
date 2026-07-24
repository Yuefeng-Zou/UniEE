#!/usr/bin/env bash
# Run the three-phase UniEE paper curriculum for one random seed.

set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-0}
GPU=${GPU:-0}
FEATURES=${FEATURES:-"openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip"}
NPZ_ROOT=${NPZ_ROOT:-multimediate26/data_processed/npz_v4}
FEATURE_STATS=${FEATURE_STATS:-multimediate26/experiments/_feature_stats/feature_stats_v4_whisper_full.npz}

echo "UniEE paper curriculum: seed=$SEED gpu=$GPU"

SEED="$SEED" GPU="$GPU" \
FEATURES="$FEATURES" NPZ_ROOT="$NPZ_ROOT" FEATURE_STATS="$FEATURE_STATS" \
EXP_NAME=paper_phase1_11feat EPOCHS=40 \
bash multimediate26/scripts/stage2_phase1_v2arch.sh

SEED="$SEED" GPU="$GPU" \
FEATURES="$FEATURES" NPZ_ROOT="$NPZ_ROOT" FEATURE_STATS="$FEATURE_STATS" \
EXP_NAME=paper_phase2_11feat EPOCHS=30 \
INIT_FROM="multimediate26/output/paper_phase1_11feat_seed${SEED}/best.pt" \
bash multimediate26/scripts/stage3_phase2_v2arch.sh

SEED="$SEED" GPU="$GPU" \
FEATURES="$FEATURES" NPZ_ROOT="$NPZ_ROOT" FEATURE_STATS="$FEATURE_STATS" \
EXP_NAME=paper_phase3_11feat EPOCHS=20 \
INIT_FROM="multimediate26/output/paper_phase2_11feat_seed${SEED}/best.pt" \
bash multimediate26/scripts/stage4_phase3_v2arch.sh
