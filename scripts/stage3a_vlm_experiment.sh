#!/usr/bin/env bash
# Phase 3a: VLM (Qwen3-VL-Embedding) experiment
# 12-feat = 11-feat + qwen3vl_emb
# Init from Phase 2 best.pt (not Phase 3, because PInSoRo has no VLM features)
# Only trains on 3 regression domains (noxi + noxi_j + mpiigi) where VLM exists
#
# Judgment: if NoXi-Add val CCC improves ≥ 0.005 over Phase 2, keep VLM

set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-2}
GPU=${GPU:-0}
INIT_FROM=${INIT_FROM:-multimediate26/output/phase2_v2arch_11feat_seed${SEED}/best.pt}
OUTPUT_DIR=multimediate26/output/phase3a_vlm_12feat_seed${SEED}
FEATURE_STATS=experiments/_feature_stats/feature_stats_12feat_vlm.npz
FEATURES="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip,qwen3vl_emb"

mkdir -p "$OUTPUT_DIR"
LOG=${OUTPUT_DIR}/train.log

echo "GPU=$GPU SEED=$SEED → $OUTPUT_DIR"
echo "  features: 12-feat (11 + qwen3vl_emb)"
echo "  init-from: $INIT_FROM"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests multimediate26/manifests/noxi_train.jsonl,multimediate26/manifests/noxi_j_train.jsonl,multimediate26/manifests/mpiigi_train.jsonl \
    --val-manifests   multimediate26/manifests/noxi_val.jsonl,multimediate26/manifests/noxi_j_val.jsonl,multimediate26/manifests/mpiigi_val_held.jsonl \
    --npz-root        multimediate26/data_processed/npz_v4 \
    --feature-stats   "$FEATURE_STATS" \
    --output-dir      "$OUTPUT_DIR" \
    --seed            "$SEED" \
    --epochs          20 \
    --steps-per-epoch 300 \
    --batch-size      32 \
    --window-len      512 \
    --train-stride    64 \
    --max-partners    3 \
    --num-workers     2 \
    --lr              3e-5 \
    --init-from       "$INIT_FROM" \
    --use-layerwise-lr \
    --use-group-fusion \
    2>&1 | tee "$LOG"
