#!/usr/bin/env bash
# PInSoRo 专用训练：从 Phase2 best checkpoint 开始，只训练分类域
# 策略：
#   1. 冻结主干 (projector + group_fusion + dapa_layers)，只训分类头 + bridge + domain_prompt
#   2. 更强的正则化：更大 dropout, 更强 label_smoothing
#   3. CC 和 CR 联合训练（共享主干特征）
#   4. 类别权重平衡（应对 class imbalance）

set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-2}
GPU=${GPU:-2}
FEATURES="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip"
NPZ_ROOT=multimediate26/data_processed/npz_v4
FEATURE_STATS=experiments/_feature_stats/feature_stats_v4_whisper_full.npz

# Phase 2 best as init (regression backbone already trained)
INIT_FROM=${INIT_FROM:-multimediate26/output/phase2_v2arch_11feat_seed${SEED}/best.pt}
OUTPUT_DIR=multimediate26/output/pinsoro_specialist_seed${SEED}

mkdir -p "$OUTPUT_DIR"
LOG=${OUTPUT_DIR}/train.log

echo "============================================"
echo "  PInSoRo Specialist Training"
echo "  GPU=$GPU, SEED=$SEED"
echo "  Init from: $INIT_FROM"
echo "============================================"

# Only PInSoRo data for training, but keep regression domains in val
# to monitor CCC doesn't degrade too much
CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests multimediate26/manifests/pinsoro_cc_train.jsonl,multimediate26/manifests/pinsoro_cr_train.jsonl \
    --val-manifests   multimediate26/manifests/pinsoro_cc_val.jsonl,multimediate26/manifests/pinsoro_cr_val.jsonl \
    --npz-root        "$NPZ_ROOT" \
    --feature-stats   "$FEATURE_STATS" \
    --output-dir      "$OUTPUT_DIR" \
    --seed            "$SEED" \
    --epochs          30 \
    --steps-per-epoch 200 \
    --batch-size      32 \
    --window-len      512 \
    --train-stride    64 \
    --max-partners    3 \
    --num-workers     2 \
    --lr              5e-5 \
    --init-from       "$INIT_FROM" \
    --enable-bridge \
    --loss-weight bridge_ccc=0.3 \
    --use-layerwise-lr \
    --use-group-fusion \
    2>&1 | tee "$LOG"
