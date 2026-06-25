#!/usr/bin/env bash
# Final Retrain A: 全量训练，无 val，固定 epoch 数
# Phase 3 保存 epoch 2 (CCC best) 和 epoch 7 (Kappa best)
# 回归域用 epoch_2.pt，分类域用 epoch_7.pt

set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-2}
GPU=${GPU:-0}
FEATURES="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip"
NPZ_ROOT=multimediate26/data_processed/npz_v4
FEATURE_STATS=experiments/_feature_stats/feature_stats_v4_whisper_full.npz
PREFIX=final_fulldata

# All data manifests (train+val merged, no held-out)
NOXI_ALL="multimediate26/manifests/full_noxi_train.jsonl,multimediate26/manifests/full_noxi_val.jsonl"
NOXIJ_ALL="multimediate26/manifests/full_noxi_j_train.jsonl,multimediate26/manifests/full_noxi_j_val.jsonl"
MPIIGI_ALL="multimediate26/manifests/full_mpiigi_train.jsonl,multimediate26/manifests/full_mpiigi_val.jsonl"
PINSORO_CC_ALL="multimediate26/manifests/full_pinsoro_cc_train.jsonl,multimediate26/manifests/full_pinsoro_cc_val.jsonl"
PINSORO_CR_ALL="multimediate26/manifests/full_pinsoro_cr_train.jsonl,multimediate26/manifests/full_pinsoro_cr_val.jsonl"

echo "============================================"
echo "  Final Retrain A: Full Data, Fixed Epochs"
echo "  GPU=$GPU, SEED=$SEED"
echo "============================================"

# ── Phase 1: 13 epochs ──
P1_DIR=multimediate26/output/${PREFIX}_p1_seed${SEED}
mkdir -p "$P1_DIR"
echo ">>> Phase 1: 13 epochs"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests ${NOXI_ALL},${NOXIJ_ALL} \
    --val-manifests   ${NOXI_ALL},${NOXIJ_ALL} \
    --npz-root "$NPZ_ROOT" --feature-stats "$FEATURE_STATS" \
    --output-dir "$P1_DIR" --seed "$SEED" \
    --epochs 13 --steps-per-epoch 350 --batch-size 32 \
    --window-len 512 --train-stride 64 --max-partners 1 --num-workers 2 \
    --use-layerwise-lr --use-group-fusion \
    2>&1 | tee "$P1_DIR/train.log"

# ── Phase 2: 9 epochs ──
P2_DIR=multimediate26/output/${PREFIX}_p2_seed${SEED}
mkdir -p "$P2_DIR"
echo ">>> Phase 2: 9 epochs"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests ${NOXI_ALL},${NOXIJ_ALL},${MPIIGI_ALL} \
    --val-manifests   ${NOXI_ALL},${NOXIJ_ALL},${MPIIGI_ALL} \
    --npz-root "$NPZ_ROOT" --feature-stats "$FEATURE_STATS" \
    --output-dir "$P2_DIR" --seed "$SEED" \
    --epochs 9 --steps-per-epoch 300 --batch-size 32 \
    --window-len 512 --train-stride 64 --max-partners 3 --num-workers 2 \
    --lr 3e-5 --init-from "$P1_DIR/last.pt" \
    --use-layerwise-lr --use-group-fusion \
    2>&1 | tee "$P2_DIR/train.log"

# ── Phase 3: 9 epochs, save epoch 2 and 7 ──
P3_DIR=multimediate26/output/${PREFIX}_p3_seed${SEED}
mkdir -p "$P3_DIR"
echo ">>> Phase 3: 9 epochs (save epoch 2 for CCC, epoch 7 for Kappa)"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests ${NOXI_ALL},${NOXIJ_ALL},${MPIIGI_ALL},${PINSORO_CC_ALL},${PINSORO_CR_ALL} \
    --val-manifests   ${NOXI_ALL},${NOXIJ_ALL},${MPIIGI_ALL},${PINSORO_CC_ALL},${PINSORO_CR_ALL} \
    --npz-root "$NPZ_ROOT" --feature-stats "$FEATURE_STATS" \
    --output-dir "$P3_DIR" --seed "$SEED" \
    --epochs 9 --steps-per-epoch 400 --batch-size 32 \
    --window-len 512 --train-stride 64 --max-partners 3 --num-workers 2 \
    --lr 2e-5 --init-from "$P2_DIR/last.pt" \
    --enable-bridge --loss-weight bridge_ccc=0.3 --loss-weight ordinal=0.1 \
    --use-layerwise-lr --use-group-fusion \
    --save-epochs 2,7 \
    2>&1 | tee "$P3_DIR/train.log"

echo "============================================"
echo "  Done! Checkpoints:"
echo "  Regression: $P3_DIR/epoch_2.pt"
echo "  Classification: $P3_DIR/epoch_7.pt"
echo "============================================"
