#!/usr/bin/env bash
# Final Retrain B: 留小部分 val 做 best ckp 选取
# 用 full_*_train (大部分数据训练) + full_*_val (每域4行验证)
# trainer 自动保存 best.pt (CCC最优) + best_kappa.pt (Kappa最优)

set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-2}
GPU=${GPU:-3}
FEATURES="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip"
NPZ_ROOT=multimediate26/data_processed/npz_v4
FEATURE_STATS=experiments/_feature_stats/feature_stats_v4_whisper_full.npz
PREFIX=final_retrain_11feat

echo "============================================"
echo "  Final Retrain B: Small Val, Best Ckp"
echo "  GPU=$GPU, SEED=$SEED"
echo "============================================"

# ── Phase 1 ──
P1_DIR=multimediate26/output/${PREFIX}_p1_seed${SEED}
mkdir -p "$P1_DIR"
echo ">>> Phase 1: 15 epochs"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests multimediate26/manifests/full_noxi_train.jsonl,multimediate26/manifests/full_noxi_j_train.jsonl \
    --val-manifests   multimediate26/manifests/full_noxi_val.jsonl,multimediate26/manifests/full_noxi_j_val.jsonl \
    --npz-root "$NPZ_ROOT" --feature-stats "$FEATURE_STATS" \
    --output-dir "$P1_DIR" --seed "$SEED" \
    --epochs 15 --steps-per-epoch 350 --batch-size 32 \
    --window-len 512 --train-stride 64 --max-partners 1 --num-workers 2 \
    --use-layerwise-lr --use-group-fusion \
    2>&1 | tee "$P1_DIR/train.log"

# ── Phase 2 ──
P2_DIR=multimediate26/output/${PREFIX}_p2_seed${SEED}
mkdir -p "$P2_DIR"
echo ">>> Phase 2: 12 epochs"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests multimediate26/manifests/full_noxi_train.jsonl,multimediate26/manifests/full_noxi_j_train.jsonl,multimediate26/manifests/full_mpiigi_train.jsonl \
    --val-manifests   multimediate26/manifests/full_noxi_val.jsonl,multimediate26/manifests/full_noxi_j_val.jsonl,multimediate26/manifests/full_mpiigi_val.jsonl \
    --npz-root "$NPZ_ROOT" --feature-stats "$FEATURE_STATS" \
    --output-dir "$P2_DIR" --seed "$SEED" \
    --epochs 12 --steps-per-epoch 300 --batch-size 32 \
    --window-len 512 --train-stride 64 --max-partners 3 --num-workers 2 \
    --lr 3e-5 --init-from "$P1_DIR/best.pt" \
    --use-layerwise-lr --use-group-fusion \
    2>&1 | tee "$P2_DIR/train.log"

# ── Phase 3 ──
P3_DIR=multimediate26/output/${PREFIX}_p3_seed${SEED}
mkdir -p "$P3_DIR"
echo ">>> Phase 3: 15 epochs (dual: best.pt + best_kappa.pt)"

CUDA_VISIBLE_DEVICES=$GPU python -m multimediate26.train.trainer \
    --features "$FEATURES" \
    --train-manifests multimediate26/manifests/full_noxi_train.jsonl,multimediate26/manifests/full_noxi_j_train.jsonl,multimediate26/manifests/full_mpiigi_train.jsonl,multimediate26/manifests/full_pinsoro_cc_train.jsonl,multimediate26/manifests/full_pinsoro_cr_train.jsonl \
    --val-manifests   multimediate26/manifests/full_noxi_val.jsonl,multimediate26/manifests/full_noxi_j_val.jsonl,multimediate26/manifests/full_mpiigi_val.jsonl,multimediate26/manifests/full_pinsoro_cc_val.jsonl,multimediate26/manifests/full_pinsoro_cr_val.jsonl \
    --npz-root "$NPZ_ROOT" --feature-stats "$FEATURE_STATS" \
    --output-dir "$P3_DIR" --seed "$SEED" \
    --epochs 15 --steps-per-epoch 400 --batch-size 32 \
    --window-len 512 --train-stride 64 --max-partners 3 --num-workers 2 \
    --lr 2e-5 --init-from "$P2_DIR/best.pt" \
    --enable-bridge --loss-weight bridge_ccc=0.3 --loss-weight ordinal=0.1 \
    --use-layerwise-lr --use-group-fusion \
    2>&1 | tee "$P3_DIR/train.log"

echo "============================================"
echo "  Done! Checkpoints:"
echo "  Regression: $P3_DIR/best.pt (CCC best)"
echo "  Classification: $P3_DIR/best_kappa.pt (Kappa best)"
echo "============================================"
