#!/usr/bin/env bash
# Launch 4-way parallel Whisper feature extraction.
#
# Sharding: each GPU sees a strided subset of the *.audio.wav files under the
# full search root. The script's --shard IDX TOTAL flag does the actual filter.
#
# Total wavs ≈ 325; on H20 fp16 each averages ~7s → ~10 min serial per shard.

set -euo pipefail
cd "$(dirname "$0")/../.."   # → project root

WAV_ROOT=${WAV_ROOT:-/mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data}
MODEL=${MODEL:-/ossfs/workspace/models/whisper-large-v3}
GPUS=${GPUS:-4,5,6,7}
LOG_ROOT=/ossfs/workspace/run_logs/whisper_extract

IFS=',' read -ra GPU_ARR <<< "$GPUS"
TOTAL=${#GPU_ARR[@]}

mkdir -p "$LOG_ROOT"
for shard_idx in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$shard_idx]}"
    LOG="$LOG_ROOT/shard${shard_idx}_gpu${GPU}.log"
    echo "Launching shard $shard_idx/$TOTAL on cuda:$GPU → $LOG"
    setsid nohup \
      python -m multimediate26.data.feature_extractor.extract_whisper \
        --wav-root "$WAV_ROOT" \
        --model    "$MODEL" \
        --device   "cuda:$GPU" \
        --dtype    float16 \
        --shard    "$shard_idx" "$TOTAL" \
      > "$LOG" 2>&1 < /dev/null &
    echo "  pid=$!"
done
echo
echo "Started. Monitor with:"
echo "  tail -f $LOG_ROOT/shard*.log"
