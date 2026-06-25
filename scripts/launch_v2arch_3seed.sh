#!/usr/bin/env bash
# Launch 3-seed v2arch Phase 1 training in parallel on GPUs 0,1,2
# Usage: bash scripts/launch_v2arch_3seed.sh [phase1|phase2|phase3]

set -euo pipefail
cd "$(dirname "$0")/../.."

PHASE=${1:-phase1}

case $PHASE in
    phase1)
        SCRIPT=scripts/stage2_phase1_v2arch.sh
        EXP_PREFIX=phase1_v2arch_whisper
        ;;
    phase2)
        SCRIPT=scripts/stage3_phase2_v2arch.sh
        EXP_PREFIX=phase2_v2arch_whisper
        ;;
    phase3)
        SCRIPT=scripts/stage4_phase3_v2arch.sh
        EXP_PREFIX=phase3_v2arch_whisper
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|phase3]"
        exit 1
        ;;
esac

echo "Launching 3-seed $PHASE on GPUs 0,1,2..."

for SEED in 0 1 2; do
    GPU=$SEED
    echo "  seed=$SEED → GPU=$GPU"
    setsid bash -c "GPU=$GPU SEED=$SEED EXP_NAME=$EXP_PREFIX bash $SCRIPT" \
        </dev/null &
done

echo "All 3 seeds launched. Check logs in multimediate26/output/${EXP_PREFIX}_seed{0,1,2}/train.log"
