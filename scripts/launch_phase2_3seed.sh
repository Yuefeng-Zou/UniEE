#!/usr/bin/env bash
# Launch Phase 2 training across 3 seeds on distinct GPUs in parallel.
# Each seed runs in its own setsid-detached process so the launcher can
# return immediately and the trainings survive shell exit.
#
# Usage:
#   bash scripts/launch_phase2_3seed.sh           # GPU 0,1,3 (skip GPU 2 watchdog)
#   GPUS=4,5,6 bash scripts/launch_phase2_3seed.sh

set -euo pipefail
cd "$(dirname "$0")/../.."   # → project root

GPUS=${GPUS:-0,1,3}
SEEDS=${SEEDS:-0,1,2}

IFS=',' read -ra GPU_ARR  <<< "$GPUS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
if [ ${#GPU_ARR[@]} -ne ${#SEED_ARR[@]} ]; then
    echo "GPUS (${#GPU_ARR[@]}) and SEEDS (${#SEED_ARR[@]}) length mismatch" >&2
    exit 1
fi

for i in "${!SEED_ARR[@]}"; do
    SEED="${SEED_ARR[$i]}"
    GPU="${GPU_ARR[$i]}"
    EXP=phase2_noxi_noxij_mpiigi
    LOGDIR=/ossfs/workspace/run_logs/${EXP}_seed${SEED}
    mkdir -p "$LOGDIR"
    echo "Launching seed=$SEED on GPU=$GPU → $LOGDIR/nohup.log"
    SEED=$SEED GPU=$GPU EXP_NAME=$EXP \
      setsid nohup bash multimediate26/scripts/stage3_phase2_joint.sh \
      > "$LOGDIR/nohup.log" 2>&1 < /dev/null &
    echo "  pid=$!"
done
echo
echo "All 3 launched. Tail logs:"
echo "  tail -f /ossfs/workspace/run_logs/phase2_noxi_noxij_mpiigi_seed{0,1,2}/train.log"
