#!/usr/bin/env bash
# Train RF-DETR inside tmux on the GPU server.
#
#   tmux new -s rfdetr
#   bash src/scripts/train_rfdetr.sh src/configs/detectors/rfdetr_m_576.yaml 0,1   # both GPUs (DDP)
#   bash src/scripts/train_rfdetr.sh src/configs/detectors/rfdetr_m_576.yaml 0     # single GPU
#   (detach: Ctrl+b then d, re-attach: tmux attach -t rfdetr)
#
# Args: <config.yaml> [gpus=0,1] [extra args for train_rfdetr.py, e.g. --dry-run, --set batch_size=4]
set -euo pipefail

CONFIG=${1:?usage: train_rfdetr.sh <config.yaml> [gpu_id] [extra args]}
GPU=${2:-0,1}
shift $(( $# >= 2 ? 2 : 1 ))

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-$HOME/miniconda3/envs/rfdetr/bin/python}
RUN=$(basename "$CONFIG" .yaml)
LOG="$ROOT/logs/${RUN}_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT"
mkdir -p logs
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

echo "[+] Config: $CONFIG"
echo "    [-] GPU: $GPU"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/        [-] /'
echo "    [-] Log: $LOG"
"$PYTHON" -m src.training.train_rfdetr --config "$CONFIG" "$@" 2>&1 | tee "$LOG"
