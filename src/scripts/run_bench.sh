#!/usr/bin/env bash
# Speed benchmark (src/evaluation/bench_fps.py) on the GPU server, every job one after the other on GPU 0.
# Start it only when no other job uses the GPUs or the CPU, otherwise the numbers are noisy again.
#
#   tmux new -s bench
#   bash src/scripts/run_bench.sh            # ~1.5 h: 3 detectors, then the tracker runs of each detector, then merge
#
# Environments: TRACK_PY (env track), RFDETR_PY (env rfdetr).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRACK_PY=${TRACK_PY:-$HOME/miniconda3/envs/track/bin/python}
RFDETR_PY=${RFDETR_PY:-$HOME/miniconda3/envs/rfdetr/bin/python}
DETS=${DETS:-"yolo11m frcnn_r50 rfdetr_m"}

cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -gt 0 ]; then
    echo "[!] $busy process(es) already use a GPU; stop them first" >&2
    exit 1
fi

for det in $DETS; do
    py=$TRACK_PY
    [ "$det" = rfdetr_m ] && py=$RFDETR_PY
    "$py" -m src.evaluation.bench_fps det --det "$det" > "logs/bench_det_${det}.log" 2>&1
    echo "[+] $(tail -n 1 "logs/bench_det_${det}.log")"
done
for det in $DETS; do
    "$TRACK_PY" -m src.evaluation.bench_fps track --det "$det" > "logs/bench_track_${det}.log" 2>&1
    echo "[+] $(tail -n 1 "logs/bench_track_${det}.log")"
done
"$TRACK_PY" -m src.evaluation.bench_fps merge 2>&1 | tee logs/bench_merge.log
echo DONE
