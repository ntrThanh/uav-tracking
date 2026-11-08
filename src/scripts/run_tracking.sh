#!/usr/bin/env bash
# Tracking benchmark on the GPU server: detection caches, the 5 trackers on every detector, TrackEval tables.
#
#   tmux new -s track
#   bash src/scripts/run_tracking.sh cache val test-dev        # 3 detection caches + oracle (YOLO/RF-DETR on GPU 0, FRCNN on GPU 1)
#   bash src/scripts/run_tracking.sh track val                 # 4 detectors x 5 trackers, default parameters
#   bash src/scripts/run_tracking.sh eval val                  # results/tables/matrix_val_mean.csv
#   bash src/scripts/run_tracking.sh all val                   # cache + track + eval
#   bash src/scripts/run_tracking.sh tune                      # stage 2: det threshold grid on val, then select_thresh
#
# Environments: TRACK_PY (env track: trackers, YOLO, Faster R-CNN, TrackEval), RFDETR_PY (env rfdetr).
# Extra variables: DETS="yolo11m frcnn_r50" TRACKERS="bytetrack botsort" to run a subset.
set -euo pipefail

STEP=${1:?usage: run_tracking.sh <cache|track|tune|eval|all> <split...>}
shift
SPLITS=${*:-val}

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRACK_PY=${TRACK_PY:-$HOME/miniconda3/envs/track/bin/python}
RFDETR_PY=${RFDETR_PY:-$HOME/miniconda3/envs/rfdetr/bin/python}
DETS=${DETS:-"oracle yolo11m frcnn_r50 rfdetr_m"}
TRACKERS=${TRACKERS:-"sort deepsort bytetrack ocsort botsort"}
THRESHOLDS=${THRESHOLDS:-"0.3 0.4 0.5 0.6 0.7 0.8"}

cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1

cache() {
    # shellcheck disable=SC2086
    "$TRACK_PY" -m src.inference.cache_dets --det oracle --split $SPLITS > logs/cache_oracle.log 2>&1
    ( "$TRACK_PY" -m src.inference.cache_dets --det yolo11m --split $SPLITS --device cuda:0 > logs/cache_yolo11m.log 2>&1
      "$RFDETR_PY" -m src.inference.cache_dets --det rfdetr_m --split $SPLITS --device cuda:0 > logs/cache_rfdetr_m.log 2>&1 ) &
    "$TRACK_PY" -m src.inference.cache_dets --det frcnn_r50 --split $SPLITS --device cuda:1 > logs/cache_frcnn_r50.log 2>&1 &
    wait
    echo "[+] Caches done: $(grep -h 'Done' logs/cache_*.log | tr '\n' ' ')"
}

track() {
    for split in $SPLITS; do
        for det in $DETS; do
            (   # one detector per process group, the five trackers one after the other
                for tracker in $TRACKERS; do
                    "$TRACK_PY" -m src.inference.run_tracker --det "$det" --split "$split" --tracker "$tracker" \
                        > "logs/track_${det}_${tracker}_${split}.log" 2>&1
                done
            ) &
        done
        wait
        echo "[+] Tracking done on $split"
    done
}

tune() {
    # Stage 2 on val only: every detector x tracker at each high threshold, folders <tracker>_thrXX
    for det in ${DETS/oracle/}; do
        (
            for thr in $THRESHOLDS; do
                tag="thr$(echo "$thr" | tr -d '.')"
                for tracker in $TRACKERS; do
                    "$TRACK_PY" -m src.inference.run_tracker --det "$det" --split val --tracker "$tracker" \
                        --det-thresh "$thr" --tag "$tag" > "logs/tune_${det}_${tracker}_${tag}.log" 2>&1
                done
            done
        ) &
    done
    wait
    "$TRACK_PY" -m src.evaluation.eval_track --split val 2>&1 | grep -v BURST | tee logs/eval_track_val.log
    "$TRACK_PY" -m src.evaluation.select_thresh | tee logs/select_thresh.log
}

evaluate() {
    for split in $SPLITS; do
        "$TRACK_PY" -m src.evaluation.eval_track --split "$split" 2>&1 | grep -v BURST | tee "logs/eval_track_${split}.log"
    done
}

case "$STEP" in
    cache) cache ;;
    track) track ;;
    eval) evaluate ;;
    tune) tune ;;
    all) cache; track; evaluate ;;
    *) echo "unknown step $STEP" >&2; exit 1 ;;
esac
