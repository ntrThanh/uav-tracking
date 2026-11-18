#!/bin/bash
cd "$(dirname "$0")"

PORT=8000
HOST="127.0.0.1"

# Tìm python trong conda env ml hoặc python3
if [ -f "/home/trong-thanh/miniconda3/envs/ml/bin/python" ]; then
    PYTHON_CMD="/home/trong-thanh/miniconda3/envs/ml/bin/python"
elif command -v conda &> /dev/null && conda env list | grep -q "ml"; then
    PYTHON_CMD="conda run --no-capture-output -n ml python"
else
    PYTHON_CMD="python3"
fi

if [ "$1" == "video" ]; then
    echo "========================================================="
    echo "  Đang chạy pipeline phát hiện & bám vết trên video demo..."
    echo "========================================================="
    exec $PYTHON_CMD scripts/run_video.py --video data/samples/car-detection.mp4 --camera config/demo_street.json --out outputs/street_demo "${@:2}"
else
    echo "========================================================="
    echo "  Đang khởi động Giao diện Web Cấu hình Vượt Đèn Đỏ..."
    echo "  URL: http://${HOST}:${PORT}"
    echo "========================================================="
    exec $PYTHON_CMD scripts/run_web.py --host "$HOST" --port "$PORT"
fi
