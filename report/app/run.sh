#!/bin/bash
cd "$(dirname "$0")"

PORT=8500
HOST="127.0.0.1"

echo "========================================================="
echo "  Đang khởi động LaTeX Studio (Local Overleaf)..."
echo "  URL: http://${HOST}:${PORT}"
echo "========================================================="

# Run using conda environment ml or default python3
if command -v conda &> /dev/null && conda env list | grep -q "ml"; then
    exec conda run --no-capture-output -n ml python app.py --host "$HOST" --port "$PORT"
else
    exec python3 app.py --host "$HOST" --port "$PORT"
fi
