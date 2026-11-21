#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
python_bin="${TRACKING_PYTHON:-$project_dir/.venv-tracking/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi
"$python_bin" -m tracking_app.check
exec "$python_bin" -m tracking_app
