#!/usr/bin/env python
"""Serve the web UI and API.

Example:
    python scripts/run_web.py --port 8000
    # then open http://localhost:8000 (on a remote server: ssh -L 8000:localhost:8000 <host>)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    args = p.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("REDLIGHT_ADMIN_TOKEN"):
        p.error("Đặt REDLIGHT_ADMIN_TOKEN trước khi mở server ra ngoài localhost")
    setup_logging("INFO")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
