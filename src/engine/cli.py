"""Run a configured local pipeline: python -m src.engine.cli --config FILE --source VIDEO."""
from __future__ import annotations

import argparse
import json

from .config import build_session
from .sources import FrameSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-id", default="camera_1")
    parser.add_argument("--kind", choices=["image", "video", "usb", "rtsp"], default="video")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    overrides = {"device": args.device} if args.device else None
    with build_session(args.config, args.source_id, overrides=overrides) as pipeline:
        with FrameSource(args.source, args.source_id, args.kind, args.stride) as source:
            for frame in source:
                outputs, errors = pipeline.run(frame)
                if errors:
                    print(json.dumps([error.__dict__ for error in errors]))
                if args.max_frames and pipeline.metrics.frames >= args.max_frames:
                    break
            pipeline.metrics.dropped = source.dropped
        print(json.dumps(pipeline.metrics.snapshot(), indent=2))


if __name__ == "__main__":
    main()
