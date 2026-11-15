#!/usr/bin/env python
"""Stage 1 CLI: detect and track vehicles in one video file.

Writes into the output directory:
    annotated.mp4  boxes, track ids, reference points and trails
    tracks.csv     one row per (frame, track)
    summary.json   timing, FPS and track-length statistics
    ref.jpg        reference frame (at --start) for drawing geometry later

Example:
    python scripts/run_video.py --video data/samples/car-detection.mp4 --out outputs/car_demo
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import CameraConfig, PipelineConfig, resolve_path  # noqa: E402
from src.common.logging_utils import setup_logging  # noqa: E402
from src.detect.vehicle import VehicleDetector  # noqa: E402
from src.ingest.video import VideoReader, save_reference_frame  # noqa: E402
from src.track.tracker import VehicleTracker  # noqa: E402
from src.viz.draw import draw_camera_geometry, draw_hud, draw_tracks  # noqa: E402

log = logging.getLogger("run_video")

CSV_FIELDS = ["frame", "time_s", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "px", "py"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input video file")
    p.add_argument("--pipeline", default="config/pipeline.json", help="pipeline config JSON")
    p.add_argument("--camera", default=None, help="camera config JSON to overlay (optional)")
    p.add_argument("--out", default=None, help="output directory (default: outputs/<video stem>)")
    p.add_argument("--start", type=float, default=None, help="start time in seconds (overrides config)")
    p.add_argument("--end", type=float, default=None, help="end time in seconds, exclusive (overrides config)")
    p.add_argument("--stride", type=int, default=None, help="process every n-th frame (overrides config)")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    pipe = PipelineConfig.load(args.pipeline)
    cam = CameraConfig.load(args.camera) if args.camera else None
    video_path = resolve_path(args.video)
    start_s = pipe.video.start_s if args.start is None else args.start
    end_s = pipe.video.end_s if args.end is None else args.end
    stride = pipe.video.stride if args.stride is None else args.stride
    write_video = pipe.output.write_video and not args.no_video
    out_dir = resolve_path(args.out or f"outputs/{video_path.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = VehicleDetector(pipe.detector)
    tracker = VehicleTracker(pipe.tracker)
    rows: list[dict] = []
    writer: cv2.VideoWriter | None = None
    t_detect = t_track = 0.0
    n_frames = 0

    with VideoReader(video_path, start_s=start_s, end_s=end_s, stride=stride) as reader:
        info = reader.info
        log.info(
            "Video %s: %dx%d @ %.2f fps, %d frames (%.1f s); window %.1f s -> %s, stride %d (~%s frames)",
            video_path.name, info.width, info.height, info.fps, info.frame_count, info.duration_s,
            start_s, "end" if end_s is None else f"{end_s:.1f} s", stride, reader.expected_frames,
        )
        ref_path = save_reference_frame(video_path, out_dir / "ref.jpg", time_s=start_s)
        log.info("Reference frame saved to %s", ref_path)
        if write_video:
            video_out = out_dir / "annotated.mp4"
            writer = cv2.VideoWriter(
                str(video_out), cv2.VideoWriter_fourcc(*pipe.output.codec), info.fps / stride, (info.width, info.height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer for {video_out} (codec {pipe.output.codec})")

        t0 = time.perf_counter()
        for frame in reader:
            ta = time.perf_counter()
            dets = detector.detect(frame.image)
            tb = time.perf_counter()
            tracks = tracker.update(dets, frame.index, frame.time_s)
            tc = time.perf_counter()
            t_detect += tb - ta
            t_track += tc - tb
            n_frames += 1

            for tr in tracks:
                px, py = tr.ref_point
                x1, y1, x2, y2 = (round(float(v), 1) for v in tr.xyxy)
                rows.append({
                    "frame": frame.index, "time_s": round(frame.time_s, 3), "track_id": tr.track_id,
                    "class": tr.class_name, "conf": round(tr.conf, 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "px": round(px, 1), "py": round(py, 1),
                })

            if writer is not None:
                img = frame.image.copy()
                fps_now = n_frames / max(time.perf_counter() - t0, 1e-6)
                draw_hud(img, [
                    f"frame {frame.index}  t={frame.time_s:6.2f}s",
                    f"dets {len(dets)}  tracks {len(tracks)}",
                    f"{fps_now:.1f} FPS pipeline",
                ])
                if cam is not None:
                    draw_camera_geometry(img, cam)
                draw_tracks(img, tracks, tracker, trail_len=pipe.output.trail_len)
                writer.write(img)

            if n_frames % pipe.output.log_every == 0:
                fps_now = n_frames / max(time.perf_counter() - t0, 1e-6)
                log.info("frame %d (t=%.1fs): %d dets, %d tracks, %.1f FPS",
                         frame.index, frame.time_s, len(dets), len(tracks), fps_now)
        total = time.perf_counter() - t0

    if writer is not None:
        writer.release()
    if n_frames == 0:
        log.error("No frames processed (check --start/--end and the video file)")
        return 1

    csv_path = out_dir / "tracks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    stats = tracker.stats()
    summary = {
        "video": str(video_path),
        "width": info.width, "height": info.height, "video_fps": info.fps,
        "start_s": start_s, "end_s": end_s, "stride": stride, "frames_processed": n_frames,
        "detector": {"weights": pipe.detector.weights, "device": detector.device,
                     "imgsz": pipe.detector.imgsz, "conf": pipe.detector.conf},
        "timing": {
            "total_s": round(total, 2), "detect_s": round(t_detect, 2), "track_s": round(t_track, 2),
            "fps_pipeline": round(n_frames / total, 2),
            "fps_detect_only": round(n_frames / t_detect, 2) if t_detect > 0 else None,
        },
        "tracks": stats,
        "outputs": {
            "annotated_video": str(out_dir / "annotated.mp4") if writer is not None else None,
            "tracks_csv": str(csv_path), "reference_frame": str(ref_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "Done: %d frames in %.1f s (%.1f FPS pipeline, %.1f FPS detect-only, device %s)",
        n_frames, total, summary["timing"]["fps_pipeline"], summary["timing"]["fps_detect_only"] or 0.0, detector.device,
    )
    log.info("Tracks: %s", json.dumps(stats, ensure_ascii=False))
    log.info("Outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
