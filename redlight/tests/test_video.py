import cv2
import numpy as np
import pytest

from src.ingest.video import VideoReader, extract_reference_frame, probe_video

FPS, N, W, H = 10.0, 30, 64, 48


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """30 frames at 10 fps; frame i has blue channel ~ 8*i so it can be identified after decoding."""
    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened(), "OpenCV build cannot write mp4v video"
    for i in range(N):
        img = np.zeros((H, W, 3), np.uint8)
        img[:, :, 0] = 8 * i
        writer.write(img)
    writer.release()
    return path


def _blue_index(img: np.ndarray) -> float:
    return float(img[:, :, 0].mean()) / 8.0


def test_probe(synthetic_video):
    info = probe_video(synthetic_video)
    assert (info.width, info.height, info.frame_count) == (W, H, N)
    assert info.fps == pytest.approx(FPS)
    assert info.duration_s == pytest.approx(3.0)


def test_iterate_all_frames(synthetic_video):
    with VideoReader(synthetic_video) as reader:
        assert reader.expected_frames == N
        frames = list(reader)
    assert [f.index for f in frames] == list(range(N))
    assert frames[7].time_s == pytest.approx(0.7)
    assert frames[0].image.shape == (H, W, 3)
    assert _blue_index(frames[20].image) == pytest.approx(20, abs=1.0)


def test_window_and_stride(synthetic_video):
    with VideoReader(synthetic_video, start_s=1.0, end_s=2.0, stride=2) as reader:
        assert reader.expected_frames == 5
        frames = list(reader)
    assert [f.index for f in frames] == [10, 12, 14, 16, 18]
    assert _blue_index(frames[0].image) == pytest.approx(10, abs=1.0)


def test_reference_frame(synthetic_video):
    img = extract_reference_frame(synthetic_video, time_s=1.5)
    assert img.shape == (H, W, 3)
    assert _blue_index(img) == pytest.approx(15, abs=1.0)


def test_invalid_arguments(synthetic_video, tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")
    with pytest.raises(ValueError):
        VideoReader(synthetic_video, stride=0)
    with pytest.raises(ValueError):
        VideoReader(synthetic_video, start_s=2.0, end_s=1.0)
