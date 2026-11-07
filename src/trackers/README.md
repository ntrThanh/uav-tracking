# Trackers

Mỗi thuật toán có hai phiên bản, chọn bằng tên tracker và dùng chung một interface:

| Thuật toán | Tự cài đặt | Dùng thư viện | Thư viện |
|---|---|---|---|
| SORT | `sort` | `sort_lib` | `sort-tracker-py` (code gốc của Bewley) |
| DeepSORT | `deepsort` | `deepsort_lib` | `deep-sort-realtime` |
| ByteTrack | `bytetrack` | `bytetrack_lib` | `boxmot` |
| OC-SORT | `ocsort` | `ocsort_lib` | `boxmot` |
| BoT-SORT | `botsort` | `botsort_lib` | `boxmot` |

Bản tự cài đặt dùng chung các thành phần trong `core/`, chỉ phụ thuộc NumPy, SciPy (thuật toán Hungarian), OpenCV (optical flow) và mạng OSNet lấy từ `boxmot` trong `appearance.py`. Bản thư viện là adapter mỏng trong `algorithms/lib_*.py`, nhận cùng file tham số và cùng đặc trưng OSNet. `boxmot` 26 chỉ có một ngưỡng cho cả ghép cặp và tạo track mới, nên `bytetrack_lib` không có `new_track_thresh`.

## Cách dùng

```python
from src.trackers import Tracker, load_config

tracker = Tracker("bytetrack", load_config("bytetrack", det_thresh=0.4))   # mỗi chuỗi tạo một Tracker mới
tracks = tracker.update(dets, frame)    # dets (N, 6): x1,y1,x2,y2,score,class -> tracks (M, 7): x1,y1,x2,y2,id,score,class
```

`frame` là ảnh BGR, chỉ bắt buộc khi `tracker.requires_image` là `True` (DeepSORT, BoT-SORT).

Chạy trên cache detection, trong env `track` (`requirements-track.txt`):

```bash
python -m src.inference.run_tracker --det frcnn_r50 --split val --tracker bytetrack
python -m src.inference.run_tracker --det frcnn_r50 --split val --tracker botsort --det-thresh 0.4 --tag thr04
python -m pytest tests
```

Kết quả nằm ở `results/tracks/<det>/<tracker>[_tag]/<split>/<seq>.txt`, mỗi dòng là `frame,id,x,y,w,h,score,class,-1,-1`. File `run_meta.json` đi kèm ghi tham số và thời gian chạy.

## Cấu trúc

| File | Trách nhiệm |
|---|---|
| `base.py` | Các hợp đồng: `TrackingAlgorithm`, `TrackerConfig`, `AppearanceEncoder`, `CameraMotionEstimator`, `FrontEnd` |
| `factory.py` | `Tracker`: tách lớp, chạy mỗi lớp một instance, cấp ID duy nhất trong chuỗi |
| `registry.py` | Ánh xạ tên sang thuật toán, tự tìm module trong `algorithms/` |
| `core/association.py` | IoU, gán cặp tối ưu, khoảng cách cosine |
| `core/motion.py` | Ba mô hình Kalman cùng một interface: XYSR, XYAH, XYWH |
| `core/track.py` | Trạng thái chung của một track |
| `camera_motion.py` | Ước lượng chuyển động camera bằng sparse optical flow |
| `appearance.py` | OSNet, nơi duy nhất phụ thuộc `boxmot` |
| `io.py` | Đọc cache detection và `seqinfo.ini`, định dạng kết quả |
| `algorithms/` | Mỗi thuật toán một file; `lib_*.py` là adapter thư viện |

## Nguyên tắc SOLID trong code

- **Single responsibility:** mỗi file ở bảng trên làm một việc. Thuật toán không biết đến lớp đối tượng, file hay ảnh; việc tách lớp nằm ở `Tracker`, việc đọc file nằm ở `io.py`.
- **Open/closed:** thêm tracker mới không phải sửa code cũ, vì `registry.py` tự tìm module mới. BoT-SORT mở rộng ByteTrack qua các hook, không sửa ByteTrack.
- **Liskov substitution:** mọi thuật toán thay thế được cho nhau qua `step()`. `tests/test_trackers.py` chạy cùng bộ kiểm tra hành vi cho mọi tracker đã đăng ký, gồm cả tracker mới.
- **Interface segregation:** `AppearanceEncoder` và `CameraMotionEstimator` là hai interface nhỏ riêng biệt. SORT không phải biết đến cả hai.
- **Dependency inversion:** thuật toán phụ thuộc vào interface, không phụ thuộc OSNet. Khi test, có thể truyền `encoder` giả vào `Tracker`.

## Thêm một tracker mới

1. Tạo `src/trackers/algorithms/<ten>.py`:

```python
from dataclasses import dataclass
import numpy as np
from ..base import FrameContext, TrackerConfig, TrackingAlgorithm, track_rows
from ..registry import register

@dataclass(frozen=True)
class MyConfig(TrackerConfig):
    max_age: int = 30

@register("mytracker")
class MyTracker(TrackingAlgorithm):
    config_cls = MyConfig

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        # dets: (N, 6) của một lớp, có thể rỗng. Trả về (M, 7).
        return track_rows([])
```

2. Tạo `src/configs/trackers/mytracker.yaml` với các tham số. Tham số sai tên sẽ báo lỗi ngay khi nạp.
3. Chạy `python -m pytest tests`. Tracker mới tự động được kiểm tra, và `--tracker mytracker` dùng được ngay.

Muốn bọc một thư viện thay vì tự viết thì làm tương tự, xem `algorithms/lib_sort.py`. Nếu thư viện tự tách lớp, đặt `handles_classes = True` như `lib_boxmot.py`.

Nếu thuật toán cần ReID hoặc bù chuyển động camera, override `needs_encoder()` và `build_front_end()` như `algorithms/botsort.py`. Nếu ngưỡng khác phụ thuộc vào `det_thresh`, override `with_det_thresh()` trong config.
