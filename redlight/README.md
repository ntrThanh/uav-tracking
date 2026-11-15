# Red-Light Violation Detection

Hệ thống phát hiện xe vượt đèn đỏ trên video offline từ camera cố định: người dùng vẽ vạch dừng, vùng và ROI đèn tín hiệu trên giao diện web, hệ thống theo vết xe, phát hiện xe cắt vạch khi đèn đỏ, nhận diện biển số xe vi phạm và lưu bằng chứng.

> Dữ liệu video giao thông và biển số xe là dữ liệu nhạy cảm. Không commit video, ảnh bằng chứng, crop biển số hay cơ sở dữ liệu vi phạm lên git (các thư mục `data/`, `outputs/`, `evidence/`, `weights/` đã được ignore).

## Trạng thái

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| GĐ1 | Đọc video, detect, track, xuất video có bbox và track ID | Xong (CPU local 25 FPS, L4 84 FPS) |
| GĐ2 | Giao diện vẽ vạch, vùng, ROI đèn; lưu/tải JSON | Xong, chờ xác nhận |
| GĐ3 | Trạng thái đèn + logic cắt vạch + unit test | Chưa làm |
| GĐ4 | Detect biển số, OCR, vote | Chưa làm |
| GĐ5 | Lưu bằng chứng + trang xem vi phạm | Chưa làm |
| GĐ6 | Script đánh giá + đo FPS | Chưa làm |

## Cài đặt

Yêu cầu Python 3.10 trở lên. Cài torch theo phần cứng trước, sau đó cài phần còn lại.

```bash
conda create -n redlight python=3.11 -y && conda activate redlight
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # hoặc bản CUDA
pip install -r requirements.txt
```

Weights YOLO đặt trong `weights/` (ví dụ `weights/yolo11n.pt`, Ultralytics tự tải nếu thiếu). Đường dẫn weights và mọi ngưỡng nằm trong `config/pipeline.json`.

## Chạy GĐ1

```bash
python scripts/run_video.py --video data/samples/car-detection.mp4 --out outputs/car_demo
python scripts/run_video.py --video <video.mp4> --start 10 --end 40 --stride 2   # xử lý một đoạn
python -m pytest                                                                 # unit test
```

Kết quả trong thư mục `--out`: `annotated.mp4` (bbox, ID, điểm tham chiếu, vệt di chuyển), `tracks.csv` (một dòng mỗi cặp frame và track), `summary.json` (FPS, thống kê độ dài track) và `ref.jpg` (frame tham chiếu để vẽ cấu hình ở GĐ2).

## Chạy GĐ2: giao diện vẽ cấu hình

```bash
python scripts/run_web.py --port 8000          # mở http://localhost:8000
# trên server: ssh -L 8000:localhost:8000 ibmcloud rồi chạy lệnh trên trong môi trường redlight
```

Trong giao diện: nhập `camera_id`, lấy frame tham chiếu từ đường dẫn video trên máy chạy server (hoặc tải ảnh lên), chọn công cụ rồi vẽ trên ảnh. Vạch dừng là hai điểm A→B kèm hướng được phép qua vạch; vùng là đa giác (Enter hoặc nháy đúp để kết thúc, có hỏi tên); ROI đèn là hình chữ nhật kéo bằng chuột. Bấm Lưu để ghi `config/<camera_id>.json`, frame tham chiếu được lưu cạnh đó là `config/<camera_id>_ref.jpg`. API: `GET/PUT /api/cameras/{id}`, `POST /api/cameras/{id}/reference/from-video`, `POST /api/cameras/{id}/reference/upload`.

## Cấu hình

`config/pipeline.json` chứa tham số model và thuật toán (detector: weights, imgsz, conf, iou, device, lớp xe; tracker: các ngưỡng ByteTrack, `track_buffer`, `history_len`; video: đoạn cần xử lý; output: codec, độ dài vệt).

`config/<camera_id>.json` chứa hình học của camera theo toạ độ chuẩn hoá 0–1, do giao diện GĐ2 ghi ra:

```json
{
  "camera_id": "cam01",
  "reference_frame": "ref.jpg",
  "stop_line": {"a": [0.20, 0.70], "b": [0.80, 0.70], "allowed_direction": "down"},
  "zones": [{"name": "intersection", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.6]]}],
  "light_roi": {"x": 0.45, "y": 0.05, "w": 0.04, "h": 0.08},
  "light_override": null
}
```

## Cấu trúc

```
redlight/
  config/              pipeline.json, cam01.json
  src/
    common/            config, geometry, logging
    ingest/video.py    đọc video, frame tham chiếu
    detect/vehicle.py  detector xe (Ultralytics YOLO)
    track/tracker.py   ByteTrack + lịch sử điểm tham chiếu mỗi track
    viz/draw.py        vẽ bbox, ID, vạch, vùng, ROI đèn
    api/app.py         FastAPI: cấu hình camera, frame tham chiếu
  web/                 giao diện vẽ (HTML/JS thuần, canvas)
  scripts/run_video.py, scripts/run_web.py
  tests/
```

Điểm tham chiếu của mỗi xe là trung điểm cạnh dưới bbox, P = ((x1 + x2) / 2, y2); lịch sử các điểm này theo từng track ID là đầu vào cho logic cắt vạch ở GĐ3.
