# src

Mã nguồn của dự án. Code được viết ở máy local, đồng bộ lên server GPU để chạy. Các lệnh chạy từ thư mục gốc của dự án dưới dạng module, ví dụ `python -m src.training.train_rfdetr`.

| Thư mục | Nội dung |
|---|---|
| `configs/detectors/` | Siêu tham số huấn luyện và suy luận cho từng detector (RF-DETR, YOLO11, Faster R-CNN) |
| `configs/trackers/` | Tham số của 5 tracker (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT) |
| `detectors/` | Lớp bọc cho từng detector với cùng giao diện `predict(image) -> (N, 6)` |
| `trackers/` | Lớp bọc chung `Tracker(name, cfg).update(dets, frame) -> (M, 7)` cho 5 tracker |
| `training/` | Script huấn luyện từng detector |
| `inference/` | Cache detections (định dạng `frame,-1,x,y,w,h,score,class`), chạy tracker trên cache, xuất video |
| `evaluation/` | Đánh giá detection bằng pycocotools và tracking bằng TrackEval, xuất bảng CSV |
| `utils/` | Hàm dùng chung: đọc ghi file, log, vẽ box, đo thời gian |
| `scripts/` | Script shell: đồng bộ code lên server, chạy job nền, kéo kết quả về |
| `notebooks/` | Notebook phân tích dữ liệu và kết quả |

Dữ liệu và code tiền xử lý dùng chung nằm ở thư mục [`data/`](../data/README.md) ngoài `src/`. Trọng số và kết quả lớn không đưa lên git. Mã nguồn RF-DETR gốc nằm ở `models/rf-detr/` (bản `1.11.2`).
