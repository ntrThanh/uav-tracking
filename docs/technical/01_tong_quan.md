# 1. Tổng quan dự án

## 1.1. Bài toán

Dự án giải quyết bài toán **theo dõi nhiều đối tượng (Multi-Object Tracking, MOT)** trong video quay từ drone. Với mỗi frame, hệ thống phải phát hiện các đối tượng như người đi bộ, ô tô, xe tải, xe buýt, rồi gán cho mỗi đối tượng một ID giữ nguyên qua thời gian. Đầu ra là tập quỹ đạo, mỗi quỹ đạo là chuỗi box của cùng một đối tượng qua các frame.

Dữ liệu drone khó hơn dữ liệu camera giám sát thông thường ở ba điểm. Đối tượng rất nhỏ vì camera bay cao, cạnh trung vị của box chỉ khoảng 33–46 px trên ảnh gốc có độ phân giải lên tới 4K. Mật độ đối tượng cao, trung bình khoảng 45 box mỗi frame. Camera di chuyển liên tục, nên vị trí đối tượng trên ảnh thay đổi cả do chính đối tượng lẫn do drone.

Dự án theo hướng **tracking-by-detection**. Một detector phát hiện đối tượng trên từng frame, sau đó một tracker liên kết các detection giữa các frame thành quỹ đạo. Cách tiếp cận này cho phép thay detector và tracker độc lập với nhau, từ đó so sánh có hệ thống từng thành phần.

## 1.2. Mục tiêu

Mục tiêu chính là so sánh **3 detector × 5 tracker** trên VisDrone2019-MOT, trả lời ba câu hỏi. Thứ nhất, detector nào phù hợp nhất với đối tượng nhỏ và dày đặc trong ảnh drone. Thứ hai, thứ hạng các tracker có giữ nguyên khi đổi detector hay không. Thứ ba, chất lượng tracking bị giới hạn chủ yếu bởi khâu phát hiện hay khâu liên kết.

| Thành phần | Các lựa chọn |
|---|---|
| Detector | RF-DETR Medium (transformer), YOLO11m (một giai đoạn), Faster R-CNN R50-FPN v2 (hai giai đoạn) |
| Tracker | SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT |
| Dữ liệu | VisDrone2019-MOT: train để huấn luyện, val để chọn tham số, test-dev để báo cáo |
| Mốc tham chiếu | Oracle: dùng ground truth làm detection để đo riêng chất lượng tracker |

Ba detector đại diện cho ba họ kiến trúc khác nhau, giúp phần so sánh có ý nghĩa về mặt phương pháp chứ không chỉ là so sánh các phiên bản. Năm tracker trải từ đơn giản nhất (SORT chỉ dùng Kalman và IoU) đến phức tạp nhất (BoT-SORT có ReID và bù chuyển động camera).

## 1.3. Phân công

| Thành viên | Detector | Việc chung phụ trách |
|---|---|---|
| Nguyễn Trọng Thành (nhóm trưởng) | RF-DETR Medium | Repo, môi trường, tải dữ liệu, tiền xử lý dùng chung, đánh giá detection, tổng hợp kết quả, điều phối |
| Đinh Bá Thi | Faster R-CNN R50-FPN v2 | Code 5 tracker và cấu hình, đo FPS thống nhất cho 3 mô hình |
| Đinh Quang Tiến | YOLO11m | Baseline zero-shot, ground truth cho TrackEval, script đánh giá tracking, chạy oracle |

Chi tiết phân công và tiến độ theo tuần nằm trong [Plan_3Detector_5Tracker.md](../Plan_3Detector_5Tracker.md).

## 1.4. Luồng xử lý

```
 data/raw (VisDrone gốc)
        │
        ▼  data/tools/prepare.py
 data/processed/{coco, rfdetr, yolo, mot} + data/splits
        │
        ├──► Train RF-DETR ──┐
        ├──► Train YOLO11m ──┼──► checkpoint tốt nhất theo mAP val
        └──► Train FRCNN  ───┘
                             │
                             ▼  cache detections (conf 0.05)
               results/dets/<detector>/<split>/<seq>.txt   (định dạng chung)
                             │
          ┌──────────────────┼────────────── + oracle (ground truth)
          ▼                  ▼
   Đánh giá detection   5 tracker chạy trên cache
   (pycocotools)             │
                             ▼
                   results/tracks/<detector>/<tracker>/<split>/
                             │
                             ▼  TrackEval, từng lớp rồi lấy trung bình
                   Ma trận 3 × 5 + oracle: HOTA, IDF1, MOTA, ...
```

Nguyên tắc cốt lõi là **train một lần, cache detection một lần, chạy tracker nhiều lần**. Mọi tracker đọc chung một file detection đã lưu, nên khác biệt giữa các tracker không bị lẫn với nhiễu từ việc chạy lại detector, và mỗi lần chạy tracker chỉ mất vài phút.

## 1.5. Cấu trúc repo

```
uav-tracking/
├── README.md                 # hướng dẫn sử dụng
├── requirements.txt
├── docs/
│   ├── README.md             # mục lục tài liệu
│   ├── technical/            # tài liệu kỹ thuật (thư mục này)
│   └── Plan_*.md, Task.md    # kế hoạch, phân công
├── data/
│   ├── README.md             # mô tả dữ liệu
│   ├── tools/                # tiền xử lý dùng chung
│   ├── splits/               # danh sách frame dùng chung
│   ├── raw/                  # dữ liệu gốc (không lên git)
│   └── processed/            # dữ liệu đã chuyển định dạng (không lên git)
├── src/
│   ├── configs/{detectors,trackers}/
│   ├── detectors/  trackers/  training/  inference/  evaluation/  utils/  scripts/
│   └── notebooks/            # 01_eda_visdrone_mot.ipynb
├── models/rf-detr/           # mã nguồn RF-DETR 1.11.2 (không lên git)
└── outputs/eda/              # biểu đồ và bảng thống kê
```

## 1.6. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Tải và kiểm tra dữ liệu VisDrone2019-MOT | Hoàn thành |
| Phân tích thống kê dữ liệu (notebook EDA) | Hoàn thành |
| Tiền xử lý dùng chung (COCO, RF-DETR, YOLO, MOT, splits) | Hoàn thành, đã kiểm tra |
| Môi trường server (conda `rfdetr`) | Có gói phân tích; chưa cài torch và `rfdetr` |
| Mã nguồn RF-DETR | Đã clone bản 1.11.2 |
| Script train, cache detection, tracker, đánh giá | Chưa viết, khung thư mục đã có trong `src/` |
| Train 3 detector | Chưa bắt đầu |
