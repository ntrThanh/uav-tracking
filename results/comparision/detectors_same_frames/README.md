# So sánh định tính 3 detector trên cùng 6 frame VisDrone-MOT

## 1. Mục tiêu

Trực quan hóa kết quả phát hiện đối tượng của **YOLO11m**, **Faster R-CNN ResNet50-FPN v2** và **RF-DETR Medium** sau fine-tuning trên **cùng 6 frame** thuộc một sequence của **VisDrone2019-MOT test-dev**.

Mỗi hình gồm bốn cột theo thứ tự:

**Ground Truth (GT) | YOLO11m | Faster R-CNN | RF-DETR**

Đây là **thí nghiệm định tính** nhằm quan sát đối tượng được phát hiện, bỏ sót, nhầm lớp và vị trí bounding box. Thí nghiệm **không tính mAP, TP/FP/FN hay chỉ số tracking** và không dùng số bounding box được vẽ để xếp hạng detector.

## 2. Dữ liệu và lựa chọn frame

- Dataset: [VisDrone MOT+DET converted trên Kaggle](https://www.kaggle.com/datasets/tiendq/visdrone-mot-det-converted).
- Split: `VisDrone2019-MOT-test-dev`.
- Ground truth: `data/processed/coco/test-dev.json` do `data/tools/prepare.py` của repo tạo, mã lớp VisDrone **1–10**.
- Cách chọn: **6 frame cách đều trong cùng một sequence**, trừ khi thiết lập `FRAME_IDS` thủ công.
- Lần chạy đã kiểm tra trước khi bàn giao: sequence **`uav0000009_03358_v`** (V00, tức sequence ở chỉ số 0 khi sắp xếp tên), các frame **1, 44, 88, 131, 175, 219**.

**Nguồn xác nhận cuối cùng:** `summary.json` ghi lại `sequence`, `frame_ids`. 

## 3. Mô hình và thiết lập inference

| Mô hình | Kiến trúc / cách nạp | Kích thước đầu vào | Ánh xạ nhãn |
|---|---|---|---|
| YOLO11m | Ultralytics, checkpoint `best_yolo11m.pt` | `imgsz=640` | Model 0–9 → VisDrone 1–10 |
| Faster R-CNN | `src.training.frcnn_colab.make_model()`, checkpoint `best_frcnn.pt` | `min_size=800`, `max_size=1333` | Giữ nguyên 1–10; 0 là background |
| RF-DETR Medium | API `RFDETRMedium(pretrain_weights=...)`, checkpoint `best_ema.pth` | `resolution=576` | Model 0–9 → VisDrone 1–10 |

- **Ngưỡng confidence hiển thị:** `0.25` cho cả ba mô hình.
- YOLO11m: `max_det=1000`, IoU/NMS `0.7` trong lệnh predict.
- Faster R-CNN: giới hạn 1.000 detection/ảnh theo cấu hình repo.
- RF-DETR: giới hạn query theo kiến trúc (thông tin cấu hình trong `summary.json`).
- Cùng ảnh gốc, cùng hệ tọa độ và cùng màu cho 10 lớp trong cả bốn cột.
- GT không có điểm confidence; mặc định không vẽ vùng `ignored` (`SHOW_IGNORED_GT=False`).

**Lưu ý về confidence:** Điểm confidence của ba kiến trúc chưa được hiệu chỉnh để tương đương về xác suất. Ngưỡng chung `0.25` chỉ phục vụ việc hiển thị, không bảo đảm ba mô hình có cùng precision/recall.

## 4. Các file kết quả

```text
results/comparison/detectors_same_frames/
├── README.md
├── comparison_01_<sequence>_frame<id>.jpg
├── comparison_02_<sequence>_frame<id>.jpg
├── comparison_03_<sequence>_frame<id>.jpg
├── comparison_04_<sequence>_frame<id>.jpg
├── comparison_05_<sequence>_frame<id>.jpg
├── comparison_06_<sequence>_frame<id>.jpg
├── counts_per_frame.csv
└── summary.json
```

- **6 ảnh JPG:** mỗi ảnh là lưới 1×4 (GT, YOLO11m, Faster R-CNN, RF-DETR), kèm chú giải 10 lớp.
- **`counts_per_frame.csv`:** tên ảnh, sequence, frame ID, số box GT, số detection hiển thị của từng detector và số vùng ignored.
- **`summary.json`:** dataset/split, sequence/frame IDs, Git commit, tên và SHA-256 checkpoint, tham số inference, phiên bản thư viện và thống kê từng frame.

Các tên JPG thực tế chứa tên sequence và frame cụ thể; danh sách chuẩn nằm trong `counts_per_frame.csv` hoặc `summary.json`. 

## 5. Chạy lại trên Kaggle

Notebook: [`src/notebooks/07_compare_detectors_same_frames_kaggle.ipynb`](../../../src/notebooks/07_compare_detectors_same_frames_kaggle.ipynb).

1. Tạo Kaggle Notebook, thêm dataset VisDrone đã chuyển đổi và các Kaggle Input chứa **ba checkpoint tương ứng**; bật GPU và Internet nếu cần cài thư viện/clone repo.
2. Kiểm tra các đường dẫn `DATASET_ROOT`, `YOLO_CHECKPOINT`, `FRCNN_CHECKPOINT`, `RFDETR_CHECKPOINT` trong cell cấu hình.
3. Giữ `SEQ_INDEX=0`, `N_FRAMES=6`, `FRAME_IDS=None`, `SCORE_THRESH=0.25` để lặp lại thiết lập chọn 6 frame V00 (hoặc đặt `SEQ_NAME`/`FRAME_IDS` khi cần).
4. Chạy các cell lần lượt. Ba mô hình inference **tuần tự** nhằm giảm sử dụng VRAM; code Faster R-CNN và RF-DETR bám theo repo.
5. Lấy kết quả từ `/kaggle/working/compare_3_detectors_mot_testdev/`. Notebook dọn **riêng thư mục này** trước khi xuất lại, sau đó chỉ tạo **6 JPG + CSV + JSON** (8 file).

Mã nguồn tham chiếu: [uav-tracking](https://github.com/ntrThanh/uav-tracking), commit được ghim trong notebook **`7a23f4c4f0af306e497391f32f2c04f441958a98`**. Hash và phiên bản chạy thực tế được ghi trong `summary.json`.

## 6. Cách diễn giải và giới hạn

Các hình cho phép **quan sát định tính** về sự khác biệt trong phát hiện vật thể nhỏ, độ khớp bounding box, bỏ sót, báo nhầm và nhầm lớp. Tuy nhiên:

- **Số box không bằng số dự đoán đúng.** Một detector vẽ nhiều box hơn có thể có cả detection đúng và false positive.
- Sáu frame của một sequence không đại diện cho toàn bộ test-dev.
- Mỗi model có cơ chế confidence và hậu xử lý khác nhau; sử dụng ngưỡng chung để vẽ không phải phép so sánh định lượng đã chuẩn hóa.
- Vùng ignored được giữ trong metadata nhưng không vẽ mặc định; hình trực quan không áp dụng đầy đủ quy tắc đánh giá ignored-region.

Để kết luận detector nào tốt hơn, cần đánh giá định lượng trên **cùng dữ liệu, cùng tập lớp và cùng giao thức** (COCO/pycocotools, AP50–95, AP_small và chính sách xử lý vùng ignored thống nhất).
