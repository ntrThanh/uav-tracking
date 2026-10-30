# 4. Detector

## 4.1. Ba họ kiến trúc

| | RF-DETR Medium | YOLO11m | Faster R-CNN R50-FPN v2 |
|---|---|---|---|
| Họ | Transformer, end-to-end | CNN một giai đoạn | CNN hai giai đoạn |
| Người phụ trách | Thành | Tiến | Thi |
| Thư viện | `rfdetr` 1.11.2 | Ultralytics | torchvision |
| Backbone | DINOv2 ViT-S, patch 16, attention theo cửa sổ | CSP với C3k2, C2PSA | ResNet-50 + FPN |
| Tham số (xấp xỉ) | 33M | 20M | 43M |
| Hậu xử lý | Không cần NMS (gán một-một kiểu Hungarian) | NMS | NMS |
| Khởi tạo | Trọng số COCO | Trọng số COCO | Trọng số COCO (`COCO_V1`) |

**RF-DETR** do Roboflow phát triển, kế thừa LW-DETR và Deformable DETR, dùng backbone DINOv2 đã được tiền huấn luyện tự giám sát. Mô hình dự đoán trực tiếp một tập cố định (300) box qua các query, được huấn luyện bằng ghép cặp một-một với ground truth, nên không cần NMS. Ưu điểm là biểu diễn đặc trưng mạnh và hậu xử lý đơn giản. Nhược điểm trên dữ liệu này là backbone ViT chia ảnh thành patch 16 px, nên vật thể nhỏ hơn một patch rất khó được biểu diễn tốt.

**YOLO11m** là detector một giai đoạn, dự đoán box trực tiếp trên ba mức feature map, tối ưu cho tốc độ. Ultralytics có sẵn pipeline augmentation mạnh (mosaic, HSV, scale).

**Faster R-CNN** là detector hai giai đoạn kinh điển. Region Proposal Network đề xuất vùng, sau đó RoI head phân loại và tinh chỉnh box. FPN giúp phát hiện ở nhiều tỉ lệ. Faster R-CNN mặc định dùng ảnh lớn (cạnh dài 1333), nên có lợi thế tự nhiên với vật thể nhỏ.

## 4.2. Cấu hình huấn luyện chung

| | RF-DETR Medium | YOLO11m | Faster R-CNN |
|---|---|---|---|
| Dữ liệu | `data/processed/rfdetr/` | `data/processed/yolo/visdrone.yaml` | `data/processed/coco/*.json` |
| Số lớp | 10 | 10 | 11 (gồm nền) |
| Input | `resolution=576` (bội số của 32) | `imgsz=640` | `min_size=800, max_size=1333` |
| Epoch | 60, `early_stopping=True` | 80, `close_mosaic=10`, `patience=20` | 24 (lịch 2x), giảm lr ×0,1 ở epoch 16 và 22 |
| Optimizer | AdamW, `lr=1e-4`, `lr_encoder=1.5e-4`, wd 1e-4 | SGD, `lr0=0.01`, momentum 0,937, wd 5e-4, cosine, warmup 3 epoch | SGD, `lr=0.02×batch/16`, momentum 0,9, wd 1e-4, warmup 500 iter |
| Batch | 4 × `grad_accum_steps=4` = 16 | 16 | 4 (lr chỉnh theo batch) |
| AMP | Bật | Bật | Bật |
| Chọn checkpoint | `checkpoint_best_total.pth` | `best.pt` theo mAP50-95 val | Epoch có mAP50-95 val cao nhất |
| Seed | 0 | 0 | 0 |

Ba quy tắc công bằng bắt buộc: dùng cùng danh sách ảnh trong `data/splits/`, tính mAP của cả ba mô hình bằng **cùng một script pycocotools trên DET cache** (không lấy số từ log của từng thư viện vì mỗi thư viện tính hơi khác nhau), và đo tốc độ trên cùng GPU với cùng điều kiện.

## 4.3. RF-DETR chi tiết

Các thông tin dưới đây được đọc trực tiếp từ mã nguồn `models/rf-detr` bản 1.11.2.

### Cấu hình mặc định của Medium

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `resolution` | 576 | `RFDETRMediumConfig` |
| `patch_size` × `num_windows` | 16 × 2 = 32 | `RFDETRMediumConfig` |
| `num_queries` / `num_select` | 300 / 300 | `ModelConfig` |
| `dec_layers` | 4 | `RFDETRMediumConfig` |
| Ràng buộc resolution | Phải chia hết cho 32 | `detr.py`, kiểm tra `resolution % (patch_size × num_windows)` |
| `multi_scale`, `expanded_scales` | Bật, bật | `TrainConfig` |
| `use_ema`, `ema_decay` | Bật, 0,993 | `TrainConfig` |
| `early_stopping_patience` | 10 epoch | `TrainConfig` |
| `eval_max_dets` | 500 | `TrainConfig` |
| `checkpoint_interval` | 10 epoch | `TrainConfig` |
| `num_workers` | 2 | `TrainConfig`, nên tăng lên 8 trên server 32 nhân |

Ràng buộc "bội số của 56" ghi trong [Task.md](../Task.md) là của các bản RF-DETR cũ dùng patch 14. Với bản hiện tại, resolution hợp lệ cho Medium là các bội số của 32 như 576, 640, 768, 896, 1024.

### Cấu trúc dữ liệu đầu vào

```
data/processed/rfdetr/
├── train/_annotations.coco.json + ảnh
├── valid/_annotations.coco.json + ảnh
└── test/_annotations.coco.json + ảnh
```

Thư viện tự kiểm tra sự tồn tại của `train/_annotations.coco.json`. Nó đánh lại mã danh mục thành chỉ số liên tục dựa trên file train, nên mã VisDrone 1–10 trở thành nhãn 0–9 bên trong mô hình. Khi xuất DET cache phải cộng 1 để trở về mã VisDrone. Kế hoạch yêu cầu sau epoch đầu tiên vẽ prediction lên ảnh để xác nhận ánh xạ lớp đúng.

### Cách gọi

Bằng Python:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(resolution=576)
model.train(
    dataset_dir="data/processed/rfdetr",
    output_dir="runs/det/T3_rfdetr_m_576",
    epochs=60, batch_size=4, grad_accum_steps=4,
    lr=1e-4, lr_encoder=1.5e-4, weight_decay=1e-4,
    early_stopping=True, seed=0, num_workers=8,
)
```

Bằng dòng lệnh, dựa trên file mẫu `models/rf-detr/configs/rfdetr_medium.yaml`:

```bash
rfdetr fit --config src/configs/detectors/rfdetr_medium.yaml
```

Script `src/training/train_rfdetr.py` (sẽ viết) là lớp bọc mỏng, đọc file YAML của nhóm, gọi `model.train(...)` và ghi một dòng vào `results/runs_log.csv`.

### Các cấu hình thí nghiệm

| Mã | Cấu hình | Mức độ | Mục đích |
|---|---|---|---|
| I1-3 | Trọng số COCO, không train | Bắt buộc | Baseline zero-shot |
| **T3** | Medium, resolution 576, augmentation mặc định | **Bắt buộc** | Mô hình chính trong ma trận 3 × 5 |
| T3-hr | Medium, resolution 1024 (hoặc 896), giữ các tham số khác | Nên làm | Kiểm chứng giả thuyết độ phân giải từ thống kê |
| T3-aug | Resolution tốt nhất + biến đổi độ sáng, độ tương phản nhẹ | Tùy chọn | Ablation augmentation |

Mỗi cấu hình chỉ thay đổi một yếu tố so với T3, để khác biệt về kết quả quy được cho đúng yếu tố đó. Khi tăng resolution lên 1024, nên giảm `batch_size` xuống 2 và tăng `grad_accum_steps` lên 8 để giữ batch hiệu dụng 16; cần đo VRAM và thời gian của epoch đầu trước khi chạy hết.

Khi trình bày, T3-hr khép kín lập luận của dự án: thống kê cho thấy ở 576 có 27,9% box nhỏ hơn 8 px, từ đó đặt giả thuyết độ phân giải là nút thắt, và thí nghiệm T3-hr kiểm chứng giả thuyết bằng AP_small và HOTA.

### Lỗi thường gặp

| Lỗi | Cách tránh |
|---|---|
| Lệch mã lớp khi xuất kết quả | Cộng 1 cho nhãn 0–9; vẽ prediction sau epoch đầu để kiểm tra |
| Chỉ giữ top-100 khi suy luận | Giữ top-300 (bằng số query) vì frame đông nhất có 180 box |
| Resolution không hợp lệ | Chọn bội số của 32 |
| Hết VRAM ở resolution cao | Nhớ rằng multi-scale có thể tạo ảnh lớn hơn resolution tới 160 px |
| Dataloader chậm | Tăng `num_workers` |

## 4.4. YOLO11m và Faster R-CNN

**YOLO11m** train bằng lệnh Ultralytics:

```bash
yolo detect train model=yolo11m.pt data=data/processed/yolo/visdrone.yaml imgsz=640 epochs=80 batch=16 \
     optimizer=SGD lr0=0.01 cos_lr=True close_mosaic=10 patience=20 seed=0 project=runs/det name=T1_yolo11m_640
```

Khi suy luận phải đặt `max_det=1000` và `conf=0.05`. Mã lớp đầu ra 0–9, cộng 1 khi xuất DET cache.

**Faster R-CNN** dùng `fasterrcnn_resnet50_fpn_v2(weights="COCO_V1")`, thay predictor bằng `FastRCNNPredictor(in_features, num_classes=11)`, đọc `data/processed/coco/train.json` với gốc ảnh `data/raw/`. Bắt buộc đặt `box_detections_per_img=1000` (mặc định chỉ 100, sẽ mất box ở frame đông), `box_score_thresh=0.05`, giữ `box_nms_thresh=0.5`.

## 4.5. Cache detection

Sau khi chọn được checkpoint, mỗi detector chạy **một lần** trên mọi frame của val (7 chuỗi) và test-dev (17 chuỗi), xuất DET cache theo định dạng chung ở mục 3.6, kèm `timing.json` ghi thời gian xử lý mỗi frame.

| Thiết lập | Giá trị |
|---|---|
| Ngưỡng score | 0,05 |
| NMS | Mặc định từng mô hình: YOLO 0,7; Faster R-CNN 0,5; RF-DETR không dùng |
| Số box tối đa | 300 (RF-DETR), 1000 (YOLO, Faster R-CNN) |
| Kiểm tra | Vẽ DET cache của một chuỗi val lên ảnh trước khi chạy tracker |

Dòng **oracle** dùng ground truth làm detection với score bằng 1, xuất cùng định dạng, để đo chất lượng tracker tách biệt khỏi detector.

## 4.6. Ước lượng tài nguyên

Server có 2 GPU NVIDIA L4 (23 GB mỗi card), 32 nhân CPU và 156 GB RAM. L4 chậm hơn RTX 3090/4090 nên thời gian thực tế có thể dài hơn ước lượng trong kế hoạch. Nên đo thời gian của epoch đầu rồi nhân lên trước khi cam kết số epoch.

| Việc | Ước lượng trong kế hoạch (GPU 24 GB) |
|---|---|
| YOLO11m, 640, 80 epoch | 6–10 giờ |
| Faster R-CNN, 24 epoch | 10–16 giờ |
| RF-DETR Medium, 60 epoch, 576 | 12–20 giờ |
| RF-DETR ở 1024 | Khoảng 2–3 lần ở 576 |
| Cache detection mỗi detector | 10–40 phút |
