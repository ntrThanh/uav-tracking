# Kế hoạch Fine-tune và Inference – MOT trên VisDrone2019-MOT

> Định hướng: Nghiên cứu. Dữ liệu chính: VisDrone2019-MOT. Dữ liệu bổ sung: VN-Traffic-MOT (tự quay).
> File này liệt kê **cụ thể từng lần train và từng lần chạy inference** cần làm, theo thứ tự phụ thuộc.
> Quy ước mã:
> - `D` = chuẩn bị dữ liệu
> - `T` = train / fine-tune
> - `I` = inference
> - `E` = đánh giá

---

## 0. Tổng quan luồng công việc

```
D1–D4 (chuẩn bị dữ liệu)
   │
   ├──► T1–T3  fine-tune 3 detector chính ──► I1 cache detections ──► I2 ma trận tracking ──► E1, E2
   │                                                                      ▲
   ├──► T4–T6  biến thể detector (C1, augmentation) ──► I1 ───────────────┤
   │                                                                      │
   ├──► T7     train ReID (C3) ───────────────────────────────────────────┘
   │
   └──► T8     fine-tune trên dữ liệu VN (C6) ──► I5 đánh giá chéo miền
```

**Nguyên tắc chung:**

- **Train một lần, cache detections, chạy tracker nhiều lần.**
  - Mọi tracker đọc chung file detections đã lưu.
  - Nhờ vậy, so sánh giữa các tracker là công bằng và chạy rất nhanh.
- **Cách dùng các split:**
  - `train`: chỉ dùng để train.
  - `val`: dùng để chọn tham số và checkpoint.
  - `test-dev`: chỉ chạy **một lần** cho cấu hình cuối, không chỉnh gì dựa trên test-dev.
- **Tái lập kết quả:** cố định `seed=0`. Ghi lại GPU, phiên bản thư viện và commit hash vào mỗi thư mục run.

---

## 1. Chuẩn bị dữ liệu (D)

| Mã | Việc | Đầu ra |
|---|---|---|
| D1 | Chuyển nhãn VisDrone-MOT sang **YOLO** cho các chuỗi train và val. Lấy mẫu **1/2 số frame** của train vì các frame liên tiếp gần như trùng nhau. Val giữ toàn bộ frame. Bỏ box lớp 0 (ignored) và lớp 11 (others). | `data/visdrone_yolo/{train,val}/{images,labels}` |
| D2 | Chuyển nhãn sang **COCO json** cho Faster R-CNN. Dùng cùng danh sách frame với D1. | `data/visdrone_coco/{train,val}.json` |
| D3 | Chuyển ground truth sang **MOTChallenge** cho TrackEval, **tách theo từng lớp đánh giá**: pedestrian, car, van, truck, bus. Lưu riêng vùng ignored để lọc kết quả. | `data/trackeval/gt/VisDrone-{val,test}/<seq>/gt/gt.txt` |
| D4 | Tạo tập **ReID** từ crop ground truth của các chuỗi train. Mỗi `target_id` là một identity. Bỏ crop có cạnh < 16 px. Lấy tối đa 20 crop/ID, cách nhau ≥ 5 frame. Chia ID theo tỉ lệ 80/20 thành train và test (query/gallery). | `data/reid_visdrone/{train,query,gallery}` |
| D5 | Tạo tập **tiling** từ D1: cắt ô 640×640, chồng lấn 20%. Bỏ các ô không có đối tượng, giữ lại tối đa 10% số ô rỗng. | `data/visdrone_tiles/` |
| D6 | Xử lý **VN-Traffic-MOT**: chuyển nhãn sang YOLO và MOT. Chia theo video: test 60 clip cố định. Phần còn lại tạo các tập con 25/50/100%. | `data/vn/{test, ft_25, ft_50, ft_100}` |

**Lớp khi train detector:**
- Train đủ **10 lớp VisDrone** (1–10), theo cách làm phổ biến.
- Khi chạy tracking, chỉ giữ 5 lớp đánh giá. Riêng dữ liệu VN giữ thêm lớp `motor`.

**Kiểm tra sau D1–D3:** vẽ nhãn lên khoảng 3 chuỗi bằng OpenCV và xem bằng mắt trước khi train.

---

## 2. Fine-tune (T)

### 2.1. Detector chính – baseline cho bài toán chính

| Mã | Mô hình | Khởi tạo | Dữ liệu | Cấu hình chính | Checkpoint giữ lại |
|---|---|---|---|---|---|
> Chia việc mỗi người 1 detector, chi tiết tham số xem [Plan_3Detector_5Tracker.md](Plan_3Detector_5Tracker.md).

| Mã | Mô hình | Khởi tạo | Dữ liệu | Cấu hình chính | Checkpoint giữ lại |
|---|---|---|---|---|---|
| **T1** | YOLO11m | COCO pretrained | D1 | imgsz 640, epochs 80, batch 16, SGD lr0 0.01, cosine, mosaic on, close_mosaic 10 | `best.pt` theo mAP50-95 trên val |
| **T2** | Faster R-CNN R50-FPN v2 | COCO pretrained (torchvision) | D2 | lịch 2x (24 epoch), min_size 800, max_size 1333, lr 0.02 × (batch/16), giảm lr tại epoch 16 và 22, `box_detections_per_img=1000` | epoch có mAP val cao nhất |
| **T3** | RF-DETR Medium | COCO pretrained | D2 | resolution 576, epochs 60, batch 4 × grad_accum 4, AdamW lr 1e-4, lr_encoder 1.5e-4, early stopping | `checkpoint_best_total.pth` |

Lệnh mẫu (Ultralytics):

```bash
# T1
yolo detect train model=yolo11m.pt data=cfg/visdrone.yaml imgsz=640 epochs=80 batch=16 seed=0 \
     optimizer=SGD lr0=0.01 cos_lr=True project=runs/det name=T1_yolo11m_640
```

> Mốc **giữa kỳ** cần xong T1. T2 và T3 có thể hoàn thành sau giữa kỳ.

### 2.2. Biến thể detector – thí nghiệm C1 và ablation augmentation

Mọi biến thể đều so với **T1**. Mỗi biến thể chỉ đổi đúng một yếu tố, các cấu hình còn lại giữ nguyên như T1.

| Mã | Thay đổi so với T1 | Mục đích |
|---|---|---|
| **T4** | Kiến trúc `yolov8m-p2.yaml` (thêm head P2, stride 4), load trọng số `yolov8m.pt` | C1: head cho vật thể nhỏ |
| **T5** | imgsz **1280**, batch 8 | C1: tăng độ phân giải |
| **T6a** | `mosaic=0` | Ablation augmentation |
| **T6b** | Train trên D5 (tiling) thay cho D1 | Ablation tiling |

```bash
# T4
yolo detect train model=yolov8m-p2.yaml pretrained=yolov8m.pt data=cfg/visdrone.yaml imgsz=640 epochs=80 batch=16 seed=0 \
     project=runs/det name=T4_yolov8m_p2_640
```

> **SAHI** (C1) **không cần train lại**: chạy inference bằng trọng số T1 theo kiểu cắt ô (xem I1).

### 2.3. ReID – thí nghiệm C3

| Mã | Backbone | Khởi tạo | Dữ liệu train | Cấu hình |
|---|---|---|---|---|
| T7-0 | OSNet x1.0, ResNet50, MobileNetV2 | Trọng số **có sẵn** từ Market-1501 (người) và VeRi-776 (xe) | – (**không train**, chỉ tải về) | – |
| **T7a** | MobileNetV2 | ImageNet | D4 | input 128×128, 60 epoch, Adam lr 3e-4, softmax + triplet (margin 0.3), batch P×K = 16×4 |
| **T7b** | ResNet50 | ImageNet | D4 | như T7a |
| **T7c** | OSNet x1.0 | ImageNet | D4 | như T7a |

- **Input 128×128 thay vì 256×128:** đối tượng nhìn từ trên cao có tỉ lệ cạnh gần vuông.
- **Đánh giá riêng phần ReID:** mAP và Rank-1 trên query/gallery của D4. Ghi lại để so sánh với T7-0.

### 2.4. Thích nghi miền dữ liệu VN – thí nghiệm C6

Khởi tạo từ `T1/best.pt`. Mỗi batch trộn dữ liệu VN với một phần VisDrone theo tỉ lệ **VN : VisDrone = 1 : 1**, để giảm hiện tượng quên kiến thức cũ.

| Mã | Dữ liệu VN | Cấu hình |
|---|---|---|
| **T8a** | ft_25 | 30 epoch, lr0 0.001, imgsz 640 |
| **T8b** | ft_50 | như trên |
| **T8c** | ft_100 | như trên |

> Mức 0% chính là T1, không cần train.

### 2.5. Tóm tắt các lần train

| Nhóm | Số lần train | Ghi chú |
|---|---|---|
| Detector chính (T1–T3) | 3 | Bắt buộc |
| Biến thể detector (T4–T6) | 4 | Chọn tối thiểu T4 hoặc T5, cộng T6a |
| ReID (T7a–c) | 3 | Có thể làm trước một backbone (OSNet) |
| Domain adaptation (T8a–c) | 3 | Làm sau khi có nhãn VN |
| **Tổng** | **13** | **Tối thiểu khả thi: 8 lần** (T1, T2, T3, T4, T6a, T7c, T8a, T8c) |

---

## 3. Inference (I)

### I1 – Cache detections

Chạy mỗi detector **một lần** trên mọi frame của từng split. Lưu theo từng chuỗi với định dạng `frame, -1, x, y, w, h, score, class`.

- **Ngưỡng confidence thấp `conf=0.05`:** ByteTrack cần cả box có điểm thấp. Tracker sẽ tự áp ngưỡng sau.
- **NMS:** IoU 0.7. Bật `agnostic_nms=False`.

| Mã | Detector | Trọng số | val | test-dev | VN test |
|---|---|---|:-:|:-:|:-:|
| I1-1 | YOLO11m | COCO pretrained | ✓ | ✓ | |
| I1-2 | Faster R-CNN | COCO pretrained | ✓ | ✓ | |
| I1-3 | RF-DETR Medium | COCO pretrained | ✓ | ✓ | |
| I1-4 | YOLO11m | T1 | ✓ | ✓ | ✓ |
| I1-5 | Faster R-CNN | T2 | ✓ | ✓ | ✓ |
| I1-6 | RF-DETR Medium | T3 | ✓ | ✓ | ✓ |
| I1-7 | YOLOv8m-P2 | T4 | ✓ | ✓ | ✓ |
| I1-8 | YOLOv8m @1280 | T5 | ✓ | ✓ | |
| I1-9 | YOLOv8m no-mosaic / tiling | T6a / T6b | ✓ | | |
| I1-10 | YOLO11m + **SAHI** (ô 640, chồng lấn 0.2) | T1 | ✓ | ✓ | ✓ |
| I1-11 | YOLO11m | T8a / T8b / T8c | ✓ | | ✓ |
| I1-12 | **Oracle**: lấy ground truth làm detections, score = 1 | – | ✓ | ✓ | ✓ |

- **Mô hình COCO pretrained:** chỉ có các lớp person, car, bus, truck, motorcycle, bicycle. Cần ánh xạ person → pedestrian, motorcycle → motor. Lớp van không có trong COCO, nên van sẽ luôn bị bỏ sót. Đây là một điểm cần phân tích trong báo cáo.
- **Đầu ra:** `results/dets/<det_id>/<split>/<seq>.txt`

### I2 – Ma trận tracking (bài toán chính)

Đọc detections từ I1. Mỗi tracker dùng tham số mặc định trước, sau đó dùng tham số đã tối ưu ở I4.

| Tracker | Dùng ReID | Dùng GMC |
|---|:-:|:-:|
| SORT (`sort.py`) | | |
| ByteTrack | | |
| OC-SORT | | |
| DeepSORT (`deep_sort_realtime`) | ✓ | |
| BoT-SORT | ✓ | ✓ |

| Mã | Detections | Tracker | Split |
|---|---|---|---|
| I2-a | I1-4, I1-5, I1-6 (đã fine-tune) | Cả 5 tracker | val → test-dev |
| I2-b | I1-12 (oracle) | Cả 5 tracker | val, test-dev |
| I2-c | I1-1, I1-2, I1-3 (COCO) | ByteTrack, BoT-SORT | val |
| I2-d | I1-7, I1-8, I1-9, I1-10 (biến thể C1) | ByteTrack, BoT-SORT | val → test-dev |

- **Tham số tracker:** xem mục 3 của `Plan_3Detector_5Tracker.md` (giống nhau cho cả 3 detector, sau đó dò ngưỡng det riêng cho từng detector).
- **Tổng số lần chạy tracking:** khoảng 15 (I2-a) + 5 (I2-b) + 6 (I2-c) + 8 (I2-d) = **khoảng 34 lần trên val**.
- **Thời gian:** mỗi lần chỉ mất vài phút vì detections đã có sẵn. Lâu nhất là các tracker có ReID, vì phải trích đặc trưng cho từng box.
- **Đầu ra:** `results/tracks/<det_id>/<tracker>/<split>/<seq>.txt` theo định dạng MOT. Lọc bỏ các track nằm trong vùng ignored trước khi đánh giá.

### I3 – Ablation khâu liên kết (C2, C3, C4)

Detections cố định là **I1-4** (YOLO11m T1). Chạy trên **val**. Cấu hình tốt nhất của mỗi nhóm được chạy thêm trên test-dev.

| Mã | Thí nghiệm | Các biến thể |
|---|---|---|
| I3-GMC | C2: bù chuyển động camera | BoT-SORT với GMC ∈ {none, sparseOptFlow, ECC, ORB}; ByteTrack và OC-SORT ± GMC sparseOptFlow → **8 lần chạy** |
| I3-ReID | C3: ReID | BoT-SORT và DeepSORT × ReID ∈ {T7-0 Market OSNet, T7-0 VeRi OSNet, T7a, T7b, T7c} → **10 lần chạy** |
| I3-BIoU | C4: hàm chi phí | ByteTrack với cost ∈ {IoU, Buffered IoU b=0.3, BIoU b=0.5, khoảng cách tâm chuẩn hóa} → **4 lần chạy** |

- **Với I3-GMC:** ghi thêm thời gian chạy riêng của GMC.
- **Với I3-ReID:** ghi thời gian trích đặc trưng trên mỗi box.

### I4 – Dò tham số (C5)

Chạy trên **val**, với detections I1-4 và tracker ByteTrack. Dò BoT-SORT nếu còn thời gian.

| Tham số | Các giá trị |
|---|---|
| `track_high_thresh` | 0.3, 0.4, 0.5, 0.6 |
| `track_low_thresh` | 0.05, 0.1, 0.2 |
| `match_thresh` | 0.7, 0.8, 0.9 |
| `track_buffer` | 30, 60, 90 |
| Kalman: hệ số nhiễu vị trí và vận tốc | ×0.5, ×1, ×2 |

- **Cách dò:** dò **từng tham số một** quanh giá trị mặc định (one-at-a-time), khoảng 16 lần chạy. Vẽ đường HOTA và IDF1 theo từng tham số.
- **Tham số tốt nhất:** dùng cho I2 trên test-dev.

### I5 – Đánh giá chéo miền dữ liệu (C6, H4)

Tracker cố định là cấu hình tốt nhất từ I2 và I4.

| Mã | Detector | Chạy trên |
|---|---|---|
| I5-1 | T1 (chỉ VisDrone) | VN test |
| I5-2 | T8a, T8b, T8c | VN test |
| I5-3 | T8c | VisDrone test-dev (kiểm tra có quên kiến thức cũ không) |
| I5-4 | T1 + SAHI | VN test |

### I6 – Trích đặc trưng phục vụ EDA

| Mã | Việc | Đầu ra |
|---|---|---|
| I6-1 | Trích đặc trưng mức ảnh bằng ResNet50 (ImageNet) hoặc CLIP ViT-B/32 cho khoảng 2.000 frame VisDrone và khoảng 2.000 frame VN (lấy mẫu đều). Giảm chiều bằng PCA (50 chiều) rồi t-SNE/UMAP (2 chiều). | `results/features/domain_tsne.png` |
| I6-2 | Embedding ReID của 20 ID ngẫu nhiên trong tập test D4, dùng T7-0 (Market) và T7c. Giảm chiều bằng t-SNE, tô màu theo ID. | `results/features/reid_tsne_{market,visdrone}.png` |
| I6-3 | Chuyển động camera theo chuỗi: độ lớn trung bình của phép biến đổi affine giữa hai frame, ước lượng bằng ECC hoặc optical flow. | `results/stats/camera_motion.csv` (dùng để phân nhóm ở E2) |

### I7 – Đo tốc độ (FPS)

- **Điều kiện cố định:** cùng GPU, batch 1, chạy 50 frame khởi động (warm-up) rồi đo trên 1 chuỗi val dài, ở imgsz đã dùng khi train.
- **Đo tách riêng 4 phần:**
  - thời gian detector;
  - thời gian ReID;
  - thời gian tracker và GMC;
  - **FPS end-to-end**.
- **Đối tượng đo:** các detector T1–T5 và SAHI; tracker cấu hình tốt nhất.

### I8 – Video định tính

- **Số lượng:** 6–10 video, mỗi video dài 10–20 giây.
- **Nội dung:** vẽ box, ID và quỹ đạo 30 frame gần nhất; tô đỏ các frame xảy ra đổi ID.
- **Các cặp so sánh đặt cạnh nhau:**
  - ByteTrack với BoT-SORT;
  - có GMC với không GMC;
  - T1 với T8c trên video VN.

---

## 4. Đánh giá (E)

| Mã | Nội dung | Công cụ |
|---|---|---|
| **E0** | mAP của detector trên val và test-dev: mAP50, mAP50-95, mAP theo kích thước small/medium/large | `ultralytics val` hoặc pycocotools trên D2 |
| **E1** | Tracking: HOTA, DetA, AssA, MOTA, IDF1, IDSW, MT, ML. Chạy **TrackEval từng lớp riêng rồi lấy trung bình**, theo cách làm của VisDrone. | `TrackEval/scripts/run_mot_challenge.py --METRICS HOTA CLEAR Identity` |
| **E2** | Phân tích theo nhóm: theo lớp, theo kích thước box, theo mật độ, theo mức chuyển động camera (từ I6-3), theo ngày/đêm (dữ liệu VN) | Script tổng hợp từ đầu ra theo từng chuỗi của TrackEval |
| **E3** | ReID: mAP và Rank-1 trên query/gallery của D4 | torchreid |
| **E4** | Độ đồng thuận nhãn VN: chạy TrackEval giữa nhãn của người A và người B | TrackEval |

**Bảng kết quả cần có trong báo cáo:**

1. mAP của 3 detector, so sánh COCO pretrained với fine-tune (E0).
2. Ma trận 3 detector × 5 tracker và dòng oracle, trên test-dev (E1).
3. Ablation C1: T1, T4, T5, SAHI. Báo mAP-small, HOTA, IDF1, FPS.
4. Ablation C2 (GMC), C3 (ReID), C4 (cost function), trên val và test-dev.
5. Biểu đồ độ nhạy tham số (C5).
6. Đường cong thích nghi miền 0/25/50/100% và bảng đánh giá chéo miền (C6).
7. Bảng FPS tách theo từng thành phần.

---

## 5. Thứ tự thực hiện theo tuần

| Tuần | Train | Inference / Đánh giá |
|---|---|---|
| 1 | D1–D3 | Kiểm tra nhãn; I1-1 (YOLO COCO) → ByteTrack → E1. Đây là **baseline zero-shot** |
| 2 | **T1** | I1-4 → I2-a (5 tracker) → E1 |
| 3 | T2, D4 | I1-12 + I2-b (oracle), I6-3 |
| 4 | T3 | **Giữa kỳ**: bảng T1 × 5 tracker và bảng oracle |
| 5 | T4 (hoặc T5), T6a | I1-5, I1-6 → I2-a đầy đủ |
| 6 | T7c (sau đó T7a, T7b) | I1-7, I1-10 (SAHI) → I2-d; I3-GMC |
| 7 | D6 (khi nhãn VN xong) | I3-ReID, I3-BIoU, I6-2 |
| 8 | T8a–c | I4 dò tham số; I5 đánh giá chéo miền; I6-1 |
| 9 | – | Chạy test-dev cho tất cả cấu hình cuối; I7 đo FPS; I8 video |
| 10+ | – | Tổng hợp bảng, viết báo cáo |

---

## 6. Ước lượng tài nguyên

- **Giả định:** 1 GPU khoảng 16–24 GB (RTX 3090/4090, A4000, hoặc T4/P100 trên Kaggle/Colab nhưng chậm hơn khoảng 2–3 lần).
- **Mức độ tin cậy:** các con số dưới đây chỉ là **ước lượng thô**. Nên đo thời gian 1 epoch thực tế rồi nhân lên.

| Việc | Thời gian ước lượng (GPU 24 GB) |
|---|---|
| T1 (YOLO11m, 640, 80 epoch, khoảng 12.000 ảnh) | khoảng 6–10 giờ |
| T2 (Faster R-CNN, 24 epoch) | khoảng 10–16 giờ |
| T3 (RF-DETR Medium, 60 epoch) | khoảng 12–20 giờ |
| T5 (1280) | khoảng 2–3 lần T1 |
| T7 (mỗi backbone ReID) | khoảng 1–3 giờ |
| T8 (mỗi lần) | khoảng 1–2 giờ |
| I1 (mỗi detector, val + test-dev) | khoảng 10–40 phút; SAHI chậm hơn 4–8 lần |
| I2–I4 (mỗi lần chạy tracker) | vài phút; khoảng 10–30 phút nếu có ReID |

**Nếu thiếu GPU:**
- giảm epoch (T1 50 epoch, T3 50 epoch);
- dùng YOLO11s thay YOLO11m;
- lấy mẫu 1/3 số frame train;
- chỉ train T7c (OSNet).

---

## 7. Quy ước thư mục và ghi log

```
runs/det/<T_id>_<model>_<imgsz>/        # checkpoint, args.yaml, results.csv
runs/reid/<T_id>_<backbone>/
results/dets/<I1_id>/<split>/<seq>.txt
results/tracks/<det_id>/<tracker>[_<variant>]/<split>/<seq>.txt
results/trackeval/<det_id>/<tracker>/<split>/   # summary + per-seq
results/tables/*.csv                     # bảng tổng hợp để đưa vào báo cáo
```

Mỗi run ghi vào `results/runs_log.csv` một dòng gồm các cột:

`run_id, ngày, người chạy, mô hình, dữ liệu, tham số chính, GPU, thời gian, mAP / HOTA / IDF1 / FPS, ghi chú`

---

## 8. Checklist

- [ ] D1–D6 xong và đã kiểm tra nhãn bằng mắt
- [ ] T1 · T2 · T3
- [ ] T4 · T5 · T6a · T6b
- [ ] T7a · T7b · T7c (+ tải T7-0)
- [ ] T8a · T8b · T8c
- [ ] I1 (12 bộ detections)
- [ ] I2-a · I2-b · I2-c · I2-d
- [ ] I3-GMC · I3-ReID · I3-BIoU
- [ ] I4 dò tham số
- [ ] I5 đánh giá chéo miền
- [ ] I6 trích đặc trưng và t-SNE · I7 FPS · I8 video
- [ ] E0–E4 và 7 bảng kết quả
