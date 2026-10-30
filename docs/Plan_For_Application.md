# Kế hoạch Thực hiện Chi tiết – Định hướng Ứng dụng (Master Task Plan)

> **Đề tài:** Theo vết đa đối tượng (MOT) trong video drone và video giao thông.  
> **Tập dữ liệu chính:** VisDrone2019-MOT.  
> **Tập dữ liệu thực tế bổ sung:** Video tự quay tại Việt Nam (VN-Traffic-MOT: ≥ 200 đoạn ngắn 5–10 giây).  
> **Sản phẩm cốt lõi:** Ứng dụng Web hoàn chỉnh (Streamlit/Gradio/Web App) phục vụ người dùng không chuyên, tích hợp phân tích giao thông thông minh.  
> **Đường dẫn tài liệu:** `/media/trong-thanh/Data/University/Fourth Year/Computer Vision (Advance)/Final Project/Plan_Dinh_huong_Ung_dung.md`

---

## 1. Yêu cầu & Tiêu chí Đánh giá của Đề tài

### 1.1. Tiêu chí Chung cho Học phần Thị giác Máy tính Nâng cao
- **Quy trình chuẩn dự án AI:** Xác định bài toán $\rightarrow$ Thu thập & Chuẩn bị dữ liệu $\rightarrow$ Tiền xử lý & Trực quan hóa dữ liệu (EDA) $\rightarrow$ Huấn luyện & Tích hợp mô hình $\rightarrow$ Thực nghiệm & Đánh giá chuyên sâu $\rightarrow$ Đóng gói sản phẩm $\rightarrow$ Kiểm thử & Báo cáo.
- **Xử lý dữ liệu đầy đủ các bước bắt buộc:**
  1. *Tiền xử lý & Chuẩn hóa:* Tách frame, resize (640), chuẩn hóa pixel, lọc vùng ignored (`ignored regions`).
  2. *Tăng cường dữ liệu (Data Augmentation):* Mosaic, Random Horizontal Flip, HSV jitter, Random Scale.
  3. *Trực quan hóa phân phối (EDA):* Phân bố lớp, kích thước bounding box, mật độ vật thể/frame, độ dài tracklet, mức độ che khuất (occlusion), heatmap vị trí và so sánh VisDrone vs. Video tự quay.
  4. *Trích xuất đặc trưng & Giảm chiều:* Trích ReID embeddings bằng OSNet pretrained (không cần train lại), giảm chiều bằng **PCA** và **t-SNE** xuống không gian 2D, vẽ scatter plot phân cụm theo ID minh họa vì sao ReID giúp giữ ID khi bị che khuất.
- **Phương pháp tiếp cận:** Tối thiểu **3 mô hình phát hiện (Detector)** và **4 thuật toán theo dõi (Tracker)**, tạo ma trận thực nghiệm đánh giá chéo.
- **Hồ sơ bàn giao:** Báo cáo cuối kỳ, Slide thuyết trình, Video demo sản phẩm (2–3 phút), Mã nguồn hoàn chỉnh kèm tài liệu hướng dẫn cài đặt 1 bước.

### 1.2. Tiêu chí Đặc thù cho Định hướng Ứng dụng
- Xây dựng **sản phẩm phần mềm hoàn chỉnh** có giao diện trực quan thân thiện với người không có chuyên môn IT.
- Đảm bảo 3 chức năng cốt lõi: Nhận dữ liệu đầu vào $\rightarrow$ Chọn phương pháp/chế độ $\rightarrow$ Hiển thị & xuất kết quả.
- Tích hợp 2–3 chức năng nghiệp vụ giao thông thực tế: Đếm xe qua vạch (Line crossing), Bản đồ nhiệt mật độ (Heatmap), Thống kê lưu lượng theo thời gian.
- Kiểm thử đa tầng: Kiểm thử chức năng, Kiểm thử hiệu năng (GPU, CPU, NPU), Kiểm thử độ chính xác trên dữ liệu tự quay, và Khảo sát độ thỏa dụng người dùng (SUS testing).

---

## 2. Bảng Tổng Hợp Toàn Bộ Công Việc (Master WBS)

Dưới đây là danh sách toàn bộ các đầu việc cần thực hiện để hoàn thành đầy đủ 100% yêu cầu của đề tài:

```
[DỰ ÁN MOT VISDRONE & GIAO THÔNG]
 ├── 1. CHUẨN BỊ & XỬ LÝ DỮ LIỆU (D1 – D5, Preprocessing, EDA, t-SNE)
 ├── 2. HUẤN LUYỆN DETECTOR (YOLOv11, Faster R-CNN, RF-DETR, SAM-MOT)
 ├── 3. TÍCH HỢP BỘ THEO DÕI (ByteTrack, OC-SORT, DeepSORT, BoT-SORT)
 ├── 4. THỰC NGHIỆM & ĐÁNH GIÁ (I1 – I6, Benchmark GPU/CPU/NPU, Ma trận 3×4)
 ├── 5. PHÁT TRIỂN ỨNG DỤNG WEB (UI tiếng Việt, 4 tính năng cơ bản, 3 nghiệp vụ giao thông)
 ├── 6. ĐÓNG GÓI & KIỂM THỬ SẢN PHẨM (Run script/Docker, Test cases, SUS Survey)
 └── 7. BÁO CÁO & HỒ SƠ TỔNG KẾT (Báo cáo cuối kỳ, Slide, Video demo)
```

---

## 3. Chi tiết Công việc: Giai đoạn 1 – Dữ liệu & Phân tích Khám phá (EDA)

### 3.1. Chuẩn bị Dữ liệu (Tasks D1 – D5)

| Mã việc | Tên công việc | Chi tiết kỹ thuật & Quy cách thực hiện | Đầu ra (Deliverables) | Trạng thái |
|:---:|:---|:---|:---|:---:|
| **D1** | **Chuyển đổi nhãn VisDrone-MOT sang chuẩn YOLO** | • Xử lý tập `train` và `val` của VisDrone2019-MOT.<br>• Lấy mẫu **1/2 số frame của tập train** (bỏ frame chẵn hoặc lấy cách 2 frame do các frame liên tiếp gần trùng lặp, giảm tải thời gian train). Giữ 100% frame tập `val`.<br>• Lọc bỏ nhãn lớp `0` (ignored regions) và lớp `11` (others).<br>• Giữ 10 lớp gốc (1–10) hoặc ánh xạ sang định dạng chuẩn YOLO `(class_id, x_center, y_center, w, h)` chuẩn hóa $[0, 1]$. | `data/visdrone_yolo/`<br>`├── train/`<br>`└── val/` | Chưa xong |
| **D2** | **Chuyển đổi nhãn sang chuẩn COCO JSON** | • Chuyển đổi nhãn sang định dạng COCO format dùng cho Faster R-CNN (và thư viện MMDetection/Detectron2).<br>• Đồng bộ chính xác danh sách frame với tập D1. | `data/visdrone_coco/`<br>`├── train.json`<br>`└── val.json` | Chưa xong |
| **D3** | **Chuyển đổi Ground Truth sang MOTChallenge** | • Chuyển đổi nhãn ground truth sang chuẩn MOT (`<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>`).<br>• Tách thành các file đánh giá theo **5 lớp chính**: *pedestrian, car, van, truck, bus*.<br>• Tách riêng mask/bounding box của vùng `ignored` để nạp vào TrackEval lọc bỏ nhiễu. | `data/trackeval/gt/`<br>`VisDrone-{val,test}/`<br>`<seq>/gt/gt.txt` | Chưa xong |
| **D4** | **Thu thập & Gán nhãn Video Tự Quay (VN-Traffic)** | • Thu thập $\ge 200$ video ngắn (5–10s, độ phân giải 1080p/4K) quay giao thông thực tế tại Việt Nam bằng điện thoại/drone/flycam từ các góc nhìn trên cao.<br>• Tách frame và chọn ra **30–50 đoạn tiêu biểu** để gán nhãn chi tiết (bằng CVAT hoặc Label Studio) làm tập test thực tế ngoài đời.<br>• Các đoạn còn lại dùng làm video demo mẫu và kiểm thử ứng dụng. | `data/own/`<br>`├── labeled/`<br>`└── demo/` | Chưa xong |
| **D5** | **Đếm tay Ground Truth cho bài toán đếm xe** | • Thiết lập một vạch kẻ cố định trên khoảng **20 video tự quay**.<br>• Tiến hành đếm thủ công (ground truth) số lượng xe qua vạch, phân tách theo từng loại phương tiện và chiều di chuyển để đối sánh độ chính xác của ứng dụng. | `data/own/count_gt.csv` | Chưa xong |

### 3.2. Tiền xử lý, Tăng cường Dữ liệu & Trực quan hóa (EDA)

| Mã việc | Tên công việc | Nội dung chi tiết | Công cụ / Thư viện |
|:---:|:---|:---|:---|
| **P1** | **Tiền xử lý ảnh** | Tách frame từ video, resize về kích thước chuẩn $640 \times 640$, chuẩn hóa giá trị pixel $[0, 1]$, xử lý lọc bỏ các vùng ảnh bị đánh dấu ignore. | OpenCV, Albumentations |
| **P2** | **Tăng cường dữ liệu (Augmentation)** | Áp dụng Mosaic Augmentation (rất hiệu quả cho vật thể nhỏ), Random Horizontal Flip, HSV Color Jitter (đối phó với thay đổi ánh sáng ngoài trời), Random Scaling. | Ultralytics augment pipeline |
| **P3** | **EDA: Trực quan hóa phân phối tập dữ liệu** | Xây dựng các biểu đồ phân tích chuyên sâu đưa vào báo cáo:<br>1. *Phân bố lớp đối tượng:* Thể hiện sự mất cân bằng giữa xe con, người đi bộ, xe máy, xe tải.<br>2. *Phân bố kích thước Box:* Tỉ lệ vật thể nhỏ (small objects $< 32^2$ px) đặc trưng của drone.<br>3. *Mật độ đối tượng:* Số lượng box/frame (crowdedness level).<br>4. *Độ dài vết theo dõi:* Thời lượng một ID tồn tại trên video (tracklet duration).<br>5. *Mức độ che khuất (Occlusion):* Tỉ lệ đối tượng bị che khuất một phần hoặc toàn bộ.<br>6. *Bản đồ nhiệt vị trí (Spatial Heatmap):* Phân bố không gian của các đối tượng trên khung hình.<br>7. *So sánh Domain Gap:* Đối chiếu sự khác biệt giữa VisDrone chuẩn và video tự quay VN (đặc biệt là lưu lượng xe máy cực lớn tại VN). | Matplotlib, Seaborn, Pandas |
| **P4** | **Trích xuất đặc trưng ReID & Giảm chiều (t-SNE/PCA)** | • Sử dụng mạng **OSNet pretrained** (không cần train) trích xuất feature vector từ các crop ground truth của 20 ID khác nhau qua nhiều frame.<br>• Áp dụng thuật toán **PCA** và **t-SNE** giảm chiều từ vector đặc trưng 512 chiều xuống không gian 2D.<br>• Vẽ scatter plot 2D tô màu theo `target_id` để minh họa rõ: các crop cùng một ID sẽ cụm sát nhau, khác ID sẽ tách rời $\rightarrow$ cơ sở lý thuyết vì sao ReID giúp giữ ID sau khi bị che khuất. | PyTorch, Torchreid, scikit-learn |

---

## 4. Chi tiết Công việc: Giai đoạn 2 – Huấn luyện Detector & Tích hợp Tracker

### 4.1. Huấn luyện Detector & Mô hình mở rộng (Tasks T1 – T4, T1-n)

Tập trung huấn luyện, kiểm thử và tích hợp đúng danh mục các kiến trúc mô hình đã xác định:

| Mã | Mô hình | Khởi tạo / Backbone | Cấu hình huấn luyện khuyến nghị (có thể chỉnh theo GPU) | Mục đích sử dụng |
|:---:|:---|:---|:---|:---|
| **T1** | **YOLOv11m** *(YOLO11)* | Pretrained COCO (Ultralytics) | `imgsz=640`, `epochs=80`, `batch=16`, SGD/AdamW, `lr0=0.01`, mosaic on | Detector chính cho chế độ **Cân bằng (Balanced)** |
| **T1-n** | **YOLO11n** *(YOLOv11 Nano)* | Pretrained COCO | `imgsz=640`, `epochs=80`, `batch=32`, SGD, mosaic on | Detector siêu nhẹ dùng cho chế độ **Nhanh (Fast / CPU Mode)** |
| **T2** | **Faster R-CNN** | ResNet-50-FPN | Pretrained COCO, Lịch 2x (24 epochs), `lr=0.02 * (batch/16)` | Đại diện dòng Two-stage Detector chuẩn mực để so sánh |
| **T3** | **RF-DETR** *(RF Dert)* | Pretrained COCO | `imgsz=640`, `epochs=72`, `batch=8`, AdamW | SOTA Real-Time Transformer Detector cho chế độ **Chính xác (Accurate)** |
| **T4** | **SAM-MOT** *(SAMMOT - Mở rộng)* | SAM / MobileSAM / SAM 2 | Prompt-based Tracking & Segmentation | **Triển khai ở cuối kỳ nếu tìm được nhãn có segmentation** (hoặc dùng zero-shot SAM với box prompt từ YOLOv11/RF-DETR để mở rộng sang bài toán MOTS) |
| **T-Exp**| **Export ONNX / OpenVINO** | Checkpoint tốt nhất của T1, T1-n | Xuất sang ONNX, FP16, OpenVINO để tăng tốc độ inference trên CPU/NPU | Tối ưu triển khai ứng dụng thực tế |

*Câu lệnh mẫu tham chiếu:*
```bash
# 1. Huấn luyện YOLOv11m (T1 - Chế độ Cân bằng)
yolo detect train model=yolo11m.pt data=cfg/visdrone.yaml imgsz=640 epochs=80 batch=16 seed=0 project=runs/det name=T1_yolo11m

# 2. Huấn luyện YOLO11n (T1-n - Chế độ Nhanh chạy trên CPU)
yolo detect train model=yolo11n.pt data=cfg/visdrone.yaml imgsz=640 epochs=80 batch=32 seed=0 project=runs/det name=T1n_yolo11n

# 3. Huấn luyện RF-DETR (T3 - Chế độ Chính xác)
python train_rf_detr.py --config configs/rf_detr.yaml --dataset visdrone --epochs 72 --batch 8
```

### 4.2. Tích hợp Bộ Theo Dõi (Trackers) & Hướng mở rộng SAM-MOT
Sử dụng các tracker có sẵn trong thư viện `ultralytics` và `boxmot` (không cần train lại tracker), kèm phương án mở rộng SAM-MOT:

| Tên Tracker | Cơ chế hoạt động | Sử dụng ReID? | Ưu điểm & Ứng dụng | Thư viện tích hợp |
|:---|:---|:---:|:---|:---:|
| **ByteTrack** | Liên kết 2 tầng (high-score & low-score detections) dựa trên IoU và Kalman Filter | Không | Cực nhanh, phù hợp thời gian thực và máy cấu hình thấp | `ultralytics` |
| **OC-SORT** | Observation-Centric SORT, ước lượng lại vận tốc khi có quan sát mới | Không | Rất bền vững với chuyển động phi tuyến tính, camera drone bay rung lắc | `boxmot` |
| **DeepSORT** | Kalman filter kết hợp đặc trưng ngoại hình Appearance Embedding | Có (OSNet) | Hạn chế nhảy ID khi đối tượng bị khuất tạm thời | `boxmot` |
| **BoT-SORT** | Kết hợp ReID + Camera Motion Compensation (CMC dựa trên OpenCV ECC/GMC) + IoU cải tiến | Có (OSNet) | Chống rung lắc camera drone, giữ ID vượt trội trong các tình huống phức tạp | `ultralytics` / `boxmot` |
| **SAM-MOT (Mở rộng)** | Kết hợp theo dõi đa đối tượng với phân vùng mặt nạ (Multi-Object Tracking and Segmentation - MOTS) | Tùy chọn | Xuất mặt nạ phân vùng (instance mask) sắc nét của từng xe/người, triển khai cuối kỳ nếu có nhãn segmentation | `Segment Anything / SAM-MOT` |

---

## 5. Chi tiết Công việc: Giai đoạn 3 – Thực nghiệm & Đánh giá (Experiments)

### 5.1. Danh mục các lần chạy Thực nghiệm (Tasks I1 – I6)

| Mã | Nội dung thí nghiệm | Dữ liệu thực nghiệm | Mục tiêu & Đầu ra |
|:---:|:---|:---|:---|
| **I1** | **Chạy & Cache Detections** | VisDrone `val`, `test-dev`; video tự quay đã gán nhãn | Chạy detector T1 (YOLO11m), T1-n (YOLO11n), T2 (Faster R-CNN), T3 (RF-DETR) với ngưỡng confidence thấp (0.05). Lưu kết quả ra file cache (`.txt` hoặc `.json`) để mọi tracker dùng chung, đảm bảo so sánh công bằng và tiết kiệm 90% thời gian chạy. |
| **I2** | **Ma trận Thực nghiệm 3 Detector × 4 Tracker (12 lần chạy)** | Tập `val` $\rightarrow$ chọn ra 3 cấu hình tốt nhất $\rightarrow$ đánh giá trên `test-dev` | Đánh giá chéo toàn bộ ma trận (YOLO11m, Faster R-CNN, RF-DETR $\times$ ByteTrack, OC-SORT, DeepSORT, BoT-SORT) bằng bộ công cụ chuẩn TrackEval. |
| **I3** | **Đánh giá trên Video Tự Quay (Real-world Generalization)** | 30–50 video tự quay đã gán nhãn (D4) | Chạy cấu hình tốt nhất trên video thực tế Việt Nam để đo mức độ suy giảm hiệu năng do Domain Gap (góc máy thấp hơn, nhiều xe máy, biển báo khác). |
| **I4** | **Đo đạc tốc độ (FPS Benchmarking)** | 1 chuỗi video `val` cố định (chuẩn hóa số frame) | • Đo FPS end-to-end trên **GPU** (NVIDIA).<br>• Đo FPS end-to-end trên **CPU** (đa luồng).<br>• Đo FPS trên **NPU Qualcomm** (*Thành thực hiện nếu có phần cứng Snapdragon/NPU*). |
| **I5** | **Đánh giá Chức năng Nghiệp vụ Đếm Xe** | 20 video tự quay có số đếm tay (D5) | So sánh số lượng xe đếm được tự động của ứng dụng với số liệu đếm tay trong `count_gt.csv`. Tính toán sai số $\text{MAE} = \frac{\|\text{máy} - \text{tay}\|}{\text{tay}}$. |
| **I6** | **Trực quan hóa Phân cụm ReID t-SNE** | Crop ground truth của 20 ID từ VisDrone | Xuất biểu đồ phân cụm t-SNE phục vụ báo cáo và giao diện phân tích. |

### 5.2. Hệ thống Thước đo Đánh giá (Metrics)
* **Chỉ số Theo dõi (Tracking - TrackEval):**
  * $\text{HOTA}$ (Higher Order Tracking Accuracy) — chỉ số tổng hợp quan trọng nhất cân bằng giữa phát hiện và định danh.
  * $\text{MOTA}$ (Multiple Object Tracking Accuracy) — đo lường tổng hợp lỗi phát hiện và nhảy ID.
  * $\text{IDF1}$ (ID F1-score) — đo khả năng duy trì định danh xuyên suốt quỹ đạo.
  * $\text{IDSW}$ (ID Switches) — số lần thuật toán bị tráo đổi nhầm ID giữa các đối tượng.
  * $\text{MT / ML}$ (Mostly Tracked $\ge 80\%$ / Mostly Lost $\le 20\%$).
* **Chỉ số Phát hiện (Detection):** $\text{mAP@0.5}$, $\text{mAP@0.5:0.95}$.
* **Chỉ số Tốc độ:** $\text{FPS}$ end-to-end (bao gồm: đọc frame $\rightarrow$ detect $\rightarrow$ track $\rightarrow$ render), Độ trễ ($\text{Latency}$ tính bằng mili-giây).
* **Chỉ số Nghiệp vụ:** Sai số đếm xe tương đối ($\%$) theo từng phân lớp xe (ô tô, xe tải, xe máy).

### 5.3. Xây dựng Biểu đồ Đánh đổi (Trade-off) & Chốt Cấu hình Sản phẩm
Dựa trên biểu đồ phân tán **FPS – HOTA**, nhóm chốt các chế độ hoạt động chính tích hợp sẵn trên giao diện ứng dụng:
1. **Chế độ Nhanh (Fast Mode):** Ưu tiên tốc độ cao nhất (FPS $\ge 30$ trên GPU hoặc chạy mượt trên CPU máy tính bảng/laptop không GPU) $\rightarrow$ Cấu hình: `YOLO11n + ByteTrack`.
2. **Chế độ Cân bằng (Balanced Mode):** Cân bằng giữa độ chính xác và tốc độ khung hình hợp lý $\rightarrow$ Cấu hình: `YOLO11m + OC-SORT` (hoặc `ByteTrack`).
3. **Chế độ Chính xác (High Accuracy Mode):** Đạt điểm HOTA và IDF1 cao nhất, hạn chế tối đa nhảy ID trong điều kiện đông đúc $\rightarrow$ Cấu hình: `RF-DETR + BoT-SORT` (bật ReID và bù rung camera CMC).
4. **Chế độ Nâng cao / Mở rộng (Cuối kỳ - Advanced Segmentation):** Tích hợp `SAM-MOT` (kết hợp box từ YOLOv11/RF-DETR và Segment Anything) để vừa theo dõi vết, vừa hiển thị mặt nạ phân vùng pixel nếu tìm được nhãn segmentation.

---

## 6. Chi tiết Công việc: Giai đoạn 4 – Phát triển Sản phẩm Ứng dụng (Web App)

### 6.1. Bốn Nhóm Chức năng Cơ bản (Bắt buộc)
1. **Nhận dữ liệu đầu vào linh hoạt:**
   - [ ] Upload video từ máy tính (hỗ trợ các định dạng `.mp4`, `.avi`, `.mov`).
   - [ ] Quay trực tiếp bằng Webcam máy tính.
   - [ ] Kết nối luồng camera an ninh qua URL hoặc giao thức RTSP (tùy chọn mở rộng).
   - [ ] **Kho video mẫu có sẵn:** Cung cấp sẵn 3–5 video mẫu (giao thông drone, giao thông đường phố VN, cảnh quay đêm) để người dùng chỉ cần click chuột là chạy thử ngay.
2. **Lựa chọn phương pháp trực quan:**
   - [ ] Cho phép người dùng bấm chọn nhanh 1 trong 3 chế độ: **Nhanh / Cân bằng / Chính xác** (kèm tùy chọn chế độ mở rộng **SAM-MOT**).
   - [ ] Tab **Nâng cao (Advanced Settings):** Cho phép tự chọn thủ công loại Detector (**YOLO11m**, **YOLO11n**, **Faster R-CNN**, **RF-DETR**), loại Tracker (**ByteTrack**, **OC-SORT**, **DeepSORT**, **BoT-SORT**, và mô hình mở rộng **SAM-MOT**), điều chỉnh thanh trượt Confidence Threshold (ngưỡng tin cậy) và lọc chỉ theo dõi một số lớp phương tiện mong muốn.
3. **Hiển thị kết quả trực quan thời gian thực:**
   - [ ] Video playback mượt mà có Bounding Box, ID đối tượng, màu sắc phân biệt theo lớp (car, bus, truck, motor, pedestrian).
   - [ ] Hiển thị vệt quỹ đạo di chuyển (Trajectory tail) của từng xe.
   - [ ] Tùy chọn hiển thị Mặt nạ phân vùng (Segmentation Mask) khi bật chế độ mở rộng SAM-MOT.
   - [ ] Đồng hồ đo FPS tức thời và thanh tiến trình xử lý (Progress bar).
4. **Xuất kết quả đa định dạng:**
   - [ ] Tải video kết quả đã vẽ box và quỹ đạo dạng file `.mp4`.
   - [ ] Tải file kết quả định dạng chuẩn MOT (`.txt`).
   - [ ] Xuất dữ liệu quỹ đạo dạng `.json`.
   - [ ] Xuất bảng thống kê tổng hợp số lượng xe theo thời gian dạng `.csv`.

### 6.2. Các Chức năng Nghiệp vụ Giao thông Nâng cao (Chọn 3 chức năng chính)

| STT | Chức năng nghiệp vụ | Mô tả chi tiết triển khai | Độ ưu tiên |
|:---:|:---|:---|:---:|
| **1** | **Đếm phương tiện qua vạch (Line Crossing Counting)** | Người dùng vẽ 1 đường thẳng trực tiếp trên frame đầu tiên của video. Ứng dụng tự động phát hiện vector chuyển động cắt qua vạch, đếm số lượng xe theo từng lớp và phân biệt rõ 2 chiều di chuyển (Ví dụ: Chiều Bắc $\rightarrow$ Nam: 45 xe; Chiều Nam $\rightarrow$ Bắc: 38 xe). | **Cao (Bắt buộc)** |
| **2** | **Bản đồ nhiệt mật độ (Traffic Density Heatmap)** | Tích lũy tọa độ tâm của tất cả các phương tiện xuất hiện trong suốt video để tạo ra ảnh bản đồ nhiệt (Heatmap màu từ xanh đến đỏ). Giúp nhận diện trực quan các điểm đen ùn tắc giao thông trên tuyến đường. | **Cao (Bắt buộc)** |
| **3** | **Thống kê lưu lượng & Biểu đồ theo thời gian** | Vẽ biểu đồ biến thiên số lượng phương tiện theo từng giây/phút; bảng phân chia tỉ lệ phần trăm các loại phương tiện; ước lượng vận tốc di chuyển trung bình tương đối (pixels/giây). | **Cao (Bắt buộc)** |
| **4** | **So sánh song song 2 phương pháp (A/B Comparison)** | Chạy 2 cấu hình (ví dụ Nhanh vs. Chính xác) trên cùng 1 video và hiển thị 2 màn hình cạnh nhau (Side-by-Side) kèm so sánh FPS và số lần nhảy ID. | **Trung bình** |
| **5** | **Vùng quan tâm đặc biệt (ROI Alert)** | Cho phép người dùng vẽ một đa giác vùng cấm/vùng ưu tiên; hệ thống tự động phát hiện, đếm và cảnh báo khi có phương tiện di chuyển vào vùng này. | **Trung bình** |

### 6.3. Đảm bảo Tiêu chí Thân thiện với Người Không Chuyên
- **Giao diện 100% Tiếng Việt:** Bố cục rõ ràng, nút bấm to, thân thiện; mọi tham số đều có tooltip (icon `?`) giải thích ngắn gọn bằng ngôn ngữ đời thường (ví dụ: *"Ngưỡng phát hiện: Kéo cao để bớt bắt nhầm, kéo thấp để không bỏ sót xe ở xa"*).
- **Cài đặt 1 bước (One-click Setup):**
  - Viết sẵn script tự động: `run.sh` (Linux/Mac) và `run.bat` (Windows).
  - Đóng gói Dockerfile hoặc deploy trực tiếp lên Hugging Face Spaces để chỉ cần gửi link trình duyệt là thầy cô/người dùng xem được ngay.
- **Tự động xử lý ngoại lệ (Fault Tolerance):**
  - Thông báo lỗi rõ ràng khi người dùng upload sai định dạng file (ví dụ file văn bản, file nhạc) hoặc video quá dung lượng cho phép.
  - Tự động kích hoạt cơ chế giảm độ phân giải (downscale frame) khi phát hiện hệ thống đang chạy trên CPU yếu để tránh giật lag, treo ứng dụng.
- **Tài liệu hướng dẫn trực quan:** Tài liệu User Guide có hình ảnh chụp màn hình minh họa từng bước và **Video hướng dẫn sử dụng 2–3 phút**.

### 6.4. Kiến trúc Hệ thống Ứng dụng Web
```
Giao diện Người dùng (Streamlit / Gradio)
   │  [Upload video / Webcam / RTSP / Chọn chế độ / Vẽ vạch đếm xe / Chọn ROI]
   ▼
Lõi Xử lý Trung tâm (core/)
   ├── detector.py   YOLOv11 (11m, 11n) / Faster R-CNN / RF-DETR (trọng số fine-tune, export ONNX)
   ├── tracker.py    ByteTrack / OC-SORT / DeepSORT / BoT-SORT (+ ReID OSNet)
   ├── sam_mot.py    SAM-MOT (Tùy chọn mở rộng: Segment Anything tracking nếu có nhãn segmentation)
   ├── analytics.py  Đếm xe qua vạch 2 chiều, Heatmap mật độ, Thống kê lưu lượng, Cảnh báo ROI
   └── exporter.py   Xuất kết quả MOT txt, JSON quỹ đạo, CSV thống kê, MP4 video
   ▼
Thư mục lưu trữ kết quả: outputs/<session_id>/
```

---

## 7. Chi tiết Công việc: Giai đoạn 5 – Kiểm thử & Đóng gói (Testing & Packaging)

| Mã | Hạng mục kiểm thử | Kịch bản kiểm thử chi tiết | Tiêu chí nghiệm thu (Acceptance Criteria) |
|:---:|:---|:---|:---|
| **QA1** | **Kiểm thử Chức năng (Functional Test)** | • Test upload đầy đủ các định dạng video `.mp4`, `.avi`, `.mov`.<br>• Test upload file rác/file lỗi.<br>• Test chạy webcam thời gian thực.<br>• Test chức năng vẽ vạch đếm xe, xuất file kết quả `.csv`, `.mp4`. | 100% các ca kiểm thử đạt kết quả Pass, không crash app. |
| **QA2** | **Kiểm thử Hiệu năng & Tải (Stress Test)** | • Thử nghiệm với các đoạn video có độ dài khác nhau: 10 giây, 1 phút, 5 phút.<br>• Đo đạc dung lượng bộ nhớ RAM và VRAM chiếm dụng.<br>• Đo FPS trên CPU và GPU cho cả 3 chế độ. | Ứng dụng không bị tràn bộ nhớ (OOM), giải phóng bộ nhớ sạch sau mỗi phiên chạy. |
| **QA3** | **Kiểm thử Người dùng Không chuyên (SUS Survey)** | • Mời **5–10 người dùng không học chuyên ngành IT** trải nghiệm ứng dụng.<br>• Đưa ra nhiệm vụ cụ thể: *"Hãy mở web, tải video mẫu lên, chọn chế độ Cân bằng, vẽ vạch đếm số lượng xe ô tô và xuất file Excel thống kê"*; ghi nhận thời gian hoàn thành tác vụ.<br>• Cho người dùng điền bảng khảo sát độ thỏa dụng chuẩn quốc tế **SUS (System Usability Scale)** thang điểm 1–5. | Điểm trung bình đánh giá $\text{SUS} \ge 80/100$ điểm (mức Rất Tốt). |

---

## 8. Chi tiết Công việc: Giai đoạn 6 – Báo cáo, Hồ sơ & Trình chiếu Cuối kỳ

Cấu trúc báo cáo cuối kỳ được tổ chức chuẩn mực gồm **Phần Mở đầu (Giới thiệu)**, **3 Chương nội dung chính**, và **Phần Kết luận & Hướng phát triển**:

### PHẦN MỞ ĐẦU: GIỚI THIỆU ĐỀ TÀI
* Bối cảnh bài toán theo vết đa đối tượng (Multi-Object Tracking – MOT) trong video drone và camera giao thông.
* Mục tiêu nghiên cứu và mục tiêu sản phẩm ứng dụng hướng đến người dùng không chuyên.
* Ý nghĩa thực tiễn, đóng góp của đề tài và bố cục tài liệu báo cáo.

### CHƯƠNG 1: DỮ LIỆU VÀ PHƯƠNG PHÁP TIẾP CẬN
* **1.1. Khảo sát dữ liệu:** Mô tả chi tiết tập dữ liệu chuẩn VisDrone2019-MOT và tập video giao thông tự quay tại Việt Nam (VN-Traffic-MOT).
* **1.2. Tiền xử lý dữ liệu & Phân tích khám phá (EDA):** Tách frame, resize, chuẩn hóa, lọc vùng ignored, các kỹ thuật tăng cường dữ liệu (Augmentation); các biểu đồ phân tích phân bố lớp, kích thước box, mật độ vật thể, độ dài tracklet và mức độ che khuất.
* **1.3. Trích xuất đặc trưng ReID & Giảm chiều không gian:** Trích xuất vector đặc trưng bằng mạng OSNet pretrained, giảm chiều không gian bằng PCA và t-SNE, vẽ biểu đồ phân cụm minh họa lý do ReID giúp duy trì ID khi bị che khuất.
* **1.4. Các mô hình phát hiện đối tượng (Object Detectors):** Cơ sở lý thuyết, kiến trúc và thiết lập cấu hình của **YOLOv11** (bản YOLO11m cân bằng, YOLO11n cho CPU), **Faster R-CNN** (ResNet-50-FPN), và **RF-DETR** (Real-Time Detection Transformer).
* **1.5. Các thuật toán theo dõi đa đối tượng (Trackers) & Hướng mở rộng SAM-MOT:** Nguyên lý hoạt động của **ByteTrack, OC-SORT, DeepSORT, BoT-SORT**; hướng mở rộng phân vùng và theo dõi kết hợp với **SAM-MOT** (Segment Anything in MOT).
* **1.6. Kiến trúc hệ thống Pipeline:** Mô tả luồng tích hợp tổng thể từ nhận diện khung hình $\rightarrow$ trích xuất ReID $\rightarrow$ liên kết quỹ đạo đối tượng.

### CHƯƠNG 2: THỰC NGHIỆM VÀ ĐÁNH GIÁ MÔ HÌNH
* **2.1. Thiết lập môi trường & Quy trình thực nghiệm:** Cấu hình phần cứng huấn luyện, siêu tham số, quy trình chạy và lưu cache detections để đảm bảo so sánh công bằng.
* **2.2. Kết quả ma trận thực nghiệm 3 Detector × 4 Tracker:** Bảng kết quả định lượng đầy đủ các chỉ số chuẩn mực HOTA, MOTA, IDF1, IDSW, MT/ML trên tập validation và test-dev.
* **2.3. Phân tích đánh đổi Tốc độ vs. Độ chính xác (FPS – HOTA):** Kết quả đo đạc FPS end-to-end trên GPU, CPU, NPU Qualcomm; căn cứ khoa học để chốt 3 cấu hình phục vụ ứng dụng (Nhanh, Cân bằng, Chính xác).
* **2.4. Đánh giá khả năng tổng quát hóa trên dữ liệu thực tế Việt Nam:** Kết quả chạy trên 30–50 video tự quay, phân tích ảnh hưởng của Domain Gap (góc máy thấp, lưu lượng xe máy lớn).
* **2.5. Đánh giá chức năng nghiệp vụ đếm xe:** Đối sánh kết quả đếm tự động của ứng dụng với số liệu đếm tay (ground truth) trên 20 video, tính sai số tương đối MAE theo từng loại phương tiện.

### CHƯƠNG 3: THIẾT KẾ, CÀI ĐẶT ỨNG DỤNG VÀ KIỂM THỬ
* **3.1. Thiết kế kiến trúc phần mềm & Giao diện:** Thiết kế kiến trúc module hướng đối tượng `core/` (detector, tracker, sam_mot, analytics, exporter), thiết kế giao diện Web tiếng Việt thân thiện với người không có chuyên môn IT.
* **3.2. Cài đặt các chức năng cốt lõi:** Nhận 4 nguồn đầu vào (upload, webcam, RTSP, video mẫu 1-click), 3 chế độ định sẵn + tab nâng cao, hiển thị video trực quan kèm vệt quỹ đạo, xuất kết quả 4 định dạng (.mp4, .txt MOT, .json, .csv).
* **3.3. Cài đặt các chức năng nghiệp vụ thông minh:** Đếm xe qua vạch 2 chiều, tạo bản đồ nhiệt mật độ giao thông (Heatmap), vẽ biểu đồ thống kê lưu lượng theo thời gian.
* **3.4. Kiểm thử hệ thống (QA Testing):** Bảng kiểm thử chức năng (Functional Testing) và kiểm thử tải hiệu năng (Stress Testing trên CPU/GPU).
* **3.5. Đánh giá độ thỏa dụng người dùng (SUS Testing):** Kịch bản thực hiện nhiệm vụ và kết quả khảo sát người dùng không chuyên theo thang đo chuẩn quốc tế System Usability Scale (SUS).

### PHẦN KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
* Tổng kết toàn bộ các kết quả đạt được đối chiếu với các mục tiêu ban đầu.
* Các điểm hạn chế và thách thức kỹ thuật còn tồn đọng.
* Các hướng mở rộng trong tương lai: Triển khai SAM-MOT phân vùng mặt nạ đối tượng (MOTS), dự đoán quỹ đạo di chuyển (trajectory prediction), phát hiện va chạm hoặc vi phạm luật giao thông.

---

## 9. Bảng Checklist Nghiệm thu Toàn diện (Actionable Checklist)

### Pha 1: Dữ liệu (Data)
- [ ] Hoàn thành D1: Chuyển VisDrone-MOT sang YOLO format (1/2 frame train, lọc class 0 và 11).
- [ ] Hoàn thành D2: Chuyển sang COCO JSON format.
- [ ] Hoàn thành D3: Chuyển ground truth sang chuẩn MOTChallenge tách theo 5 lớp chính cho TrackEval.
- [ ] Hoàn thành D4: Thu thập $\ge 200$ video ngắn tự quay tại Việt Nam; gán nhãn xong 30–50 đoạn.
- [ ] Hoàn thành D5: Đếm tay số lượng xe qua vạch trên 20 video $\rightarrow$ lưu file `count_gt.csv`.
- [ ] Chạy pipeline EDA phân tích phân bố lớp, kích thước box, mật độ vật thể, độ dài track.
- [ ] Trích xuất vector đặc trưng OSNet ReID và vẽ biểu đồ phân cụm 2D bằng PCA & t-SNE.

### Pha 2: Mô hình & Thực nghiệm (Modeling & Evaluation)
- [ ] Huấn luyện xong T1 (**YOLOv11m**) đạt `best.pt`.
- [ ] Huấn luyện xong T1-n (**YOLO11n**) cho chế độ Nhanh trên CPU.
- [ ] Huấn luyện xong T2 (**Faster R-CNN** ResNet-50).
- [ ] Huấn luyện xong T3 (**RF-DETR**).
- [ ] Khảo sát / Triển khai mở rộng T4 (**SAM-MOT**) ở cuối kỳ nếu tìm được nhãn segmentation hoặc dùng box-prompt zero-shot.
- [ ] Xuất bản ONNX / OpenVINO cho T1 và T1-n.
- [ ] Chạy I1: Lưu cache detections cho các detector với ngưỡng conf = 0.05.
- [ ] Chạy I2: Thực nghiệm ma trận Detector $\times$ Tracker (**YOLO11m, Faster R-CNN, RF-DETR** $\times$ **ByteTrack, OC-SORT, DeepSORT, BoT-SORT**).
- [ ] Chạy I3: Đánh giá mô hình tốt nhất trên tập video tự quay tại Việt Nam.
- [ ] Chạy I4: Đo FPS end-to-end trên GPU, CPU (và NPU Qualcomm nếu Thành có thời gian).
- [ ] Chạy I5: Đo sai số đếm xe so với số đếm tay D5.
- [ ] Vẽ biểu đồ FPS – HOTA và chốt 3 cấu hình chính thức cho ứng dụng (Nhanh, Cân bằng, Chính xác).

### Pha 3: Sản phẩm Web App & Kiểm thử (Application & QA)
- [ ] Hoàn thành 4 nguồn đầu vào: Upload video, Webcam, RTSP/URL camera, Video mẫu chọn sẵn.
- [ ] Hoàn thành 3 chế độ chạy nhanh + Tab tùy chỉnh nâng cao (Confidence, chọn lớp).
- [ ] Hiển thị video có Bounding Box, ID, vệt quỹ đạo và đồng hồ đo FPS thời gian thực.
- [ ] Xuất đủ 4 định dạng đầu ra: video `.mp4`, file MOT `.txt`, file `.json`, bảng dữ liệu `.csv`.
- [ ] Hoàn thành Chức năng 1: Đếm xe qua vạch 2 chiều.
- [ ] Hoàn thành Chức năng 2: Bản đồ nhiệt mật độ (Heatmap).
- [ ] Hoàn thành Chức năng 3: Biểu đồ thống kê lưu lượng xe theo thời gian.
- [ ] Giao diện tiếng Việt 100%, có tooltip giải thích rõ ràng cho người không chuyên.
- [ ] Cơ chế xử lý lỗi: báo định dạng sai, tự giảm độ phân giải khi chạy CPU yếu.
- [ ] Đóng gói 1 bước: Viết script `run.sh`, `run.bat` (hoặc Docker / Hugging Face Spaces).
- [ ] Thực hiện kiểm thử chức năng và kiểm thử tải hiệu năng.
- [ ] Thực hiện khảo sát người dùng không chuyên theo thang đo chuẩn SUS ($\ge 5$ người tham gia).

### Pha 4: Hồ sơ Nghiệm thu Cuối kỳ (Deliverables)
- [ ] Hoàn thiện Báo cáo tổng kết đồ án cuối kỳ (Phần Mở đầu, 3 Chương nội dung chính, Phần Kết luận).
- [ ] Hoàn thiện Bộ Slide thuyết trình đồ án chuyên nghiệp.
- [ ] Quay và dựng Video demo sản phẩm chất lượng cao (2–3 phút) có thuyết minh.
- [ ] Làm sạch mã nguồn, cấu trúc thư mục chuẩn và hoàn thiện tài liệu `README.md` hướng dẫn sử dụng.
