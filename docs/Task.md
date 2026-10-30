# Đặc tả Input / Output Chi tiết cho Từng Task – Định hướng Ứng dụng

> Tài liệu đi kèm `Plan_Dinh_huong_Ung_dung.md`. Mỗi task được mô tả theo khung:
> **Phụ thuộc → Đầu vào → Xử lý → Đầu ra → Tiêu chí xong (DoD)**.
> Mọi đường dẫn tính từ thư mục gốc dự án `mot_project/` (thư mục mã nguồn).

---

## 0. Quy ước chung (áp dụng cho tất cả task)

### 0.1. Dữ liệu gốc VisDrone2019-MOT (đầu vào thô của toàn dự án)

```
data/raw/
├── VisDrone2019-MOT-train/     # 56 chuỗi, ~24.201 frame
│   ├── sequences/<seq>/0000001.jpg, 0000002.jpg, ...
│   └── annotations/<seq>.txt
├── VisDrone2019-MOT-val/       # 7 chuỗi, ~2.846 frame
└── VisDrone2019-MOT-test-dev/  # 17 chuỗi, ~6.635 frame
```

- Mỗi dòng trong `annotations/<seq>.txt` có 10 cột:
  `frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion`
  - `frame_index`: bắt đầu từ 1, khớp tên file ảnh `0000001.jpg`.
  - `score`: trong GT, `1` = được tính khi đánh giá, `0` = bỏ qua.
  - `object_category`: 0 ignored region · 1 pedestrian · 2 people · 3 bicycle · 4 car · 5 van · 6 truck · 7 tricycle · 8 awning-tricycle · 9 bus · 10 motor · 11 others.
  - `truncation`: 0 không bị cắt · 1 bị cắt một phần (1–50%).
  - `occlusion`: 0 không che · 1 che một phần (1–50%) · 2 che nhiều (50–100%).
- Độ phân giải ảnh không cố định (từ khoảng 1344×756 đến 3840×2160). Phải đọc kích thước thật của từng ảnh khi chuẩn hóa toạ độ.

### 0.2. Danh sách lớp thống nhất (dùng xuyên suốt)

| VisDrone id | Tên | YOLO id (D1) | COCO category_id (D2) | Lớp đánh giá tracking (D3) |
|:-:|:-|:-:|:-:|:-:|
| 1 | pedestrian | 0 | 1 | ✓ |
| 2 | people | 1 | 2 | |
| 3 | bicycle | 2 | 3 | |
| 4 | car | 3 | 4 | ✓ |
| 5 | van | 4 | 5 | ✓ |
| 6 | truck | 5 | 6 | ✓ |
| 7 | tricycle | 6 | 7 | |
| 8 | awning-tricycle | 7 | 8 | |
| 9 | bus | 8 | 9 | ✓ |
| 10 | motor | 9 | 10 | ✓ *(chỉ với dữ liệu VN – xem I3)* |

→ Lưu bảng này thành `cfg/classes.yaml` để mọi script đọc chung (không hard-code).

### 0.3. Định dạng trung gian dùng lại nhiều lần

| Tên | Định dạng 1 dòng | Dùng ở |
|:-|:-|:-|
| **DET cache** | `frame,-1,x,y,w,h,score,class` (x,y = góc trái trên, pixel ảnh gốc, `class` = YOLO id 0–9) | I1 → I2, I3, I4 |
| **TRACK (MOT + class)** | `frame,id,x,y,w,h,conf,class,-1,-1` | I2, I3, app export |
| **Tracker API nội bộ** | vào: `dets` = `np.ndarray (N,6)` `[x1,y1,x2,y2,conf,cls]` + `frame` BGR `(H,W,3) uint8`; ra: `np.ndarray (M,7)` `[x1,y1,x2,y2,track_id,conf,cls]` | `core/tracker.py` |

### 0.4. Nhật ký chạy

Mọi task có train/inference ghi thêm 1 dòng vào `results/runs_log.csv`:
`run_id, ngay, nguoi_chay, task_id, model, data, tham_so_chinh, gpu, thoi_gian, chi_so_chinh, ghi_chu`.

---

## 1. GIAI ĐOẠN 1 – DỮ LIỆU (D1–D5)

### D1 – Chuyển nhãn VisDrone-MOT sang YOLO

- **Phụ thuộc:** không có (task đầu tiên).
- **Đầu vào:**
  - `data/raw/VisDrone2019-MOT-train/` và `data/raw/VisDrone2019-MOT-val/` (ảnh + annotations, mục 0.1).
  - `cfg/classes.yaml` (bảng ánh xạ mục 0.2).
  - Tham số: `train_stride=2` (lấy 1/2 frame), `val_stride=1`.
- **Xử lý:**
  1. Với mỗi chuỗi train: chỉ giữ frame có `frame_index % 2 == 1` (frame lẻ). Val giữ toàn bộ.
  2. Bỏ các dòng có `object_category ∈ {0, 11}` hoặc `score == 0`.
  3. Đổi toạ độ: `xc = (left + w/2)/W`, `yc = (top + h/2)/H`, `wn = w/W`, `hn = h/H`; cắt (clip) về `[0,1]`; bỏ box có `w<2` hoặc `h<2` px.
  4. Đổi lớp: `yolo_id = visdrone_id - 1`.
  5. *(Gộp với P1)* Tô xám (giá trị 114) các vùng `object_category == 0` trên ảnh train để mô hình không bị phạt khi phát hiện đối tượng trong vùng ignored.
  6. Đặt tên phẳng `<seq>_<frame:07d>.jpg` để không trùng tên giữa các chuỗi.
- **Đầu ra:**
  ```
  data/visdrone_yolo/
  ├── train/images/<seq>_<frame>.jpg     # ~12.100 ảnh
  ├── train/labels/<seq>_<frame>.txt     # mỗi dòng: class xc yc w h (chuẩn hóa)
  ├── val/images/  val/labels/           # ~2.846 ảnh
  └── frame_list_{train,val}.txt         # danh sách frame đã chọn → D2 dùng lại
  cfg/visdrone.yaml                       # path, train, val, nc: 10, names: [...]
  ```
  Ví dụ một dòng label: `3 0.512344 0.433102 0.021875 0.034259`
- **DoD:** số ảnh = số file label; không có toạ độ ngoài `[0,1]`; vẽ ngẫu nhiên 3 chuỗi bằng OpenCV, kiểm tra box khớp vật thể bằng mắt; `yolo detect val model=yolo11n.pt data=cfg/visdrone.yaml` chạy được không lỗi.

### D2 – Chuyển nhãn sang COCO JSON

- **Phụ thuộc:** D1 (dùng `frame_list_{train,val}.txt`).
- **Đầu vào:** annotations gốc + danh sách frame của D1 + `cfg/classes.yaml`.
- **Xử lý:** tạo 3 khối `images` (`id, file_name, width, height`), `annotations` (`id, image_id, category_id, bbox=[x,y,w,h]` pixel, `area=w*h`, `iscrowd=0`), `categories` (10 lớp, `id` 1–10). Có thể ghi thêm `track_id`, `occlusion` vào annotation để EDA dùng lại.
- **Đầu ra:**
  ```
  data/visdrone_coco/
  ├── train.json   val.json            # Faster R-CNN (torchvision / MMDetection)
  └── rfdetr/                          # RF-DETR (thư viện rfdetr) cần cấu trúc kiểu Roboflow:
      ├── train/_annotations.coco.json + ảnh (symlink)
      ├── valid/_annotations.coco.json + ảnh
      └── test/_annotations.coco.json  (có thể trỏ = valid)
  ```
- **DoD:** `pycocotools.COCO('val.json')` load được; số box của D2 bằng đúng số dòng label của D1; `file_name` trỏ đúng ảnh của D1.

> ⚠ **Lưu ý:** trong plan, T3 (RF-DETR) được ghi là dùng `cfg/visdrone.yaml` (YOLO), nhưng thư viện `rfdetr` chủ yếu nhận dữ liệu **COCO**. Vì vậy D2 phải xuất thêm thư mục `rfdetr/` như trên.

### D3 – Chuyển Ground Truth sang MOTChallenge cho TrackEval

- **Phụ thuộc:** không có (đọc trực tiếp từ dữ liệu gốc).
- **Đầu vào:** annotations gốc của **val** và **test-dev** (giữ 100% frame, không lấy mẫu).
- **Xử lý:**
  1. Với mỗi lớp `c ∈ {pedestrian, car, van, truck, bus}` tạo một "benchmark" riêng: `VisDrone-val-<c>`, `VisDrone-test-<c>`.
  2. File `gt.txt` của lớp `c` chỉ gồm các box thuộc lớp `c`, theo định dạng TrackEval MOT 9 cột: `frame,id,x,y,w,h,1,1,visibility` (cột lớp ghi `1` = pedestrian vì TrackEval MOT chỉ đánh giá lớp này; `visibility` = 1.0 / 0.75 / 0.25 tương ứng occlusion 0/1/2).
  3. Ghi các vùng `category==0` (và box `score==0`) vào `gt.txt` dưới lớp **distractor (8)**, để TrackEval tự loại bỏ các dự đoán rơi vào vùng này khi `DO_PREPROC=True`.
  4. Mỗi chuỗi có `seqinfo.ini` (`name, imDir=img1, frameRate=30, seqLength, imWidth, imHeight, imExt=.jpg`).
  5. Có file seqmap liệt kê các chuỗi.
- **Đầu ra:**
  ```
  data/trackeval/gt/
  ├── VisDrone-val-car/<seq>/gt/gt.txt
  ├── VisDrone-val-car/<seq>/seqinfo.ini
  ├── ... (5 lớp × 2 split = 10 benchmark)
  └── seqmaps/VisDrone-val-car.txt ...
  data/trackeval/ignored/<split>/<seq>.txt     # frame,x,y,w,h – bản lưu riêng vùng ignored
  ```
- **DoD:** chấm điểm chính GT (dùng GT làm tracker output) phải ra HOTA ≈ 100, MOTA ≈ 100. Đây là cách test nhanh nhất để chắc chắn D3 đúng.

### D4 – Thu thập & gán nhãn video tự quay VN-Traffic

- **Phụ thuộc:** không có (chạy song song với D1–D3). Cần `cfg/classes.yaml` để thống nhất danh sách lớp gán nhãn.
- **Đầu vào:**
  - Thiết bị: điện thoại / drone / flycam, quay từ trên cao (cầu vượt, tầng cao, drone hover).
  - ≥ 200 clip, mỗi clip 5–10 s, 1080p hoặc 4K, 25–30 fps.
  - Bảng mô tả bối cảnh do người quay điền: địa điểm, giờ, ngày/đêm, thời tiết, độ cao/góc nhìn, camera tĩnh hay chuyển động.
  - Công cụ gán nhãn: CVAT (khuyến nghị, vì có nội suy track) hoặc Label Studio.
- **Xử lý:**
  1. Đặt tên chuẩn `vn_<YYYYMMDD>_<địa điểm>_<stt>.mp4`.
  2. Chọn 30–50 clip đa dạng (ngày/đêm, đông/thưa, góc cao/thấp) để gán nhãn. Gán theo **track** (giữ ID xuyên suốt clip); dùng keyframe cách 5–10 frame rồi nội suy, sau đó chỉnh lại.
  3. Lớp gán nhãn: dùng chính 10 lớp VisDrone (đặc biệt là `motor`, `car`, `bus`, `truck`, `pedestrian`). Vùng quá đông hoặc quá xa thì đánh dấu là `ignored`.
  4. Xuất từ CVAT ở định dạng "MOT 1.1", rồi dùng script đổi sang (a) MOT + class, (b) YOLO, (c) cấu trúc TrackEval giống D3.
- **Đầu ra:**
  ```
  data/own/
  ├── raw/*.mp4                          # toàn bộ ≥200 clip gốc
  ├── metadata.csv                       # video_id, dia_diem, gio, ngay_dem, thoi_tiet, goc_quay, camera_tinh, fps, W, H, so_frame, dung_cho(labeled/demo)
  ├── labeled/<video_id>/
  │   ├── img1/000001.jpg ...            # frame đã tách
  │   ├── gt/gt.txt                      # frame,id,x,y,w,h,1,class,visibility
  │   └── seqinfo.ini
  ├── labeled_yolo/{images,labels}/      # để EDA / có thể fine-tune sau
  ├── trackeval/gt/VN-test-<class>/...   # giống cấu trúc D3, thêm lớp motor
  └── demo/*.mp4                         # các clip còn lại → kho video mẫu của app và QA
  ```
- **DoD:** ≥ 200 clip có trong `metadata.csv`; 30–50 clip có `gt.txt`; mở lại bằng script vẽ, ID không nhảy giữa các frame; nếu có 2 người gán nhãn thì đo độ đồng thuận giữa hai người trên 3 clip.

### D5 – Đếm tay Ground Truth cho bài toán đếm xe

- **Phụ thuộc:** D4 (cần các clip). Nên chọn 20 clip quay bằng **camera tĩnh** (xem lưu ý ở mục 5.2).
- **Đầu vào:** 20 clip từ `data/own/` + công cụ xem từng frame (VLC / script OpenCV có phím tua chậm).
- **Xử lý:**
  1. Định nghĩa vạch đếm bằng 2 điểm pixel `(x1,y1)-(x2,y2)` trên frame đầu và lưu lại. **Đây cũng chính là vạch mà I5 sẽ dùng.**
  2. Định nghĩa chiều: `A→B` = đi từ phía bên trái sang phía bên phải của vector `(x1,y1)→(x2,y2)` (theo dấu tích có hướng). Ghi tên dễ hiểu, ví dụ "Bắc→Nam".
  3. Đếm tay theo lớp × chiều. Nên có 2 người đếm độc lập, nếu lệch thì xem lại.
- **Đầu ra:**
  ```
  data/own/count_lines.json   # {"<video_id>": {"p1":[x,y], "p2":[x,y], "dir_A_name":"Bắc→Nam", "dir_B_name":"Nam→Bắc"}}
  data/own/count_gt.csv       # video_id, class, direction(A|B), count, nguoi_dem, ghi_chu
  ```
  Ví dụ: `vn_20261020_cauvuot_03, motor, A, 57, Thanh, "2 xe đi sát vạch"`
- **DoD:** 20 video × đủ các lớp xuất hiện × 2 chiều; mỗi video có đúng một vạch trong `count_lines.json`.

---

## 2. GIAI ĐOẠN 1 (tiếp) – TIỀN XỬ LÝ, AUGMENTATION, EDA, ReID (P1–P4)

### P1 – Tiền xử lý ảnh

- **Phụ thuộc:** D1 (bước tô xám ignored đã gộp vào D1).
- **Đầu vào:** frame BGR `uint8 (H,W,3)` từ ảnh dataset hoặc từ video (`cv2.VideoCapture`); danh sách vùng ignored (nếu có).
- **Xử lý:** đây là module dùng chung cho cả train, inference và app:
  - `extract_frames(video_path, out_dir, stride)` → tách frame.
  - `letterbox(img, 640)` → resize giữ tỉ lệ + pad (114), trả về ảnh cùng `ratio, pad` để đổi toạ độ ngược lại.
  - `normalize(img)` → `float32 [0,1]`, đổi BGR→RGB, HWC→CHW.
  - `mask_ignored(img, regions)`.
- **Đầu ra:** `core/preprocess.py` (các hàm trên) + `scripts/extract_frames.py`; ảnh mẫu trước/sau tiền xử lý `results/figures/p1_preprocess_demo.png` để đưa vào báo cáo mục 1.2.
- **DoD:** đổi một box qua letterbox rồi đổi ngược lại thì sai số ≤ 1 px.

> ⚠ Không resize vật lý toàn bộ dataset về 640×640. Ultralytics, RF-DETR và Faster R-CNN đều tự letterbox/resize khi train. Resize trước sẽ làm mất chi tiết của vật thể nhỏ.

### P2 – Tăng cường dữ liệu (Augmentation)

- **Phụ thuộc:** D1, D2.
- **Đầu vào:** `cfg/visdrone.yaml` + bộ tham số augmentation:
  `mosaic=1.0, close_mosaic=10, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5, translate=0.1`.
  Faster R-CNN: RandomHorizontalFlip(0.5) + multi-scale (short side 640–800) + ColorJitter. RF-DETR: dùng augmentation mặc định của thư viện.
- **Đầu ra:**
  - `cfg/aug_yolo.yaml`, `cfg/aug_frcnn.py` (ghi lại để tái lập).
  - Ảnh minh họa: `results/figures/p2_aug_mosaic.png`, `p2_aug_flip.png`, `p2_aug_hsv.png`, `p2_aug_scale.png` (1 ảnh gốc → 4 biến thể, vẽ cả box) và `train_batch0.jpg` do Ultralytics tự sinh.
- **DoD:** box trên ảnh augment vẫn khớp vật thể.

### P3 – EDA: trực quan hóa phân phối dữ liệu

- **Phụ thuộc:** D1 (hoặc annotations gốc), D4 (để so sánh domain).
- **Đầu vào:** annotations gốc train/val/test-dev (dùng **100% frame**, không lấy mẫu) + `data/own/labeled/*/gt/gt.txt` + `metadata.csv`.
- **Xử lý → đầu ra** (`results/figures/eda/` và `results/tables/eda/`):

| # | Biểu đồ | Cách tính | File đầu ra |
|:-:|:-|:-|:-|
| 1 | Phân bố lớp | đếm box theo lớp × split (cột nhóm, trục y log) | `eda_class_dist.png`, `class_counts.csv` |
| 2 | Kích thước box | histogram `sqrt(w*h)`; tỉ lệ small (<32²), medium (32²–96²), large (>96²) theo chuẩn COCO | `eda_box_size.png`, `box_size_stats.csv` |
| 3 | Mật độ | số box/frame (histogram + boxplot theo chuỗi) | `eda_density.png` |
| 4 | Độ dài tracklet | số frame mỗi `target_id` tồn tại, theo lớp | `eda_tracklet_len.png` |
| 5 | Occlusion | tỉ lệ occlusion 0/1/2 theo lớp (cột chồng) | `eda_occlusion.png` |
| 6 | Heatmap vị trí | histogram 2D của tâm box sau khi chuẩn hóa về `[0,1]` | `eda_spatial_heatmap.png` |
| 7 | Domain gap VisDrone vs VN | ghép 1, 2, 3 cho hai tập lên cùng một hình (chú ý tỉ lệ `motor`) | `eda_domain_gap.png`, `domain_compare.csv` |

  Notebook nguồn: `notebooks/03_eda.ipynb` (hoặc `scripts/eda.py`).
- **DoD:** đủ 7 hình, mỗi hình có tiêu đề, nhãn trục tiếng Việt và 1–2 câu nhận xét để đưa vào báo cáo mục 1.2.

### P4 – Trích đặc trưng ReID OSNet & giảm chiều (PCA / t-SNE)

- **Phụ thuộc:** annotations gốc (val hoặc train).
- **Đầu vào:**
  - 20 `target_id` chọn từ 2–3 chuỗi, gồm cả car và pedestrian. Mỗi ID lấy 10–20 crop, các crop cách nhau ≥ 5 frame, cạnh ≥ 16 px. **Cố ý lấy cả các frame có `occlusion ≥ 1`** để làm rõ luận điểm "ReID giữ được ID khi bị che".
  - Trọng số OSNet pretrained: `osnet_x1_0` (Market-1501/MSMT17) cho người; nếu có thì dùng thêm bản VeRi cho xe. Có thể lấy từ `torchreid` hoặc file `.pt` của `boxmot`.
- **Xử lý:** crop → resize 256×128 (hoặc 128×128 cho xe nhìn từ trên) → normalize ImageNet → OSNet → vector 512 chiều → chuẩn hóa L2 → PCA (2D, và 50D làm đầu vào cho t-SNE) → t-SNE (`perplexity=30`, `random_state=0`).
- **Đầu ra:**
  ```
  results/features/reid/
  ├── crops/<id>/<seq>_<frame>.jpg
  ├── embeddings.npy        # (N, 512) float32
  ├── meta.csv              # idx, seq, frame, target_id, class, occlusion
  ├── pca_2d.npy  tsne_2d.npy
  └── silhouette.txt        # điểm silhouette theo ID: con số định lượng cho mức "cụm tốt"
  ```
- **DoD:** có đủ file embedding + meta để I6 vẽ hình mà không phải chạy lại mô hình.

---

## 3. GIAI ĐOẠN 2 – HUẤN LUYỆN DETECTOR (T1, T1-n, T2, T3, T4, T-Exp)

### T1 – YOLO11m (chế độ Cân bằng)

- **Phụ thuộc:** D1, P2.
- **Đầu vào:** `data/visdrone_yolo/`, `cfg/visdrone.yaml`, trọng số `yolo11m.pt` (COCO), GPU ≥ 12 GB.
  Lệnh: `yolo detect train model=yolo11m.pt data=cfg/visdrone.yaml imgsz=640 epochs=80 batch=16 seed=0 project=runs/det name=T1_yolo11m`.
- **Đầu ra:**
  ```
  runs/det/T1_yolo11m/
  ├── weights/best.pt  last.pt
  ├── args.yaml  results.csv  results.png
  ├── confusion_matrix.png  PR_curve.png  F1_curve.png
  └── val_batch*_pred.jpg
  ```
  Chỉ số cần ghi: mAP50, mAP50-95 trên val (toàn bộ và theo từng lớp), thời gian train.
- **DoD:** mAP50 trên val cao hơn rõ ràng so với YOLO11m COCO zero-shot; đường loss hội tụ.

### T1-n – YOLO11n (chế độ Nhanh / CPU)

- **Đầu vào/ra:** như T1, thay `model=yolo11n.pt`, `batch=32`, `name=T1n_yolo11n` → `runs/det/T1n_yolo11n/weights/best.pt`.
- **DoD:** có `best.pt`; mAP thấp hơn T1 nhưng chạy nhanh hơn nhiều trên CPU (kiểm chứng ở I4).

### T2 – Faster R-CNN ResNet-50-FPN

- **Phụ thuộc:** D2.
- **Đầu vào:** `data/visdrone_coco/{train,val}.json` + ảnh; trọng số COCO (`torchvision fasterrcnn_resnet50_fpn_v2` hoặc config MMDetection `faster-rcnn_r50_fpn_2x_coco`); `num_classes = 11` (10 lớp + background nếu dùng torchvision); lịch 24 epoch, `lr = 0.02 × batch/16`, giảm lr tại epoch 16 và 22.
- **Đầu ra:** `runs/det/T2_frcnn_r50/` gồm `best.pth`, `config.py`, `log.json`, `metrics_val.json` (COCO mAP, mAP-small/medium/large).
- **DoD:** chạy được hàm `predict(image) → (N,6)` thống nhất với `core/detector.py`.

### T3 – RF-DETR

- **Phụ thuộc:** D2 (thư mục `rfdetr/`).
- **Đầu vào:** `data/visdrone_coco/rfdetr/`, pretrained COCO (`RFDETRBase` / `RFDETRMedium`), `epochs=72, batch=8` (giảm batch và dùng `grad_accum_steps` nếu thiếu VRAM), `resolution` là bội số của 56 (ví dụ 560 hoặc 672) theo yêu cầu thư viện.
- **Đầu ra:** `runs/det/T3_rfdetr/` gồm `checkpoint_best_total.pth`, `log.txt`, `metrics_val.json`.
- **DoD:** như T2.

### T4 – SAM-MOT (mở rộng, cuối kỳ)

- **Phụ thuộc:** T1 hoặc T3 (cấp box prompt) + I2 (cấp track).
- **Đầu vào:** frame + box của track `[x1,y1,x2,y2,id]` mỗi frame; trọng số SAM (`sam_vit_b`), MobileSAM, hoặc SAM 2 (`sam2_hiera_tiny`); nếu muốn đo định lượng thì cần nhãn segmentation.
- **Xử lý:** zero-shot: mỗi box của track → SAM → mask nhị phân; với SAM 2 có thể lan truyền mask theo thời gian.
- **Đầu ra:** `results/masks/<video>/` theo định dạng **MOTS txt** (`frame id class H W rle`) + video overlay mask; module `core/sam_mot.py` với API `segment(frame, tracks) → list[mask (H,W) bool]`.
- **DoD:** chạy được trên ít nhất 1 video demo; đo FPS; nếu không có nhãn segmentation thì chỉ đánh giá định tính.

### T-Exp – Export ONNX / OpenVINO

- **Phụ thuộc:** T1, T1-n.
- **Đầu vào:** `best.pt` của T1 và T1-n.
  Lệnh: `yolo export model=best.pt format=onnx imgsz=640 dynamic=False simplify=True` và `yolo export model=best.pt format=openvino imgsz=640 half=True` (thêm `int8=True data=cfg/visdrone.yaml` nếu muốn INT8).
- **Đầu ra:** `best.onnx`, `best_openvino_model/{best.xml, best.bin, metadata.yaml}`; bảng `results/tables/export_check.csv`: `model, backend, precision, mAP50, mAP50-95, fps_cpu`.
- **DoD:** mAP sau export lệch ≤ 1 điểm so với `.pt`; FPS CPU tăng. Với NPU Qualcomm: cần ONNX đã lượng tử hóa QDQ, chạy bằng ONNX Runtime QNN EP (chỉ làm nếu có máy Snapdragon).

---

## 4. GIAI ĐOẠN 2 (tiếp) – TÍCH HỢP TRACKER

### TRK – Module `core/tracker.py` (ByteTrack, OC-SORT, DeepSORT, BoT-SORT)

- **Phụ thuộc:** không phụ thuộc mô hình (test được bằng DET cache của bất kỳ detector nào).
- **Đầu vào:** mỗi frame: `dets (N,6)` + `frame` BGR (DeepSORT/BoT-SORT cần ảnh để trích ReID và CMC) + file cấu hình `cfg/trackers/<name>.yaml` (`track_high_thresh, track_low_thresh, new_track_thresh, match_thresh, track_buffer, with_reid, reid_weights, cmc_method`).
- **Xử lý:** lớp chung `Tracker.update(dets, frame) → (M,7)` bọc thư viện:
  - ByteTrack, OC-SORT, BoT-SORT: dùng `boxmot` (một thư viện cho cả 3, giúp so sánh công bằng).
  - DeepSORT: `boxmot` **không có DeepSORT bản gốc**. Dùng `deep_sort_realtime`, hoặc dùng `StrongSORT` của boxmot và ghi rõ trong báo cáo là "DeepSORT-family".
  - ReID: `osnet_x0_25_msmt17.pt` hoặc `osnet_x1_0` (cùng trọng số với P4).
- **Đầu ra:** `core/tracker.py`, `cfg/trackers/*.yaml`, script `scripts/run_tracker.py --dets <cache_dir> --tracker <name> --split val --out results/tracks/...`.
- **DoD:** chạy được 4 tracker trên 1 chuỗi val từ DET cache; ID ổn định khi xem bằng mắt.

---

## 5. GIAI ĐOẠN 3 – THỰC NGHIỆM & ĐÁNH GIÁ (I1–I6, Trade-off)

### I1 – Chạy & cache detections

- **Phụ thuộc:** T1, T1-n, T2, T3; ảnh test-dev; D4 (`labeled/`).
- **Đầu vào:** trọng số của 4 detector + ảnh các chuỗi `val`, `test-dev`, `VN-labeled`; tham số `conf=0.05`, `iou(NMS)=0.7`, `max_det=1000` (VisDrone rất đông), `imgsz` bằng lúc train.
- **Đầu ra:**
  ```
  results/dets/<det_id>/<split>/<seq>.txt     # DET cache: frame,-1,x,y,w,h,score,class
  results/dets/<det_id>/<split>/timing.json   # ms/frame trung bình, GPU dùng
  ```
  `det_id ∈ {yolo11m, yolo11n, frcnn_r50, rfdetr}`; `split ∈ {val, test-dev, vn}`.
- **DoD:** mỗi chuỗi có đúng 1 file; số frame trong file = `seqLength` (frame không có box thì không có dòng, điều này vẫn hợp lệ); kiểm tra lại mAP từ cache bằng pycocotools ra khớp với mAP lúc val.

### I2 – Ma trận Detector × Tracker

- **Phụ thuộc:** I1, TRK, D3.
- **Đầu vào:** DET cache val/test-dev; ảnh các chuỗi (cho ReID/CMC); `cfg/trackers/*.yaml`; GT TrackEval của D3.
- **Xử lý:**
  1. 3 detector (YOLO11m, Faster R-CNN, RF-DETR) × 4 tracker = **12 lần chạy trên val**.
  2. Tách output theo 5 lớp → `results/trackeval_in/VisDrone-val-<c>/<det>__<trk>/data/<seq>.txt` (định dạng MOT 10 cột).
  3. Chạy TrackEval: `python TrackEval/scripts/run_mot_challenge.py --GT_FOLDER data/trackeval/gt --TRACKERS_FOLDER results/trackeval_in --BENCHMARK VisDrone --SPLIT_TO_EVAL val-<c> --METRICS HOTA CLEAR Identity --DO_PREPROC True`.
  4. Lấy trung bình 5 lớp, chọn top-3 cấu hình rồi chạy **một lần** trên test-dev.
- **Đầu ra:**
  ```
  results/tracks/<det>/<tracker>/<split>/<seq>.txt        # TRACK (MOT + class)
  results/trackeval/<split>/<c>/<det>__<trk>/pedestrian_summary.txt, pedestrian_detailed.csv
  results/tables/matrix_val.csv        # det, tracker, class, HOTA, DetA, AssA, MOTA, IDF1, IDSW, MT, ML, FPS_track
  results/tables/matrix_val_mean.csv   # trung bình 5 lớp → bảng 2.2 trong báo cáo
  results/tables/matrix_testdev.csv
  results/figures/matrix_heatmap_HOTA.png
  ```
- **DoD:** đủ 12 ô; chạy lại cho kết quả giống hệt (seed và tham số cố định).

> 💡 **Đề xuất:** chạy thêm `YOLO11n × ByteTrack` (và nếu được thì YOLO11n × 4 tracker). Chế độ Nhanh dùng YOLO11n nhưng YOLO11n không nằm trong ma trận 3×4, nên nếu không chạy thêm sẽ không có HOTA để đặt lên biểu đồ trade-off.

### I3 – Đánh giá trên video tự quay VN

- **Phụ thuộc:** I1 (split `vn`), D4, kết quả I2 (cấu hình tốt nhất).
- **Đầu vào:** DET cache `vn` + `data/own/labeled/` + GT TrackEval `VN-test-<c>` (gồm cả lớp **motor**) + `metadata.csv` (để phân nhóm ngày/đêm, góc cao/thấp).
- **Đầu ra:** `results/tables/vn_eval.csv` (`config, class, HOTA, MOTA, IDF1, IDSW`); `results/tables/domain_gap.csv` (`metric, VisDrone_testdev, VN, delta`); `results/tables/vn_by_group.csv` (theo ngày/đêm, góc quay); 4–6 ảnh lỗi điển hình `results/figures/vn_failure_*.jpg`.
- **DoD:** có con số độ suy giảm (delta) cho từng chỉ số và giải thích nguyên nhân bằng các ảnh lỗi.

### I4 – Đo tốc độ FPS (GPU / CPU / NPU)

- **Phụ thuộc:** T1, T1-n, T2, T3, T-Exp, TRK.
- **Đầu vào:** 1 chuỗi val cố định (ví dụ chuỗi dài nhất, cắt cố định 500 frame); danh sách cấu hình (ít nhất 3 chế độ cộng toàn bộ các detector); thiết bị: GPU NVIDIA, CPU (ghi số luồng), NPU (nếu có); backend: PyTorch / ONNX / OpenVINO.
- **Xử lý:** warm-up 50 frame, batch=1; đo tách riêng `t_read, t_preprocess, t_det, t_reid, t_track, t_render`; dùng `torch.cuda.synchronize()` trước mỗi lần lấy thời gian trên GPU.
- **Đầu ra:** `results/benchmark/fps.csv`:
  `config, detector, tracker, device, backend, precision, imgsz, n_frames, t_read_ms, t_det_ms, t_reid_ms, t_track_ms, t_render_ms, latency_p50_ms, latency_p95_ms, fps_e2e, vram_peak_mb, ram_peak_mb` + thông tin máy `results/benchmark/hardware.txt`.
- **DoD:** mỗi cấu hình đo 3 lần, báo trung bình ± độ lệch chuẩn.

### I5 – Đánh giá chức năng đếm xe

- **Phụ thuộc:** D5, module `analytics.py` (mục 6.2-1), cấu hình Cân bằng đã chốt.
- **Đầu vào:** 20 video + `count_lines.json` (dùng đúng vạch đã đếm tay) + `count_gt.csv` + cấu hình chạy.
- **Đầu ra:** `results/counting/pred_counts.csv` (`video_id, class, direction, count_pred`); `results/counting/count_eval.csv` (`video_id, class, direction, gt, pred, abs_err, rel_err`); bảng tổng hợp theo lớp `count_summary.csv` (`class, MAE, MAPE(%), tổng_gt, tổng_pred`); hình `count_scatter.png` (gt và pred, có đường y=x).
- **DoD:** có sai số theo từng lớp (car, truck, bus, motor).

> ⚠ Công thức trong plan `|máy − tay| / tay` là **sai số tương đối (MAPE)**, không phải MAE. Nên báo cả hai: `MAE = mean|pred − gt|` (đơn vị: xe) và `MAPE = mean(|pred − gt| / gt) × 100%` (bỏ các ô có `gt = 0`, hoặc dùng `sum|err| / sum gt`).

### I6 – Vẽ phân cụm ReID t-SNE

- **Phụ thuộc:** P4.
- **Đầu vào:** `results/features/reid/{tsne_2d.npy, pca_2d.npy, meta.csv}`.
- **Đầu ra:** `results/figures/reid_tsne.png`, `reid_pca.png` (tô màu theo ID, đánh dấu riêng các điểm bị occlusion), `reid_tsne.json` (để app hiển thị biểu đồ tương tác nếu cần).
- **DoD:** hình đọc được ở cỡ giấy A4; chú thích nêu rõ silhouette score.

### TO – Biểu đồ Trade-off FPS–HOTA & chốt cấu hình

- **Phụ thuộc:** I2, I4.
- **Đầu vào:** `matrix_val_mean.csv` + `fps.csv`.
- **Đầu ra:** `results/figures/fps_vs_hota.png` (mỗi điểm là một cấu hình, có vẽ đường Pareto); `cfg/modes.yaml`:
  ```yaml
  fast:     {detector: yolo11n, backend: openvino, tracker: bytetrack, imgsz: 640, conf: 0.25}
  balanced: {detector: yolo11m, backend: pt,       tracker: ocsort,    imgsz: 640, conf: 0.25}
  accurate: {detector: rfdetr,  backend: pt,       tracker: botsort,   reid: osnet_x1_0, cmc: sparseOptFlow}
  advanced: {base: accurate, sam: mobile_sam}
  ```
- **DoD:** app chỉ đọc `cfg/modes.yaml`, không hard-code cấu hình.

---

## 6. GIAI ĐOẠN 4 – ỨNG DỤNG WEB

### 6.1. Các module lõi `core/`

| Module | Đầu vào | Đầu ra |
|:-|:-|:-|
| `detector.py` – `Detector(name, weights, backend, device).predict(frame, conf, classes)` | frame BGR `(H,W,3)`, ngưỡng conf, danh sách lớp | `np.ndarray (N,6)` `[x1,y1,x2,y2,conf,cls]` (toạ độ ảnh gốc) |
| `tracker.py` – `Tracker(name, cfg).update(dets, frame)` | `(N,6)` + frame | `(M,7)` `[x1,y1,x2,y2,id,conf,cls]` |
| `sam_mot.py` – `segment(frame, tracks)` | frame + `(M,7)` | list mask `(H,W) bool` theo id |
| `analytics.py` | track từng frame, `fps`, vạch/ROI | số đếm, sự kiện, heatmap, thống kê (mục 6.2) |
| `visualizer.py` | frame + track + lịch sử tâm + mask | frame đã vẽ (box, ID, màu theo lớp, đuôi quỹ đạo 30 frame, FPS) |
| `exporter.py` | toàn bộ kết quả của một phiên | 4 file xuất (mục 6.1-4) |
| `pipeline.py` – `run(source, mode_cfg, options, callback)` | nguồn video + cấu hình | `outputs/<session_id>/` + gọi callback tiến độ cho UI |

### 6.1-1. Nhận dữ liệu đầu vào

| Nguồn | Đầu vào | Kiểm tra (validate) | Đầu ra cho pipeline |
|:-|:-|:-|:-|
| Upload | file `.mp4/.avi/.mov`, ≤ N MB (ví dụ 500 MB), ≤ 5 phút | đuôi file + MIME + `cv2.VideoCapture` mở được + đọc được frame đầu | `source = path`, `fps, W, H, n_frames` |
| Webcam | quyền camera trình duyệt (`streamlit-webrtc` / Gradio `Image(sources="webcam", streaming=True)`) | có frame trong 3 s | luồng frame |
| RTSP/URL | chuỗi `rtsp://...` hoặc `http://...m3u8` | kết nối được trong 5 s | luồng frame (không có `n_frames` nên không có progress bar %) |
| Video mẫu | click chọn 1 trong 3–5 video trong `assets/samples/` (lấy từ `data/own/demo/` + 1 chuỗi VisDrone + 1 cảnh đêm) | – | `source = path` |

### 6.1-2. Chọn phương pháp

- **Đầu vào UI:** radio Nhanh / Cân bằng / Chính xác / (SAM-MOT); tab Nâng cao: dropdown detector, dropdown tracker, slider conf (0.05–0.9, mặc định 0.25), multiselect lớp.
- **Đầu ra:** dict `run_cfg` = `modes.yaml[mode]` được ghi đè bởi lựa chọn ở tab Nâng cao; lưu vào `outputs/<sid>/run_config.json`.

### 6.1-3. Hiển thị

- **Đầu vào:** frame đã vẽ từ `visualizer.py` + tiến độ `i/n_frames` + FPS tức thời (trung bình trượt 30 frame).
- **Đầu ra:** khung video cập nhật liên tục, progress bar, đồng hồ FPS, bảng đếm trực tiếp.

### 6.1-4. Xuất kết quả

| File | Định dạng / Schema |
|:-|:-|
| `outputs/<sid>/result.mp4` | H.264 (`avc1`) để trình duyệt phát được; cùng fps với video gốc |
| `outputs/<sid>/result_mot.txt` | `frame,id,x,y,w,h,conf,class,-1,-1` |
| `outputs/<sid>/trajectories.json` | `{"video":{...,"fps":30,"W":1920,"H":1080}, "tracks":[{"id":7,"class":"car","frames":[12,13,...],"boxes":[[x,y,w,h],...],"centers":[[cx,cy],...],"first_seen_s":0.4,"last_seen_s":6.1}]}` |
| `outputs/<sid>/stats.csv` | `time_s, class, n_in_frame, n_cross_A, n_cross_B, cum_A, cum_B` (mỗi giây 1 dòng × lớp) |
| *(kèm)* `heatmap.png`, `counts.csv`, `events.csv`, `run_config.json` | xem 6.2 |

Có nút "Tải tất cả (.zip)".

### 6.2. Chức năng nghiệp vụ

**6.2-1. Đếm xe qua vạch (2 chiều)**
- **Đầu vào:** 2 điểm vạch do người dùng vẽ trên frame đầu (`streamlit-drawable-canvas` hoặc Gradio `ImageEditor`), toạ độ phải đổi về pixel ảnh gốc; tâm (hoặc điểm giữa cạnh dưới) của mỗi track qua các frame; tên 2 chiều do người dùng đặt.
- **Xử lý:** với mỗi `id`, xét đoạn `p(t-1)→p(t)`: nếu đoạn này cắt đoạn vạch và dấu tích có hướng đổi từ − sang + thì là A, ngược lại là B. Mỗi id chỉ đếm 1 lần mỗi chiều. Thêm khoảng đệm vài pixel để chống đếm trùng khi box rung quanh vạch.
- **Đầu ra:** `counts.csv` (`class, direction, count`), `events.csv` (`frame, time_s, id, class, direction`), chữ đếm vẽ trực tiếp trên video.

**6.2-2. Heatmap mật độ**
- **Đầu vào:** tâm của tất cả track mọi frame + kích thước frame + frame nền (frame đầu hoặc trung vị).
- **Xử lý:** cộng dồn vào lưới `(H/4, W/4)` → Gaussian blur → chuẩn hóa → colormap JET → chồng lên frame nền với alpha 0.5. Tùy chọn: "theo thời gian lưu lại" (cộng mỗi frame, phản ánh ùn tắc) hoặc "theo lượt xe" (mỗi id cộng 1 lần).
- **Đầu ra:** `heatmap.png`, `heatmap_raw.npy`.

**6.2-3. Thống kê lưu lượng theo thời gian**
- **Đầu vào:** track theo frame + `fps` + `events.csv`.
- **Đầu ra:** `stats.csv`; biểu đồ đường số xe theo giây/phút (Plotly); biểu đồ tròn tỉ lệ lớp; `speed.csv` (`id, class, v_mean_px_s`), trong đó vận tốc = độ dài quãng đường tâm (đã làm mượt) / thời gian tồn tại.

**6.2-4. So sánh A/B** *(trung bình)*: đầu vào là 1 video + 2 `run_cfg`; đầu ra là 2 video đặt cạnh nhau + bảng `FPS, số track, số đếm` (IDSW chỉ tính được nếu video có GT).

**6.2-5. ROI Alert** *(trung bình)*: đầu vào là đa giác `[[x,y],...]` + track; xử lý bằng `cv2.pointPolygonTest` trên tâm; đầu ra là `roi_events.csv` (`id, class, t_enter, t_exit`), viền ROI đổi màu và cảnh báo trên UI.

> ⚠ **Giới hạn quan trọng:** đếm vạch, heatmap và ROI giả định **camera đứng yên** (drone hover, camera cố định). Với video drone bay di chuyển, toạ độ pixel không còn gắn với mặt đường nên kết quả không còn ý nghĩa. Cần ghi điều này trong UI/User Guide, hoặc phát hiện camera chuyển động (dùng ước lượng affine như CMC) để cảnh báo người dùng.

### 6.3. Thân thiện với người không chuyên & đóng gói

| Việc | Đầu vào | Đầu ra |
|:-|:-|:-|
| UI tiếng Việt + tooltip | danh sách tham số trên UI | `app/i18n_vi.py` (dict chuỗi + tooltip) |
| Xử lý ngoại lệ | file sai định dạng, file quá lớn, RTSP không kết nối được, hết VRAM | thông báo lỗi tiếng Việt (không hiện traceback); `logs/app.log` |
| Tự giảm độ phân giải | FPS đo được < ngưỡng (ví dụ 5) trong 30 frame đầu, hoặc không có CUDA | tự resize frame (ví dụ cạnh dài 960) và hiện thông báo |
| One-click setup | `requirements.txt` (cố định phiên bản), trọng số | `run.sh`, `run.bat` (tạo venv → cài → tải trọng số → mở trình duyệt); `Dockerfile`; (tùy chọn) Hugging Face Space |
| User Guide | ảnh chụp màn hình từng bước | `docs/User_Guide.pdf` + video hướng dẫn 2–3 phút |

---

## 7. GIAI ĐOẠN 5 – KIỂM THỬ

### QA1 – Kiểm thử chức năng

- **Đầu vào:** bộ file test `tests/assets/`: `ok.mp4`, `ok.avi`, `ok.mov`, `fake.mp4` (thực chất là file text), `song.mp3`, `broken.mp4` (cắt đứt giữa chừng), `huge.mp4` (vượt giới hạn), 1 URL RTSP sai; danh sách test case.
- **Đầu ra:** `tests/test_cases.xlsx` (hoặc `.csv`): `TC_id, chuc_nang, buoc_thuc_hien, du_lieu, ket_qua_mong_doi, ket_qua_thuc_te, Pass/Fail, anh_minh_chung`; test tự động `tests/test_core.py` (pytest cho detector/tracker/analytics/exporter, ví dụ: đếm vạch trên quỹ đạo giả lập phải ra đúng số).
- **DoD:** 100% Pass, không crash.

### QA2 – Kiểm thử hiệu năng / tải

- **Đầu vào:** video 10 s, 1 phút, 5 phút (1080p) × 3 chế độ × {CPU, GPU}; chạy liên tiếp 5 phiên để kiểm tra rò rỉ bộ nhớ.
- **Đầu ra:** `results/qa/stress.csv` (`video_len, mode, device, fps, ram_peak_mb, vram_peak_mb, ram_after_mb, thoi_gian_xu_ly, OOM(Y/N)`); biểu đồ RAM theo thời gian `results/qa/mem_timeline.png`.
- **DoD:** không OOM; RAM/VRAM quay về mức nền sau mỗi phiên.

### QA3 – Khảo sát SUS

- **Đầu vào:** 5–10 người không học IT; kịch bản nhiệm vụ (trong plan); phiếu SUS chuẩn 10 câu, thang 1–5 (Google Form); đồng hồ bấm giờ.
- **Đầu ra:** `results/qa/sus_raw.csv` (`user, q1..q10, thoi_gian_hoan_thanh_s, hoan_thanh(Y/N), so_lan_can_tro_giup, gop_y`); `sus_summary.csv` (điểm từng người + trung bình + độ lệch chuẩn); danh sách cải tiến UI rút ra từ góp ý.
  Công thức: `SUS = 2.5 × [Σ(q_lẻ − 1) + Σ(5 − q_chẵn)]` (thang 0–100).
- **DoD:** SUS trung bình ≥ 80.

---

## 8. GIAI ĐOẠN 6 – HỒ SƠ BÀN GIAO

| Sản phẩm | Đầu vào (lấy từ đâu) | Đầu ra |
|:-|:-|:-|
| Báo cáo – Mở đầu | mục tiêu, bối cảnh | `report/` mục Mở đầu |
| Chương 1 | D1–D4 (thống kê dữ liệu), P1–P4 (hình EDA, augmentation, t-SNE), mô tả T1–T4, TRK | mục 1.1–1.6 + sơ đồ pipeline |
| Chương 2 | `runs_log.csv`, `matrix_*.csv`, `fps.csv`, `fps_vs_hota.png`, `vn_eval.csv`, `domain_gap.csv`, `count_summary.csv` | mục 2.1–2.5 |
| Chương 3 | sơ đồ kiến trúc `core/`, ảnh chụp UI, `test_cases.xlsx`, `stress.csv`, `sus_summary.csv` | mục 3.1–3.5 |
| Kết luận | toàn bộ kết quả trên | hạn chế + hướng phát triển |
| Slide | các hình/bảng của báo cáo | `slides/final.pptx` (khoảng 15–20 slide) |
| Video demo | app chạy trên 2–3 video mẫu, có thuyết minh | `demo/demo_2-3min.mp4` |
| Mã nguồn | toàn bộ repo | `README.md` (cài 1 bước, cấu trúc thư mục, cách tái lập từng task), `LICENSE`, link tải trọng số |

---

## 9. Sơ đồ phụ thuộc (thứ tự làm)

```
D1 ─┬─► D2 ─┬─► T2 ───────────┐
    │       └─► T3 ───────────┤
    ├─► P1, P2 ─► T1, T1-n ───┼─► I1 ─► I2 ─┬─► TO (modes.yaml) ─► App (6.x) ─► QA1–QA3 ─► Báo cáo/Slide/Video
    │              └─► T-Exp ─┤          ▲  │                        ▲
D3 ─┴─────────────────────────┼──────────┘  └─► I3                   │
D4 ─┬─► P3 (domain gap)       └─► I4 ───────────► TO                 │
    └─► D5 ──────────────────────────────────────► I5 ◄── analytics.py
P4 ─► I6                       TRK ─► I2       T4 (SAM) ─► chế độ advanced (cuối kỳ)
```

**Đường găng (critical path):** D1 → T1 → I1 → I2 → TO → App → QA3 (SUS cần app gần hoàn chỉnh). D4/D5 (quay, gán nhãn, đếm tay) tốn công nhất về con người, nên **bắt đầu ngay song song** với D1.

---

## 10. Các điểm chưa nhất quán trong plan gốc cần chốt

1. **RF-DETR dùng dữ liệu COCO, không phải YOLO** → D2 phải xuất thêm cấu trúc `rfdetr/`. Câu lệnh mẫu `train_rf_detr.py --dataset visdrone` cần trỏ đúng vào thư mục này.
2. **YOLO11n không có trong ma trận I2** nhưng lại là nền của chế độ Nhanh → cần thêm ít nhất `YOLO11n × ByteTrack` để có HOTA cho biểu đồ trade-off.
3. **DeepSORT không có sẵn trong boxmot** → dùng `deep_sort_realtime` hoặc StrongSORT và ghi rõ trong báo cáo.
4. **D1/D2 chỉ có train/val**, nhưng I1/I2 chạy cả test-dev → cần tải ảnh test-dev và D3 phải có GT test-dev (đã có trong D3).
5. **Công thức "MAE" ở I5 thực ra là sai số tương đối** → báo cả MAE và MAPE.
6. **P1 "resize 640×640" không nên làm vật lý trên dataset** → chỉ letterbox lúc train/infer.
7. **P4 và I6 trùng nhau** → P4 = trích đặc trưng (ra file `.npy`), I6 = vẽ hình/đưa vào app.
8. **Lớp `motor`** không nằm trong 5 lớp đánh giá VisDrone nhưng là lớp chủ đạo ở VN → khi đánh giá VN (I3) và đếm xe (I5) nên thêm `motor`.
9. **Các chức năng nghiệp vụ cần camera tĩnh** → chọn video D5/demo có camera tĩnh và cảnh báo trong UI.
