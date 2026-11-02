# T1 — YOLO11m fine-tuning trên VisDrone2019-DET (Kaggle)

## 1. Tổng quan

Thí nghiệm **T1** sử dụng **YOLO11m** khởi tạo từ trọng số pretrained COCO (`yolo11m.pt`) và **fine-tune toàn bộ mô hình** trên VisDrone2019-DET, với 10 lớp đối tượng. Quá trình train, đánh giá trên DET test-dev và so sánh với YOLO11m gốc được thực hiện bằng Kaggle Notebook.

- **Run:** `T1_yolo11m_640_det`.
- **Model:** YOLO11m — full fine-tuning, không cấu hình đóng băng backbone/neck (`freeze=None`; Ultralytics tự đóng băng trọng số DFL mặc định).
- **Dữ liệu:** [VisDrone MOT + DET đã chuyển đổi](https://www.kaggle.com/datasets/tiendq/visdrone-mot-det-converted).
- **Checkpoint:** `weights/best.pt` trong Kaggle Dataset chứa output training.
- **Link checkpoint trên Kaggle:** https://www.kaggle.com/datasets/tiendq/result-train-det-yolo11m-80e.
- **Trạng thái/epoch thực chạy:** 80.

## 2. Dữ liệu và tiền xử lý

| Split | Số ảnh | Vai trò |
|---|---:|---|
| VisDrone2019-DET train | 6.414 | Fine-tuning |
| VisDrone2019-DET val | 548 | Validation mỗi epoch, chọn `best.pt` |
| VisDrone2019-DET test-dev | 1.610 | Đánh giá cuối cùng sau khi chọn checkpoint |

Dữ liệu được xử lý bằng [`data/tools/prepare.py`](../../../data/tools/prepare.py) và [`data/tools/visdrone.py`](../../../data/tools/visdrone.py):

- Giữ 10 lớp VisDrone (ID 1–10); nhãn YOLO được đổi thành ID 0–9.
- Clip bounding box nằm trong biên ảnh, loại box không hợp lệ; loại 57 ảnh DET train thuộc nguồn video trùng với MOT val/test-dev theo danh sách chung.
- Xuất `data/processed/yolo_det/` ở định dạng YOLO `class cx cy w h` (tọa độ chuẩn hóa).
- Không resize hoặc augmentation ảnh offline; Ultralytics xử lý ảnh và augmentation khi train.

Kaggle Input ban đầu dùng ảnh gốc và symlink; notebook khôi phục đường dẫn ảnh trong `/kaggle/working/` để huấn luyện và đánh giá. Dataset gốc và dataset chuyển đổi không đưa lên Git.

## 3. Cấu hình training

| Tham số | Thiết lập |
|---|---|
| Pretrained checkpoint | `yolo11m.pt` (COCO) |
| Image size | `640` |
| Epoch tối đa (cấu hình mặc định) | `80` |
| Batch | `16` |
| Optimizer | SGD |
| Learning rate ban đầu | `0.01` |
| LR scheduler | Cosine (`cos_lr=True`) |
| Early stopping patience | `20` |
| Close mosaic | `10` epoch cuối |
| Seed | `0` |
| AMP | Bật |
| Max detections mỗi ảnh | `1000` |
| Chọn checkpoint | `best.pt`, theo fitness trên validation của Ultralytics (ưu tiên mAP50-95) |

Thông số chính xác của **lần chạy đã bàn giao** nằm trong [`run_config.yaml`](run_config.yaml), [`env.txt`](env.txt) và [`train/results.csv`](train/results.csv). 

## 4. Notebook và cách tái lập

Các notebook thuộc thư mục [`src/notebooks/`](../../../src/notebooks/):

| Notebook | Mục đích |
|---|---|
| `03_prepare_visdrone_kaggle.ipynb` | Chuẩn bị dữ liệu MOT/DET bằng script của repo và đóng gói Kaggle Dataset |
| `04_yolo11m_train_det_kaggle.ipynb` | Train YOLO11m trên DET; lưu checkpoint, loss/mAP và ảnh lỗi validation |
| `05_yolo11m_eval_det_test_kaggle.ipynb` | Nạp `best.pt`, đánh giá đủ 1.610 ảnh DET test-dev |
| `06_yolo11m_compare_pretrained_finetuned.ipynb` | So sánh YOLO11m COCO gốc với YOLO11m fine-tuned trên tập ảnh chung |


**Chạy trên Kaggle:**

1. Notebook train: Add Input dataset chuyển đổi, bật GPU, đặt `VARIANT="det"`, chạy để tạo `T1_yolo11m_640_det/weights/best.pt` cùng ZIP artifacts. Mặc định tham số train theo bảng ở mục 3.
2. Lưu ZIP training và checkpoint thành Kaggle Dataset riêng. Giữ cả `best.pt` và `last.pt` ngoài GitHub (checkpoint phục vụ inference là `best.pt`).
3. Notebook test: Add Input dataset chuyển đổi **và** dataset checkpoint; chạy evaluator Ultralytics với `split="test"`.
4. Notebook comparison: Add hai dataset đầu vào tương tự, sử dụng bản đã sửa COCOeval; kết quả so sánh này có giao thức **6 nhóm lớp quy đổi**, không phải mAP 10 lớp VisDrone chính thức.

## 5. Kết quả và artifact đã bàn giao

Thư mục này chỉ chứa **file nhỏ cần thiết cho báo cáo/tái lập**, lấy từ output thực tế của ba notebook Train, Test và Comparison:

```text
T1_yolo11m_640_det_seed0/
├── README.md
├── run_config.yaml
├── env.txt
├── data_provenance.json                từ training output
├── train/
│   ├── results.csv
│   ├── validation_metrics.json
│   ├── per_class_ap.csv
│   ├── training_curves_loss_map.png
│   └── errors/
│       ├── 01_FN_small.png
│       ├── 02_FP.png
│       └── 03_class_confusion.png
├── test/
│   ├── test_metrics.json
│   ├── test_summary.csv
│   ├── per_class_ap_test.csv
│   ├── test_eval_config.yaml
│   ├── confusion_matrix.png
│   └── confusion_matrix_normalized.png
└── comparison/
    ├── metrics_comparison.csv
    ├── per_class_ap.csv
    ├── summary.json
    ├── comparison_metrics.png
    └── visual_examples/
```

### 5.1. Kết quả DET validation và DET test-dev

| Chỉ số | DET val | DET test-dev |
|---|---:|---:|
| mAP50-95 | 0.275 | 0.219 |
| mAP50 | 0.463 | 0.377 |
| mAP75 | 0.275 | 0.222 |
| Precision | 0.578 | 0.499 |
| Recall | 0.454 | 0.402 |

Các chỉ số trên là kết quả **Ultralytics YOLO detection evaluation**, không tự động tương đương với kết quả từ pycocotools của detector khác. Khi so sánh chính thức ba detector, cần chuẩn hóa evaluator, vùng ignored, số detection tối đa và tập dữ liệu.

### 5.2. Phân tích định lượng và định tính

- **Đường cong:** `train/training_curves_loss_map.png` thể hiện train/val box loss, classification loss, DFL loss và mAP theo epoch. File số liệu gốc là `train/results.csv`.
- **Ca lỗi:** thư mục `train/errors/` chứa các crop Ground Truth / Prediction minh họa FN vật nhỏ, FP và nhầm lớp nếu có ca phù hợp.
- **Test:** `test/per_class_ap_test.csv` và confusion matrix cho phép xem lớp nào hoạt động tốt hoặc kém.
- **Baseline:** `comparison/metrics_comparison.csv` và ảnh trong `comparison/visual_examples/` cho thấy thay đổi sau fine-tuning. Phép so sánh baseline quy về **6 nhóm lớp chung** (person, bicycle, car, truck, bus, motorcycle); không dùng bảng này thay cho đánh giá DET 10 lớp.

## 6. Lưu trữ checkpoint và giới hạn

- **Kaggle Dataset checkpoint:** https://www.kaggle.com/datasets/tiendq/result-train-det-yolo11m-80e.
- **Kaggle Dataset đã chuyển đổi:** https://www.kaggle.com/datasets/tiendq/visdrone-mot-det-converted
- **Tính tái lập:** kiểm tra phiên bản thư viện, GPU, thời gian và tham số thực tế trong `env.txt` / `run_config.yaml`; lưu provenance của dataset.
- **Phạm vi:** kết quả DET test-dev là đánh giá detection; chưa phải kết quả tracking trên MOT.
