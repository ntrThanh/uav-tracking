# UAV Tracking

Benchmarking 3 detectors (YOLO11, Faster R-CNN, RF-DETR) × 5 trackers (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT) for multi-object tracking on VisDrone2019-MOT.

Đồ án cuối kỳ môn Thị giác máy tính nâng cao. Dự án so sánh ba họ detector (transformer, một giai đoạn, hai giai đoạn) kết hợp với năm thuật toán tracking trên video quay từ drone, nơi đối tượng rất nhỏ, dày đặc và camera liên tục chuyển động. Tài liệu kỹ thuật đầy đủ nằm trong [docs/](docs/README.md).

## Mục lục

1. [Thành viên](#1-thành-viên)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt](#3-cài-đặt)
4. [Chuẩn bị dữ liệu](#4-chuẩn-bị-dữ-liệu)
5. [Phân tích dữ liệu](#5-phân-tích-dữ-liệu)
6. [Huấn luyện detector](#6-huấn-luyện-detector)
7. [Cache detection, tracking và đánh giá](#7-cache-detection-tracking-và-đánh-giá)
8. [Làm việc với server](#8-làm-việc-với-server)
9. [Quy tắc chung của nhóm](#9-quy-tắc-chung-của-nhóm)
10. [Trạng thái](#10-trạng-thái)

## 1. Thành viên

| Thành viên | Detector | Việc chung |
|---|---|---|
| Nguyễn Trọng Thành (nhóm trưởng) | RF-DETR Medium | Repo, dữ liệu, tiền xử lý dùng chung, đánh giá detection, tổng hợp |
| Đinh Bá Thi | Faster R-CNN R50-FPN v2 | Code 5 tracker, đo FPS thống nhất |
| Đinh Quang Tiến | YOLO11m | Baseline zero-shot, ground truth TrackEval, đánh giá tracking, oracle |

## 2. Cấu trúc thư mục

```
uav-tracking/
├── docs/                    # tài liệu kỹ thuật, kế hoạch, phân công
├── data/
│   ├── README.md            # mô tả bộ dữ liệu
│   ├── tools/               # tiền xử lý dùng chung: visdrone.py, prepare.py, check.py
│   ├── splits/              # danh sách frame dùng chung cho cả nhóm
│   ├── raw/                 # dữ liệu gốc (tự tải, không lên git)
│   └── processed/           # sinh bởi prepare.py (không lên git)
├── src/
│   ├── configs/             # cấu hình detector và tracker
│   ├── detectors/           # lớp bọc chung cho 3 detector
│   ├── trackers/            # lớp bọc chung cho 5 tracker
│   ├── training/            # script train
│   ├── inference/           # cache detection, chạy tracker, xuất video
│   ├── evaluation/          # đánh giá detection và tracking
│   ├── utils/  scripts/
│   └── notebooks/           # notebook phân tích
├── models/rf-detr/          # mã nguồn RF-DETR 1.11.2 (clone riêng, không lên git)
├── outputs/eda/             # biểu đồ và bảng thống kê dữ liệu
└── requirements.txt
```

## 3. Cài đặt

**Clone repo và mã nguồn RF-DETR.** RF-DETR được giữ cố định ở bản release `1.11.2` để cả nhóm dùng cùng phiên bản.

```bash
git clone git@github.com:ntrThanh/uav-tracking.git
cd uav-tracking
git clone --depth 1 --branch 1.11.2 https://github.com/roboflow/rf-detr.git models/rf-detr
```

**Tạo môi trường.** Python 3.11, cài các gói phân tích và tiền xử lý:

```bash
conda create -n rfdetr python=3.11 -y
conda activate rfdetr
pip install -r requirements.txt
```

**Cài thư viện để train** (chỉ cần trên máy có GPU). Người train RF-DETR cài từ mã nguồn đã clone; lệnh này kéo theo PyTorch, torchvision và PyTorch Lightning:

```bash
pip install -e models/rf-detr
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Người train YOLO11m cài `ultralytics`, người train Faster R-CNN dùng `torchvision` có sẵn sau bước trên.

## 4. Chuẩn bị dữ liệu

**Bước 1: tải dữ liệu.** Cần ba phần train, val và test-dev của VisDrone2019-MOT từ [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset). Link Google Drive gốc hay báo "Quota exceeded", cách ổn định là dùng `rclone` với tài khoản Google của chính mình (xem [quy trình tải dữ liệu](docs/technical/06_quy_trinh.md#66-tải-dữ-liệu)). Nếu làm trên server của nhóm thì dữ liệu đã có sẵn, bỏ qua bước này.

**Bước 2: giải nén vào `data/raw/`** để có cấu trúc:

```
data/raw/VisDrone2019-MOT-train/{sequences,annotations}
data/raw/VisDrone2019-MOT-val/{sequences,annotations}
data/raw/VisDrone2019-MOT-test-dev/{sequences,annotations}
```

**Bước 3: tạo các định dạng dùng chung và kiểm tra.** Chạy từ thư mục gốc của repo:

```bash
python data/tools/prepare.py all
python data/tools/check.py
```

Lệnh đầu chạy khoảng 30 giây. Lệnh sau phải in ra `All checks passed` và lưu ảnh vẽ nhãn vào `data/processed/check/` để xem bằng mắt. Kết quả:

| Đầu ra | Dùng cho | Người dùng |
|---|---|---|
| `data/splits/{train,val,test-dev}.txt` | Danh sách frame (train lấy mẫu stride 2) | Cả nhóm |
| `data/processed/rfdetr/{train,valid,test}/` | Train RF-DETR | Thành |
| `data/processed/yolo/visdrone.yaml` | Train YOLO11m | Tiến |
| `data/processed/coco/{train,val,test-dev}.json` | Train Faster R-CNN, tính mAP | Thi, cả nhóm |
| `data/processed/mot/{split}/{seq}/` | Đánh giá tracking bằng TrackEval | Cả nhóm |

Ảnh trong `data/processed/` là symlink trỏ về `data/raw/`, không tốn thêm dung lượng. Các bước tiền xử lý và lý do của từng bước được mô tả ở [data/README.md](data/README.md) và [docs/technical/03_tien_xu_ly.md](docs/technical/03_tien_xu_ly.md).

## 5. Phân tích dữ liệu

Notebook [01_eda_visdrone_mot.ipynb](src/notebooks/01_eda_visdrone_mot.ipynb) thống kê chi tiết bộ dữ liệu: toàn vẹn ảnh, phân bố lớp, kích thước box (gốc và sau khi resize), mật độ, che khuất, phân bố không gian, quỹ đạo, vùng ignored, chất lượng nhãn và độ dư thừa giữa các frame. Biểu đồ và bảng được lưu vào `outputs/eda/`.

```bash
cd src/notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 01_eda_visdrone_mot.ipynb
```

Một số con số quan trọng:

| Chỉ số | train | val | test-dev |
|---|---|---|---|
| Số frame | 24.201 | 2.846 | 6.635 |
| Số box (10 lớp) | 1.105.516 | 114.132 | 307.799 |
| Box/frame trung bình (tối đa) | 45,7 (180) | 40,1 (116) | 46,4 (152) |
| Box small (< 32² px) | 24,9% | 19,2% | 47,8% |
| Box có cạnh < 8 px khi resize về 576 | 27,9% | 24,3% | 24,3% |

## 6. Huấn luyện detector

Mỗi thành viên train một detector, dùng chung dữ liệu trong `data/processed/` và danh sách frame trong `data/splits/`. Cấu hình chi tiết, lý do lựa chọn và các lỗi thường gặp nằm trong [docs/technical/04_detector.md](docs/technical/04_detector.md).

**RF-DETR Medium (T3, T3-hr).** Cấu hình nằm trong `src/configs/detectors/rfdetr_m_576.yaml` và `rfdetr_m_1024.yaml`, script train là `src/training/train_rfdetr.py`. Trên server, chạy trong tmux:

```bash
tmux new -s rfdetr
cd ~/AI_LAB/ntrthanh/rfdetr
bash src/scripts/train_rfdetr.sh src/configs/detectors/rfdetr_m_576.yaml 0 --dry-run   # kiểm tra, không train
bash src/scripts/train_rfdetr.sh src/configs/detectors/rfdetr_m_576.yaml 0             # train trên GPU 0
# thoát tmux mà job vẫn chạy: Ctrl+b rồi d; quay lại: tmux attach -t rfdetr
```

Tham số thứ hai là số hiệu GPU (0 hoặc 1). Có thể thêm `--epochs N`, `--run-name TEN` hoặc `--resume runs/det/<run>/last.ckpt`. Kết quả nằm ở `runs/det/<run_name>/`, log đầy đủ ở `logs/`, mỗi run kết thúc ghi một dòng vào `results/runs_log.csv`. Đo trên GPU L4 ở resolution 576: khoảng 5,2 iteration/giây, 3.032 iteration mỗi epoch (khoảng 10 phút, chưa tính đánh giá trên val), VRAM khoảng 4,3 GB.

**YOLO11m (T1).**

```bash
yolo detect train model=yolo11m.pt data=data/processed/yolo/visdrone.yaml imgsz=640 epochs=80 batch=16 \
     optimizer=SGD lr0=0.01 cos_lr=True close_mosaic=10 patience=20 seed=0 project=runs/det name=T1_yolo11m_640
```

**Faster R-CNN R50-FPN v2 (T2).** Dùng `fasterrcnn_resnet50_fpn_v2(weights="COCO_V1")` với `num_classes=11`, đọc `data/processed/coco/train.json` với gốc ảnh `data/raw/`, bắt buộc đặt `box_detections_per_img=1000`.

Script train chung cho cả ba mô hình sẽ nằm trong `src/training/` và đọc cấu hình từ `src/configs/detectors/`.

## 7. Cache detection, tracking và đánh giá

Quy trình gồm bốn bước, phần code đang được xây dựng trong `src/`:

1. **Cache detection:** chạy checkpoint tốt nhất của mỗi detector một lần trên val và test-dev với ngưỡng score 0,05, ghi ra `results/dets/<det_id>/<split>/<seq>.txt` theo định dạng chung `frame,-1,x,y,w,h,score,class` (mã lớp VisDrone 1–10).
2. **Đánh giá detection:** pycocotools trên DET cache: mAP50-95, mAP50, AP_small/medium/large, AR, FPS.
3. **Tracking:** 5 tracker đọc chung DET cache, ghi ra `results/tracks/<det_id>/<tracker>/<split>/`.
4. **Đánh giá tracking:** TrackEval theo từng lớp trong 5 lớp rồi lấy trung bình: HOTA, DetA, AssA, IDF1, MOTA, IDSW, MT, ML.

Dòng oracle (ground truth làm detection) được chạy với cả 5 tracker để tách chất lượng tracker khỏi chất lượng detector. Chi tiết tham số và chỉ số nằm trong [docs/technical/05_tracking_danh_gia.md](docs/technical/05_tracking_danh_gia.md).

## 8. Làm việc với server

Code được viết ở máy cá nhân và quản lý bằng git; server GPU (2 × NVIDIA L4) chỉ dùng để chạy. Đồng bộ code lên server:

```bash
rsync -av --exclude .git --exclude __pycache__ --exclude data/raw --exclude data/processed \
      --exclude outputs --exclude runs --exclude results ./ 128.168.137.198:AI_LAB/ntrthanh/rfdetr/
```

Job dài chạy trong `tmux` để không bị dừng khi mất kết nối SSH. Kiểm tra GPU rảnh bằng `nvidia-smi` trước khi chạy, vì server có 2 GPU dùng chung cho cả nhóm.

```bash
tmux new -s rfdetr                     # tạo phiên
tmux attach -t rfdetr                  # quay lại phiên
tmux ls                                # liệt kê phiên
```

Xem thêm cách kéo kết quả về, chạy notebook và ghi log thí nghiệm ở [docs/technical/06_quy_trinh.md](docs/technical/06_quy_trinh.md).

## 9. Quy tắc chung của nhóm

Ba detector chỉ so sánh được khi mọi điều kiện khác giống nhau, nên cả nhóm tuân theo các quy tắc sau.

| Quy tắc | Nội dung |
|---|---|
| Dữ liệu | Chỉ dùng dữ liệu sinh bởi `data/tools/prepare.py`; không tự viết bộ chuyển đổi riêng |
| Danh sách frame | Dùng đúng `data/splits/`; train lấy mẫu stride 2, val và test-dev giữ đủ |
| Lớp | Train 10 lớp (mã 1–10); đánh giá tracking 5 lớp: pedestrian, car, van, truck, bus |
| Định dạng kết quả | DET cache `frame,-1,x,y,w,h,score,class` với mã lớp VisDrone 1–10 |
| Đánh giá | mAP của cả 3 mô hình tính bằng cùng một script pycocotools trên DET cache |
| Split | Val để chọn checkpoint và tham số; test-dev chỉ chạy một lần cho cấu hình cuối |
| Tái lập | `seed=0`; mỗi run ghi một dòng vào `results/runs_log.csv` kèm cấu hình và commit hash |
| Git | Không đưa dữ liệu, trọng số, `runs/` lên git; sửa tiền xử lý thì chạy lại `check.py` |

## 10. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Dữ liệu VisDrone2019-MOT | Đã tải và kiểm tra |
| Notebook thống kê dữ liệu | Hoàn thành |
| Tiền xử lý dùng chung | Hoàn thành, `check.py` đạt |
| Script train, cache, tracker, đánh giá | Đang xây dựng |
| Train 3 detector | Chưa bắt đầu |

## Trích dẫn

```bibtex
@article{zhu2021detection,
  title   = {Detection and Tracking Meet Drones Challenge},
  author  = {Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2021}
}
```
