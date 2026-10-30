# UAV Tracking

Benchmarking 3 detectors (YOLO11, Faster R-CNN, RF-DETR) × 5 trackers (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT) for multi-object tracking on VisDrone2019-MOT.

Đồ án cuối kỳ môn Thị giác máy tính nâng cao.

## Thành viên

| Thành viên | Detector |
|---|---|
| Nguyễn Trọng Thành (nhóm trưởng) | RF-DETR Medium |
| Đinh Bá Thi | Faster R-CNN R50-FPN v2 |
| Đinh Quang Tiến | YOLO11m |

## Cấu trúc thư mục

```
.
├── docs/             # kế hoạch, phân công, mô tả nhiệm vụ
├── src/
│   └── notebooks/    # notebook phân tích (EDA, ...)
├── outputs/eda/      # biểu đồ và bảng thống kê dữ liệu
├── data/             # dữ liệu, không đưa lên git
└── requirements.txt
```

## Dữ liệu

Tải VisDrone2019-MOT (train, val, test-dev) từ [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) và giải nén vào `data/raw/`:

```
data/raw/VisDrone2019-MOT-{train,val,test-dev}/{sequences,annotations}
```

## Cài đặt

```bash
conda create -n rfdetr python=3.11 -y
conda activate rfdetr
pip install -r requirements.txt
```

## Notebook

| Notebook | Nội dung |
|---|---|
| [01_eda_visdrone_mot.ipynb](src/notebooks/01_eda_visdrone_mot.ipynb) | Thống kê chi tiết dữ liệu VisDrone2019-MOT và khuyến nghị khi train |
