# Dữ liệu VisDrone2019-MOT

Thư mục này mô tả bộ dữ liệu và chứa bộ công cụ tiền xử lý dùng chung cho cả nhóm. Mọi thành viên tạo dữ liệu bằng cùng một lệnh, nên ba detector (RF-DETR, YOLO11, Faster R-CNN) được train và đánh giá trên đúng cùng một tập ảnh, cùng nhãn và cùng mã lớp.

## 1. Giới thiệu bộ dữ liệu

VisDrone2019-MOT là bộ dữ liệu theo dõi nhiều đối tượng trong video quay từ drone, do nhóm AISKYEYE (Đại học Thiên Tân) công bố. Video được quay ở nhiều thành phố của Trung Quốc, trong điều kiện thời tiết và ánh sáng khác nhau, với góc nhìn xiên từ trên cao. Mỗi đối tượng được gán một box và một ID giữ nguyên qua các frame.

| Split | Số chuỗi | Số frame | Số box (lớp 1–10) | Số quỹ đạo | Vai trò trong đồ án |
|---|---|---|---|---|---|
| train | 56 | 24.201 | 1.105.516 | 7.342 | Huấn luyện detector |
| val | 7 | 2.846 | 114.132 | 758 | Chọn checkpoint, dò tham số tracker |
| test-dev | 17 | 6.635 | 307.799 | 2.293 | Báo cáo kết quả cuối, chỉ chạy một lần |

Ảnh có 7 độ phân giải khác nhau, từ 680×382 đến 3840×2160, phần lớn train là 1904×1071 và 2688×1512. Trung bình mỗi frame có khoảng 45 box, frame đông nhất có 180 box. Đối tượng rất nhỏ, cạnh trung vị khoảng 46 px ở train và 33 px ở test-dev. Phân tích chi tiết nằm trong notebook [01_eda_visdrone_mot.ipynb](../src/notebooks/01_eda_visdrone_mot.ipynb).

### Các lớp

| Mã | Lớp | Dùng để train | Dùng khi đánh giá tracking |
|---|---|:-:|:-:|
| 0 | ignored region | | (dùng để lọc) |
| 1 | pedestrian | ✓ | ✓ |
| 2 | people | ✓ | |
| 3 | bicycle | ✓ | |
| 4 | car | ✓ | ✓ |
| 5 | van | ✓ | ✓ |
| 6 | truck | ✓ | ✓ |
| 7 | tricycle | ✓ | |
| 8 | awning-tricycle | ✓ | |
| 9 | bus | ✓ | ✓ |
| 10 | motor | ✓ | |
| 11 | others | | |

`pedestrian` là người đi bộ hoặc đứng, còn `people` là người ở tư thế khác như ngồi hoặc đi xe.

### Định dạng nhãn gốc

Mỗi chuỗi có một file `annotations/<seq>.txt`, mỗi dòng là một box tại một frame:

```
<frame>,<track_id>,<x>,<y>,<w>,<h>,<score>,<class>,<truncation>,<occlusion>
```

Tọa độ tính theo pixel của ảnh gốc, `(x, y)` là góc trên bên trái. Trong ground truth, `score = 1` nghĩa là box được tính khi đánh giá. `truncation` cho biết đối tượng bị cắt ở mép ảnh (0 hoặc 1), `occlusion` cho biết mức bị che (0 không, 1 một phần, 2 nhiều).

## 2. Cấu trúc thư mục

```
data/
├── README.md
├── tools/                  # code tiền xử lý dùng chung (đưa lên git)
│   ├── visdrone.py         # hằng số, mã lớp, đọc nhãn, quy tắc làm sạch box
│   ├── det_overlap.py      # tìm ảnh DET trùng cảnh với MOT val/test-dev
│   ├── prepare.py          # tạo splits và các định dạng COCO, RF-DETR, YOLO, MOT
│   └── check.py            # kiểm tra chéo các định dạng và vẽ nhãn lên ảnh
├── splits/                 # danh sách ảnh dùng chung (đưa lên git)
│   ├── train.txt           # 12.114 frame MOT, lấy mẫu stride 2
│   ├── train_motdet.txt    # 10.476 ảnh: MOT stride 6 + 6.414 ảnh DET
│   ├── {train,val,test-dev}_det.txt  # biến thể det: 6.414 / 548 / 1.610 ảnh DET
│   ├── det_train_excluded.txt  # 57 ảnh DET bị loại vì trùng video với MOT val/test-dev
│   ├── val.txt             # 2.846 frame
│   └── test-dev.txt        # 6.635 frame
├── raw/                    # dữ liệu gốc đã giải nén (không đưa lên git)
│   ├── VisDrone2019-MOT-{train,val,test-dev}/{sequences,annotations}
│   └── VisDrone2019-DET-{train,val,test-dev}/{images,annotations}
└── processed/              # sinh ra bởi prepare.py (không đưa lên git)
    ├── coco/  coco_motdet/  coco_det/        # {train,val,test-dev}.json
    ├── rfdetr/  rfdetr_motdet/  rfdetr_det/  # {train,valid,test}/
    ├── yolo/  yolo_motdet/  yolo_det/        # {train,val,test-dev}/{images,labels} + visdrone.yaml
    ├── mot/{train,val,test-dev}/<seq>/{gt,img1,seqinfo.ini}
    └── check/              # ảnh vẽ nhãn để kiểm tra bằng mắt
```

## 3. Cách sử dụng

Tải VisDrone2019-MOT (train, val, test-dev) và VisDrone2019-DET train, val, test-dev (Task 1) từ [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset), giải nén vào `data/raw/`. Sau đó chạy từ thư mục gốc của dự án:

```bash
python data/tools/det_overlap.py     # chỉ cần khi có DET; tạo splits/det_train_excluded.txt
python data/tools/prepare.py all     # khoảng 1 phút, sinh mọi biến thể có đủ dữ liệu gốc
python data/tools/check.py           # phải in ra "All checks passed"
```

Có thể tạo riêng từng định dạng hoặc biến thể, ví dụ `python data/tools/prepare.py coco rfdetr --variants mot`. Nếu chưa tải DET, công cụ chỉ sinh biến thể `mot`. Ảnh trong `processed/` là symlink trỏ về `raw/`, nên không tốn thêm dung lượng. Công cụ chỉ cần Python 3.10 trở lên và Pillow.

### Ba biến thể dữ liệu cho detector

| Biến thể | Train | Val | Test | Đầu ra |
|---|---|---|---|---|
| `mot` | MOT train stride 2 (12.114 ảnh, 553.391 box) | MOT val | MOT test-dev | `processed/{coco,rfdetr,yolo}/` |
| `motdet` | MOT train stride 6 + DET train (10.476 ảnh) | MOT val | MOT test-dev | `processed/{coco,rfdetr,yolo}_motdet/` |
| `det` | DET train (6.414 ảnh, 338.285 box) | DET val (548 ảnh) | DET test-dev (1.610 ảnh) | `processed/{coco,rfdetr,yolo}_det/` |

Ground truth cho tracking (`processed/mot/`) luôn lấy từ VisDrone2019-MOT, bất kể detector được train trên biến thể nào. Với biến thể `det`, danh sách ảnh nằm trong `splits/{train,val,test-dev}_det.txt`.

Biến thể `motdet` và `det` ra đời sau khi chẩn đoán cho thấy RF-DETR train trên `mot` bị overfit: 12.114 frame chỉ đến từ 56 video, mAP train tăng từ 0,391 lên 0,580 trong khi mAP val giảm từ 0,233 xuống 0,178 (xem `results/diagnostics/`). DET bổ sung hàng nghìn cảnh khác nhau với cùng 10 lớp; biến thể `det` còn giúp train nhanh (khoảng 1,7 giờ cho 40 epoch RF-DETR trên 2 GPU) và cho mAP so sánh được với benchmark VisDrone-DET. Cả ba detector phải dùng cùng một biến thể để so sánh công bằng.

Trước khi gộp, `det_overlap.py` loại 57 ảnh DET có cùng mã video với MOT val (45 ảnh từ 4 video) hoặc test-dev (12 ảnh từ 2 video), vì chúng là frame của chính các video dùng để đánh giá. Công cụ còn so khớp perceptual hash (dHash 64 bit) với các frame val/test-dev; kiểm tra bằng mắt cho thấy các cặp có khoảng cách 1–8 đều là cảnh khác nhau có bố cục tương tự, nên mặc định chỉ loại thêm ảnh có hash trùng hoàn toàn (không có trường hợp nào ngoài 57 ảnh trên).

## 4. Quy tắc tiền xử lý chung

Các quy tắc dưới đây được cài đặt trong `visdrone.py` và áp dụng cho mọi định dạng đầu ra. Mỗi quy tắc đều có căn cứ từ notebook thống kê.

| Quy tắc | Lý do |
|---|---|
| Chỉ giữ 10 lớp đối tượng (mã 1–10), bỏ lớp 0 và lớp 11 khỏi nhãn train | Theo chuẩn đánh giá của VisDrone |
| Lưu vùng ignored riêng, không xóa và không tô đen trên ảnh | Trong vùng ignored vẫn có đối tượng thật nhưng không được gán nhãn đầy đủ; vùng này dùng để lọc detection khi đánh giá |
| Clip box về trong ảnh | Có 34 box vượt mép ảnh 1–2 px trong nhãn gốc |
| Train lấy mẫu stride 2 (frame 1, 3, 5, …), val và test-dev giữ đủ | Frame liên tiếp gần như trùng nhau; stride 2 giữ 99,6% số quỹ đạo và tỉ lệ lớp không đổi |
| Giữ nguyên ranh giới split theo chuỗi | Tránh rò rỉ dữ liệu giữa train và val |
| Không resize, không chỉnh màu, không khử nhiễu ảnh trước | Mỗi mô hình tự resize khi train; resize trước làm mất chi tiết vật thể nhỏ |
| Ghi kích thước theo từng ảnh | Dữ liệu có 7 độ phân giải khác nhau |

Augmentation (lật ngang, đa tỉ lệ) được thực hiện trong lúc train bởi từng thư viện, không nằm trong bước tiền xử lý này.

## 5. Các định dạng đầu ra

| Định dạng | Dùng cho | Đặc điểm |
|---|---|---|
| `coco/<split>.json` | Faster R-CNN, đánh giá mAP bằng pycocotools | `file_name` tương đối so với `data/raw/`; `category_id` là mã VisDrone 1–10; mỗi box có thêm `track_id`, `occlusion`, `truncation`; vùng ignored nằm ở khóa riêng `ignored_regions` |
| `rfdetr/{train,valid,test}/` | RF-DETR | Đúng cấu trúc thư viện `rfdetr` yêu cầu, mỗi thư mục có `_annotations.coco.json` và ảnh |
| `yolo/` + `visdrone.yaml` | YOLO11 (Ultralytics) | Mã lớp YOLO = mã VisDrone − 1 (0–9); tọa độ chuẩn hóa `cx cy w h` |
| `mot/<split>/<seq>/` | Đánh giá tracking (TrackEval) | Đủ mọi frame; `gt/gt.txt` dạng `frame,id,x,y,w,h,1,class,occlusion`; `gt/ignored.txt` chứa vùng ignored; `seqinfo.ini` với `frameRate=30` (VisDrone không công bố FPS) |

Tên ảnh trong RF-DETR và YOLO có dạng `<seq>_<frame:07d>.jpg`, ví dụ `uav0000013_00000_v_0000001.jpg`; ảnh DET có tiền tố `det_`, ví dụ `det_0000169_00401_d_0000001.jpg`. Trong COCO json, mỗi ảnh có trường `source` là `mot` hoặc `det`. Nhãn DET gốc có 8 cột `x,y,w,h,score,class,truncation,occlusion` (không có frame và ID) và được làm sạch theo đúng các quy tắc ở mục 4.

### Định dạng chung cho detection (DET cache)

Mọi detector xuất kết quả theo cùng một định dạng để chạy chung code tracker và code đánh giá. Mỗi chuỗi một file `results/dets/<det_id>/<split>/<seq>.txt`, mỗi dòng một box:

```
<frame>,-1,<x>,<y>,<w>,<h>,<score>,<class>
```

Tọa độ theo pixel ảnh gốc, `class` là mã VisDrone 1–10. RF-DETR tự đánh lại mã lớp thành 0–9 bên trong mô hình, còn YOLO dùng 0–9, nên phải cộng thêm 1 trước khi ghi file.

## 6. Trích dẫn

```bibtex
@article{zhu2021detection,
  title   = {Detection and Tracking Meet Drones Challenge},
  author  = {Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2021}
}
```
