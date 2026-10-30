# 3. Tiền xử lý dữ liệu

## 3.1. Mục đích

Ba detector dùng ba thư viện khác nhau, mỗi thư viện đòi một định dạng dữ liệu riêng. Nếu mỗi thành viên tự viết bộ chuyển đổi, rất dễ xảy ra sai khác nhỏ như khác danh sách ảnh, khác cách xử lý box tràn mép hay lệch mã lớp, khiến kết quả của ba mô hình không còn so sánh được. Vì vậy toàn bộ tiền xử lý được gom vào một bộ công cụ duy nhất trong `data/tools/`, sinh mọi định dạng từ cùng một nguồn và cùng một bộ quy tắc.

Tiền xử lý trong dự án chia làm hai giai đoạn. Giai đoạn **offline** chạy một lần, làm sạch nhãn và chuyển định dạng, do `data/tools/prepare.py` thực hiện. Giai đoạn **online** chạy trong lúc train, gồm resize, augmentation và chuẩn hóa, do thư viện của từng detector thực hiện.

## 3.2. Bộ công cụ

| File | Vai trò |
|---|---|
| `data/tools/visdrone.py` | Nơi duy nhất định nghĩa mã lớp, bước lấy mẫu, quy tắc clip box và cách đọc nhãn |
| `data/tools/prepare.py` | Sinh `data/splits/` và `data/processed/{coco,rfdetr,yolo,mot}` |
| `data/tools/check.py` | Đối chiếu chéo các định dạng và vẽ nhãn lên ảnh để kiểm tra bằng mắt |

Công cụ chỉ cần Python 3.10 trở lên và Pillow. Cách chạy, từ thư mục gốc của dự án:

```bash
python data/tools/prepare.py all          # hoặc chọn: splits coco rfdetr yolo mot
python data/tools/prepare.py coco --splits val
python data/tools/check.py --draw 6
```

Trên server, `prepare.py all` chạy hết khoảng 30 giây cho cả ba split.

## 3.3. Các bước tiền xử lý offline

**Bước 1: đọc nhãn và kích thước ảnh.** Với mỗi chuỗi, công cụ đọc file nhãn và lấy kích thước ảnh từ frame đầu tiên. Notebook EDA đã xác nhận mọi frame trong một chuỗi có cùng độ phân giải, nên đọc một ảnh là đủ.

**Bước 2: clip box về trong ảnh.** Box được cắt về khung `[0, W] × [0, H]`. Box nào sau khi cắt có cạnh dưới 1 px thì bị bỏ. Trên dữ liệu hiện tại có 34 box được clip và không box nào bị bỏ, nên số box đầu ra bằng đúng số box gốc.

**Bước 3: phân loại box.** Box lớp 1–10 là đối tượng và được đưa vào nhãn. Box lớp 0 là vùng ignored và được lưu riêng. Box lớp 11 (others) bị loại.

**Bước 4: chọn frame.** Train lấy các frame lẻ (1, 3, 5, …), tương đương stride 2. Val và test-dev giữ toàn bộ frame. Danh sách frame được ghi vào `data/splits/<split>.txt`, mỗi dòng có dạng `<seq>/<frame:07d>`.

**Bước 5: xuất định dạng.** Từ cùng một tập box và frame đã chọn, công cụ sinh bốn định dạng ở mục 3.5.

**Bước 6: kiểm tra.** `check.py` xác nhận số ảnh và số box của mọi định dạng khớp nhau và khớp dữ liệu gốc, mã lớp hợp lệ, tọa độ nằm trong ảnh, symlink không hỏng, rồi vẽ nhãn lên ảnh mẫu.

## 3.4. Quy tắc và căn cứ

| Quy tắc | Căn cứ từ thống kê |
|---|---|
| Train 10 lớp, bỏ lớp 0 và 11 | Theo chuẩn của VisDrone; lớp 11 chỉ có 3.904 box và không thống nhất về hình dạng |
| Lưu vùng ignored riêng, không xóa hay tô đen trên ảnh | Khoảng 1% box đối tượng nằm trong vùng ignored; tô đen sẽ tạo vùng ảnh giả mà lúc suy luận không có |
| Clip box về trong ảnh | 34 box vượt mép 1–2 px; một số thư viện báo lỗi hoặc bỏ box tràn mép |
| Lấy mẫu stride 2 cho train | Sai khác 4,7/255 giữa frame liên tiếp; stride 2 giữ 99,6% quỹ đạo, tỉ lệ lớp lệch tối đa 0,016 điểm phần trăm, thời gian train giảm một nửa |
| Val và test-dev giữ đủ frame | Đánh giá tracking cần mọi frame; chọn checkpoint trên đủ dữ liệu |
| Giữ nguyên split theo chuỗi | Frame liên tiếp gần như trùng nhau; xáo frame rồi chia lại sẽ làm val chứa ảnh gần giống train, mAP val bị cao ảo |
| Ghi kích thước theo từng ảnh | 7 độ phân giải khác nhau |
| Không resize trước | Mỗi detector có input size riêng và tự resize; resize trước làm mất chi tiết của vật thể nhỏ |
| Không chỉnh màu, khử nhiễu, tăng nét | Không có ảnh hỏng; pipeline lúc train và lúc suy luận phải giống nhau |
| Giữ nguyên phân bố lớp | Baseline của ba detector phải cùng điều kiện; cân bằng lớp để dành cho thí nghiệm sau |

## 3.5. Các định dạng đầu ra

### COCO: `data/processed/coco/<split>.json`

Dùng cho Faster R-CNN và cho đánh giá mAP bằng pycocotools. `file_name` tính tương đối từ `data/raw/`. `category_id` giữ đúng mã VisDrone 1–10. Ngoài các trường chuẩn, mỗi ảnh có thêm `seq` và `frame`, mỗi box có thêm `track_id`, `occlusion`, `truncation`. Vùng ignored nằm ở khóa riêng `ignored_regions`, pycocotools bỏ qua khóa này.

```json
{
  "images": [{"id": 1, "file_name": "VisDrone2019-MOT-train/sequences/uav0000013_00000_v/0000001.jpg",
              "width": 1344, "height": 756, "seq": "uav0000013_00000_v", "frame": 1}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [x, y, w, h], "area": 1234.0,
                   "iscrowd": 0, "track_id": 3, "occlusion": 0, "truncation": 0}],
  "categories": [{"id": 1, "name": "pedestrian", "supercategory": "visdrone"}],
  "ignored_regions": [{"image_id": 1, "bbox": [x, y, w, h]}]
}
```

| Split | Ảnh | Box | Vùng ignored | Dung lượng |
|---|---|---|---|---|
| train | 12.114 | 553.391 | 31.195 | 91 MB |
| val | 2.846 | 114.132 | 3.994 | 19 MB |
| test-dev | 6.635 | 307.799 | 23.472 | 50 MB |

### RF-DETR: `data/processed/rfdetr/{train,valid,test}/`

Đúng cấu trúc mà thư viện `rfdetr` yêu cầu: mỗi thư mục có file `_annotations.coco.json` và ảnh cùng cấp. Tên thư mục là `valid` và `test` (tương ứng val và test-dev) theo quy ước của Roboflow. Ảnh có tên phẳng `<seq>_<frame:07d>.jpg` và là symlink trỏ về `data/raw/`. Nội dung json giống định dạng COCO ở trên, chỉ khác `file_name`.

### YOLO: `data/processed/yolo/<split>/{images,labels}` và `visdrone.yaml`

Mỗi ảnh (symlink) có một file nhãn cùng tên, mỗi dòng `<class> <cx> <cy> <w> <h>` với tọa độ chuẩn hóa về `[0, 1]`. Mã lớp YOLO bằng mã VisDrone trừ 1, nên nằm trong 0–9. File `visdrone.yaml` khai báo đường dẫn ba split và tên lớp, dùng trực tiếp với Ultralytics.

### MOTChallenge: `data/processed/mot/<split>/<seq>/`

Dùng cho đánh giá tracking. Giữ **mọi frame** của cả ba split (train cũng không lấy mẫu).

```
mot/<split>/
├── <seq>/
│   ├── img1 -> ../../../../raw/VisDrone2019-MOT-<split>/sequences/<seq>
│   ├── gt/gt.txt         # frame,id,x,y,w,h,1,class,occlusion
│   ├── gt/ignored.txt    # frame,-1,x,y,w,h,0,0,0
│   └── seqinfo.ini       # name, imDir, frameRate=30, seqLength, imWidth, imHeight, imExt
└── <split>_seqmap.txt
```

VisDrone không công bố FPS của video, nên `frameRate` được đặt là 30 theo giả định phổ biến. Bước tách ground truth theo từng lớp cho TrackEval sẽ được làm trong script đánh giá tracking.

### Danh sách frame: `data/splits/<split>.txt`

| File | Số frame | Ghi chú |
|---|---|---|
| `train.txt` | 12.114 | Frame lẻ của 56 chuỗi |
| `val.txt` | 2.846 | Toàn bộ |
| `test-dev.txt` | 6.635 | Toàn bộ |

Ba file này được đưa lên git, là "nguồn sự thật" về tập ảnh mà cả ba detector dùng.

## 3.6. Định dạng chung cho detection (DET cache)

Mọi detector phải xuất kết quả theo cùng một định dạng để dùng chung code tracker và code đánh giá.

```
results/dets/<det_id>/<split>/<seq>.txt
<frame>,-1,<x>,<y>,<w>,<h>,<score>,<class>
```

| Quy ước | Giá trị |
|---|---|
| `det_id` | `rfdetr_m`, `yolo11m`, `frcnn_r50`, `oracle` |
| Tọa độ | Pixel ảnh gốc, `(x, y)` là góc trên bên trái |
| `class` | Mã VisDrone 1–10 |
| Ngưỡng score khi cache | 0,05 (ByteTrack cần cả box score thấp) |
| Phạm vi frame | Mọi frame của val và test-dev |

Lưu ý về mã lớp: RF-DETR tự đánh lại mã danh mục thành chỉ số liên tục 0–9 dựa trên file train, YOLO cũng dùng 0–9, nên phải cộng thêm 1 trước khi ghi DET cache. Faster R-CNN của torchvision dùng 0 cho nền và 1–10 cho đối tượng nếu train với `num_classes=11`, nên giữ nguyên.

## 3.7. Tiền xử lý online khi train

Phần này do thư viện của từng detector thực hiện, không nằm trong `prepare.py`. Chi tiết của RF-DETR được đọc trực tiếp từ mã nguồn bản 1.11.2.

| Thao tác | RF-DETR Medium | YOLO11m | Faster R-CNN |
|---|---|---|---|
| Resize khi train | Đa tỉ lệ, ảnh vuông, kích thước ngẫu nhiên mỗi batch trong 416–736 px (ở resolution 576) | Letterbox về 640, giữ tỉ lệ | Cạnh ngắn 800, cạnh dài tối đa 1333 |
| Lật ngang | p = 0,5 | p = 0,5 | p = 0,5 |
| Augmentation khác | Không | Mosaic, HSV, scale, translate (mặc định Ultralytics, tắt mosaic 10 epoch cuối) | Có thể thêm multi-scale |
| Chuẩn hóa | Mean/std ImageNet | Chia 255 | Mean/std ImageNet |

Với RF-DETR, khoảng đa tỉ lệ được tính bằng `(resolution // 32 + k) × 32` với k từ −5 đến 5, vì `expanded_scales=True` theo mặc định và block size của Medium là 16 × 2 = 32. Ở resolution 1024, khoảng tương ứng là 864–1184 px, nên VRAM cần cho ảnh lớn nhất tăng đáng kể.

### Những augmentation không dùng

Preset `AUG_AERIAL` của RF-DETR (lật dọc, xoay 90°) dành cho ảnh vệ tinh nhìn thẳng từ trên xuống. VisDrone quay xiên, vật ở trên ảnh thì nhỏ và xa, ở dưới thì lớn và gần, nên lật dọc hay xoay tạo ra cảnh không tồn tại. Mosaic và copy-paste cũng không cần cho RF-DETR vì ảnh đã rất đông. Nếu muốn thử augmentation, chỉ nên thêm biến đổi độ sáng và độ tương phản nhẹ (khoảng ±0,15), có cơ sở là test-dev có cảnh thiếu sáng như `uav0000073_04464_v`.

## 3.8. Kiểm tra

Kết quả `check.py` trên server, cả ba split đều đạt:

| Kiểm tra | Mô tả |
|---|---|
| Số frame trong `splits/` | Bằng số frame được chọn từ dữ liệu gốc |
| COCO | Số ảnh, số box khớp; mã lớp trong 1–10; box nằm trong ảnh; mọi `file_name` tồn tại |
| RF-DETR | Số ảnh, số box khớp COCO; mọi symlink hợp lệ |
| YOLO | Số ảnh bằng số file nhãn; số box khớp; mã lớp 0–9; tọa độ trong `[0, 1]` |
| MOT | Số box trong `gt.txt` bằng tổng box gốc của mọi frame |
| Kiểm tra bằng mắt | Ảnh vẽ nhãn trong `data/processed/check/` cho thấy box khớp đối tượng và đúng lớp |

Mỗi khi sửa `visdrone.py` hoặc `prepare.py`, phải chạy lại cả hai lệnh và xác nhận `All checks passed` trước khi train.
