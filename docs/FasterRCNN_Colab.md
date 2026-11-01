# Faster R-CNN

## Model và kết quả

- Model: Faster R-CNN ResNet50-FPN v2 của torchvision, khởi tạo bằng trọng số COCO.
- Fine-tune toàn bộ backbone; đầu dự đoán gồm 10 lớp VisDrone và 1 lớp background.
- Train trên **VisDrone2019-DET train**: 6.414 ảnh sau khi loại 57 ảnh theo danh sách chung của nhóm.
- Chọn checkpoint trên **DET val**: 548 ảnh.
- Run: `T2_frcnn_r50_v2_det_seed0`.
- Kết quả được bàn giao hiện tại: **24 epoch**, checkpoint tốt nhất tại **epoch 17**.

| Tập | Số ảnh | mAP50–95 | mAP50 |
|---|---:|---:|---:|
| DET val | 548 | 29,10% | 48,10% |
| DET test-dev | 1.610 | 21,88% | 37,75% |

Chỉ số được tính bằng pycocotools, COCO bbox với `maxDets=[1,100,500]`; mAP50–95 lấy tại maxDets 500. Ngưỡng score 0,05, NMS 0,5, tối đa 1.000 box đầu ra mỗi ảnh. **Kết quả hiện chưa lọc vùng ignored**. Khi so sánh chính thức, cả nhóm cần dùng cùng dữ liệu và cùng script/quy tắc đánh giá. Không ghép chỉ số DET với chỉ số MOT.

JSON/CSV lưu AP và AR dạng 0–1; bảng trên đã nhân 100. Các file kết quả nằm trong [thư mục kết quả của run](../results/frcnn/T2_frcnn_r50_v2_det_seed0/).

## Link tải trên Google Drive

- **best.pt — checkpoint tốt nhất:** https://drive.google.com/file/d/1IbOBpkj5c2_lh3KOnKbrDj-t6Ivt5QgU/view?usp=sharing
- **val_cache.zip — detection cache MOT val:** https://drive.google.com/file/d/1NM8-I89JDpWUrcIAty6gifWabjQ6m3Sn/view?usp=sharing
- **test-dev_cache.zip — detection cache MOT test-dev:** https://drive.google.com/file/d/1Hl7LN-BwmmblRuBAvBQ4D0TZ7JdX7bDK/view?usp=sharing

Vị trí gốc trên Drive của người chạy:

```text
MyDrive/AdvanceCV/runs/T2_frcnn_r50_v2_det_seed0/
├── best.pt
└── results/dets/frcnn_r50/
    ├── val_cache.zip
    └── test-dev_cache.zip
```

Không đưa checkpoint, cache hoặc dataset lên Git. `last.pt` giữ trên Drive của người chạy để resume, không cần để sử dụng cache.

## Notebook và cách chạy

- [Notebook Colab đã chạy, kèm output](../src/notebooks/02_fasterrcnn_visdrone_colab.ipynb).
- [Code Faster R-CNN](../src/training/frcnn_colab.py).
- Notebook chứa sẵn code Python và công cụ chuẩn bị dữ liệu của nhóm; chỉ cần tải notebook lên Colab, không cần clone repo.

Chuẩn bị GPU runtime và các ZIP trong:

```text
MyDrive/AdvanceCV/datasets/
├── VisDrone2019-DET-train.zip
├── VisDrone2019-DET-val.zip
├── VisDrone2019-DET-test-dev.zip
├── VisDrone2019-MOT-val.zip
└── VisDrone2019-MOT-test-dev.zip
```

1. Chạy ô 1–6 để kết nối Drive, chuẩn bị dữ liệu và kiểm tra ảnh/nhãn. Nếu train một thí nghiệm mới, dùng `RUN_NAME` mới để không trộn kết quả với run đã bàn giao.
2. Ô 7 train 24 epoch hoặc resume từ `last.pt`. **Bỏ qua cell bổ sung có `CFG["epochs"] = 50` nằm sau ô 7** khi tái lập run này: cell đó chưa chạy trong bản kết quả bàn giao. Không dùng Run all khi chưa bỏ qua cell này.
3. Ô 8 tạo biểu đồ và ảnh dự đoán từ checkpoint tốt nhất.
4. **Chạy cell “Sửa cấu trúc ZIP DET test-dev trước khi chạy cell 9” trước ô 9.** ZIP DET test-dev đang dùng chứa trực tiếp `images/` và `annotations/`; cell này giải nén vào `data/raw/VisDrone2019-DET-test-dev/` và đánh dấu hoàn tất. Ô sửa hiện dành cho ZIP phẳng này; nếu ZIP đã có thư mục bao ngoài đúng tên dataset thì bỏ qua ô sửa. Cách sửa nằm trong notebook, chưa tích hợp vào hàm `ensure_dataset()` trong file Python.
5. Sau khi chốt checkpoint, bật tùy chọn cần thiết và chạy ô 9:

| Tùy chọn | Việc thực hiện |
|---|---|
| `EVALUATE_DET_TEST_DEV` | Đánh giá detection trên 1.610 ảnh DET test-dev |
| `EXPORT_MOT_VAL` | Xuất cache cho 7 chuỗi, 2.846 frame MOT val |
| `EXPORT_MOT_TEST_DEV` | Xuất cache cho 17 chuỗi, 6.635 frame MOT test-dev |

Ô 9 không train lại. Nếu checkpoint/cấu hình vẫn khớp, nó dùng lại kết quả đã hoàn tất. Nếu checkpoint thay đổi, không trộn cache cũ với mới; cần một bộ kết quả riêng. Test-dev không dùng để chọn checkpoint hoặc dò tham số.

## Sử dụng cache cho tracking

Giải nén mỗi ZIP cache vào thư mục split tương ứng. Mỗi chuỗi có một file `.txt` gồm 8 cột:

```text
frame,-1,x,y,w,h,score,class
```

- Frame bắt đầu từ 1; tọa độ theo pixel ảnh gốc, box dạng xywh.
- Class là mã VisDrone **1–10**, không cộng thêm 1.
- Cache giữ đủ 10 lớp; runner tracking chọn 5 lớp `[1,4,5,6,9]`.
- Cột ID là `-1`: đây là detection, **chưa phải kết quả tracking**.
- Frame không có detection không có dòng; `manifest.json` lưu số frame và số detection từng frame.
- `timing.json` là thời gian xuất detection một lượt, không phải FPS của toàn pipeline detector + ReID + tracker.

Người chạy tracking còn cần ảnh MOT gốc cho ReID/bù chuyển động camera, code và cấu hình tracker, cùng GT/công cụ đánh giá để tính HOTA, IDF1 và MOTA. Notebook này chưa triển khai 5 tracker hoặc TrackEval.

## File kết quả nhẹ trong repo

Trong `results/frcnn/T2_frcnn_r50_v2_det_seed0/`:

- `history.csv`: lịch sử epoch 1–24, loss, mAP, LR, thời gian và VRAM.
- `run_config.json`, `environment.txt`, `source_manifest.json`, `status.json`: cấu hình, môi trường, nguồn code và trạng thái run.
- `best_det_val_metrics.json`: chỉ số DET val của epoch tốt nhất.
- `DET-test-dev/metrics.json`, `metrics.csv`: chỉ số DET test-dev.
- `figures/`: đường cong train, AP từng lớp và 6 ảnh GT cạnh dự đoán DET val.

Checkpoint và cache lớn được cung cấp qua ba link Drive ở trên. Notebook đã chạy được giữ làm bằng chứng thực nghiệm; không xóa output hoặc ghi đè nếu chưa lưu bản dự phòng.
