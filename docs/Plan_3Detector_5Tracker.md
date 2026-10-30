# Kế hoạch 3 Detector × 5 Tracker – mỗi người 1 mô hình

## Context

Nhóm 3 người muốn chia việc như sau: mỗi người phụ trách 1 detector (YOLO11, Faster R-CNN, RF-DETR), fine-tune trên VisDrone2019-MOT. Sau đó mỗi người chạy 5 thuật toán tracking trên detections của mô hình mình để tính chỉ số.

**Kết luận: cách chia này làm được và khớp với yêu cầu học phần** (ít nhất 3 phương pháp, có so sánh và giải thích). Nó cũng khớp với hướng đã ghi trong [Task.md](Task.md) (T1 YOLO11m, T2 Faster R-CNN, T3 RF-DETR). Tuy vậy, có **một điều kiện bắt buộc**: 3 người phải dùng chung dữ liệu, chung định dạng cache, chung code tracker và chung code đánh giá. Nếu mỗi người tự viết riêng, 15 con số sẽ không so sánh được với nhau.

**Phạm vi:** chỉ dùng **VisDrone2019-MOT** (train để huấn luyện, val để chọn checkpoint và dò tham số, test-dev để báo cáo kết quả cuối). Gồm 3 detector, mỗi detector chạy 5 tracker, cộng thêm 5 dòng oracle. **Chưa dùng dữ liệu Việt Nam.** Các phần liên quan đến dữ liệu VN (D6, T8, I5 trong [Plan_FineTune_Inference.md](Plan_FineTune_Inference.md)) để lại làm sau.

---

## 1. Phân công

| Người | Detector | Việc chung phụ trách (làm cho cả nhóm) |
|---|---|---|
| Nguyễn Trọng Thành (nhóm trưởng) | **RF-DETR Medium** (`rfdetr`) | Repo + môi trường, tải dữ liệu, `cfg/classes.yaml` + danh sách frame train chung, D2 (COCO json, kể cả bản cho RF-DETR), `scripts/eval_det.py` (pycocotools), thống kê max box/frame; tổng hợp giữa kỳ, biểu đồ trade-off, điều phối |
| Đinh Bá Thi | **Faster R-CNN R50-FPN v2** (torchvision) | `core/tracker.py` + `scripts/run_tracker.py` + `cfg/trackers/*.yaml` (5 tracker); đo FPS thống nhất cho cả 3 mô hình |
| Đinh Quang Tiến | **YOLO11m** (Ultralytics) | D1 (nhãn YOLO), baseline zero-shot YOLO11m, D3 (GT TrackEval tách 5 lớp + vùng ignored), `scripts/eval_track.py` (TrackEval → CSV), chạy **oracle × 5 tracker**, ghép ma trận + heatmap |

Danh sách hình trực quan, bảng tham số và bảng điền kết quả: [Phan_cong_3Detector_5Tracker.xlsx](Phan_cong_3Detector_5Tracker.xlsx).

Mỗi người tự train mô hình của mình, cache detections, chạy 5 tracker bằng **script chung**, rồi đẩy CSV vào `results/tables/`.

**Quy tắc công bằng (bắt buộc):**
1. Dùng cùng danh sách ảnh train/val. Train lấy 1/2 số frame, val lấy toàn bộ. Lưu danh sách trong `data/splits/{train,val}.txt`.
2. Dùng cùng 10 lớp VisDrone (id 1–10), bỏ lớp 0 (ignored) và lớp 11 (others).
3. Cùng định dạng DET cache: `frame,-1,x,y,w,h,score,class` (tọa độ ảnh gốc, class theo VisDrone id).
4. **mAP của cả 3 mô hình tính bằng cùng một script pycocotools trên DET cache.** Không lấy số từ `yolo val` hay log của RF-DETR, vì mỗi thư viện tính mAP hơi khác nhau.
5. Tracker chạy bằng cùng code, cùng file cfg, cùng ReID weight, cùng phiên bản thư viện (pin trong `requirements.txt`).
6. `seed=0`. Mỗi run ghi một dòng vào `results/runs_log.csv`.

---

## 2. Cấu hình huấn luyện từng detector

| | YOLO11m | Faster R-CNN R50-FPN v2 | RF-DETR Medium |
|---|---|---|---|
| Khởi tạo | `yolo11m.pt` (COCO) | `fasterrcnn_resnet50_fpn_v2(weights=COCO_V1)`, thay `FastRCNNPredictor(…, num_classes=11)` | `RFDETRMedium()` (COCO) |
| Số tham số (≈) | 20M | 43M | 33M |
| Input | `imgsz=640` | `min_size=800, max_size=1333` | `resolution=576` (phải là bội số theo yêu cầu thư viện) |
| Epoch | 80, `close_mosaic=10`, `patience=20` | 24 (lịch 2x), giảm lr ×0.1 tại epoch 16 và 22 | 60, `early_stopping=True` |
| Optimizer | `optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=5e-4 cos_lr=True warmup_epochs=3` | SGD `lr=0.02×batch/16`, momentum 0.9, wd 1e-4, warmup tuyến tính 500 iter | AdamW `lr=1e-4`, `lr_encoder=1.5e-4`, wd 1e-4 (mặc định thư viện) |
| Batch | 16 | 4 (hoặc theo VRAM, lr scale theo batch) | 4 × `grad_accum_steps=4` (= 16) |
| Augment | mặc định Ultralytics (mosaic, HSV, flip, scale) | horizontal flip (+ multi-scale nếu kịp) | mặc định thư viện |
| AMP | bật | bật | bật |
| Chọn checkpoint | `best.pt` theo mAP50-95 val | epoch có mAP50-95 val cao nhất | `checkpoint_best_total.pth` |

**Các lỗi hay gặp cần chỉnh:**
- Faster R-CNN: `box_detections_per_img` mặc định là 100, **phải tăng lên 1000** vì frame VisDrone rất đông. Đặt `box_score_thresh=0.05`. Giữ `box_nms_thresh=0.5` (mặc định).
- YOLO11: khi infer đặt `max_det=1000`.
- RF-DETR: số query giới hạn số box mỗi ảnh (mặc định 300). Cần đếm số đối tượng tối đa trên mỗi frame trong GT VisDrone; nếu có frame vượt 300 thì ghi vào phần hạn chế. Format COCO của `rfdetr` dễ bị lệch category id, nên **sau 1 epoch phải vẽ prediction ra để kiểm tra mapping lớp**.
- Cả 3 mô hình: vẽ DET cache của 1 chuỗi val lên ảnh và xem bằng mắt trước khi chạy tracker.

**Cache detections (I1), giống nhau cho cả 3:** `conf=0.05`, NMS giữ mặc định của từng mô hình (YOLO 0.7, FRCNN 0.5, RF-DETR không dùng NMS), chạy trên toàn bộ frame của `val` (7 chuỗi) và `test-dev` (17 chuỗi). Lưu kèm `timing.json` (ms/frame). Ghi vào `results/dets/<det_id>/<split>/<seq>.txt`.

---

## 3. Năm thuật toán tracking và tham số

Bản thân boxmot **không có SORT và DeepSORT gốc**, nên chọn thư viện như sau:

| Tracker | Thư viện | ReID | GMC |
|---|---|:-:|:-:|
| SORT | `sort.py` (abewley), bọc vào `core/tracker.py` | | |
| DeepSORT | `deep_sort_realtime` | ✓ | |
| ByteTrack | `boxmot` | | |
| OC-SORT | `boxmot` | | |
| BoT-SORT | `boxmot` | ✓ | ✓ |

Dùng chung ReID weight `osnet_x0_25_msmt17.pt` cho DeepSORT và BoT-SORT để hai tracker so sánh được với nhau.

**Tham số giai đoạn 1 (mặc định, giống hệt cho cả 3 detector):**

| Tracker | Tham số |
|---|---|
| SORT | ngưỡng det vào `0.4`, `max_age=30`, `min_hits=3`, `iou_threshold=0.3` |
| DeepSORT | ngưỡng det vào `0.4`, `max_age=30`, `n_init=3`, `max_iou_distance=0.7`, `max_cosine_distance=0.2`, `nn_budget=100` |
| ByteTrack | `track_high_thresh=0.5`, `track_low_thresh=0.1`, `new_track_thresh=0.6`, `match_thresh=0.8`, `track_buffer=30` |
| OC-SORT | `det_thresh=0.5`, `max_age=30`, `min_hits=3`, `iou_threshold=0.3`, `delta_t=3`, `inertia=0.2`, `asso_func=iou` |
| BoT-SORT | như ByteTrack + `proximity_thresh=0.5`, `appearance_thresh=0.25`, `with_reid=True`, `cmc_method=sof` |

Thiết lập chung cho cả 5 tracker:
- `frame_rate` lấy theo FPS thật của chuỗi. Cần kiểm tra trong dữ liệu; nếu không có thì đặt 30.
- Chỉ đưa vào tracker 5 lớp đánh giá (pedestrian, car, van, truck, bus).
- Bật chế độ **per-class**: chỉ ghép các box cùng lớp với nhau.
- Lọc bỏ track nằm trong vùng ignored trước khi đánh giá.

**Giai đoạn 2 (nên làm, vì cần cho công bằng):** score của 3 detector không cùng thang đo. Ví dụ score của DETR thường thấp hơn score của YOLO, nên ngưỡng 0.5 sẽ có lợi cho detector này và bất lợi cho detector khác. Vì vậy mỗi người dò riêng **ngưỡng det cao** (`track_high_thresh` / `det_thresh` / ngưỡng vào SORT/DeepSORT ∈ {0.3, 0.4, 0.5, 0.6}) trên **val** cho detector của mình, rồi chọn theo HOTA. Báo cáo cả 2 bảng: mặc định và đã dò. Test-dev chỉ chạy **1 lần** với cấu hình đã chọn.

---

## 4. Chỉ số cần đo

### 4.1. Detection (E0): đo cho từng mô hình, trên val và test-dev

| Chỉ số | Ý nghĩa / ghi chú |
|---|---|
| **mAP50-95** (chính) | COCO AP trung bình trên các ngưỡng IoU 0.5:0.95 |
| mAP50, mAP75 | |
| **AP_small** / AP_medium / AP_large | rất quan trọng với drone vì vật thể nhỏ chiếm đa số |
| AP theo từng lớp (10 lớp, nhấn mạnh 5 lớp tracking) | |
| AR@maxDets | đặt `maxDets=[1,100,500]` vì frame đông (giống VisDrone-DET) |
| Precision, Recall, F1 tại conf 0.5 | điểm làm việc mà tracker thực sự dùng |
| Params (M), GFLOPs, latency detector (ms/frame), FPS detector | đo trên cùng GPU, batch 1, warm-up 50 frame, có `torch.cuda.synchronize()` |
| Thời gian train, VRAM peak | |

### 4.2. Tracking (E1): mỗi tổ hợp detector × tracker, chạy TrackEval **từng lớp rồi lấy trung bình 5 lớp**

| Nhóm | Chỉ số |
|---|---|
| **Chính** | **HOTA**, **IDF1**, **MOTA** |
| Tách HOTA | DetA (chất lượng phát hiện), AssA (chất lượng liên kết ID), LocA |
| Định danh | IDSW, IDP, IDR, Frag |
| Theo vết | MT (%), ML (%) |
| Lỗi thô | FP, FN, MOTP |
| Tốc độ | FPS tracker riêng (không tính detector) + FPS end-to-end (detector + ReID + tracker) |

Lệnh: `run_mot_challenge.py --METRICS HOTA CLEAR Identity --BENCHMARK VisDrone --SPLIT_TO_EVAL <split>-<class>`.

**Dòng oracle** (GT làm detections, score = 1) × 5 tracker. Dòng này tách riêng chất lượng của tracker ra khỏi chất lượng của detector.

### 4.3. Các bảng kết quả

1. **Bảng detection:** 3 dòng (+ 3 dòng COCO zero-shot để so sánh): mAP50-95, mAP50, AP_s/m/l, P/R, Params, FPS_det.
2. **Ma trận tracking:** 3 detector × 5 tracker = 15 dòng + 5 dòng oracle. Các cột: HOTA, DetA, AssA, MOTA, IDF1, IDSW, MT, ML, FPS_e2e. Kèm heatmap HOTA và IDF1.
3. **Bảng theo lớp:** HOTA/IDF1 của 5 lớp cho cấu hình tốt nhất mỗi detector.
4. **Biểu đồ:** FPS_e2e vs HOTA (trade-off), và mAP vs HOTA (detector tốt hơn có kéo theo tracking tốt hơn không).

**Các ý cần phân tích trong báo cáo:**
- Thứ hạng tracker có giữ nguyên khi đổi detector không.
- DetA bị chi phối bởi detector, AssA bị chi phối bởi tracker.
- Khoảng cách giữa kết quả thật và oracle.
- ReID (DeepSORT, BoT-SORT) có giúp ích với vật thể nhỏ không.
- Ảnh hưởng của GMC khi camera drone chuyển động.

---

## 5. File và cấu trúc dùng chung

```
cfg/classes.yaml  cfg/visdrone.yaml  cfg/trackers/{sort,deepsort,bytetrack,ocsort,botsort}.yaml
data/splits/{train,val}.txt
core/tracker.py                      # Tracker(name,cfg).update(dets(N,6), frame) -> (M,7)
scripts/cache_dets_<model>.py        # mỗi người 1 file, cùng output format
scripts/run_tracker.py --dets results/dets/<det>/<split> --tracker <name> --out results/tracks/<det>/<tracker>/<split>
scripts/eval_det.py   --dets ... --gt data/visdrone_coco/<split>.json  -> results/tables/det_<det>_<split>.csv
scripts/eval_track.py --det <det> --split <split>                      -> results/tables/track_<det>_<split>.csv
results/tables/matrix_val_mean.csv  matrix_testdev_mean.csv
```

`det_id ∈ {yolo11m, frcnn_r50, rfdetr_m, oracle}`. Các CSV có cùng tên cột để ghép thẳng vào bảng tổng.

---

## 6. Tiến độ (tính từ tuần bắt đầu, mốc giữa kỳ ở tuần 4)

| Tuần | Thành – RF-DETR | Thi – Faster R-CNN | Tiến – YOLO11m |
|---|---|---|---|
| 1 | Repo, tải dữ liệu, splits, D2, `eval_det.py`, thống kê max box/frame | `core/tracker.py` + 5 cfg; test trên cache zero-shot | D1, D3, `eval_track.py`; YOLO11m COCO zero-shot → cache; chạy oracle × 5 tracker |
| 2 | Train RF-DETR | Train FRCNN (bắt đầu sớm vì chậm nhất) | Train YOLO11m |
| 3 | Train (tiếp) | Train (tiếp) | Cache val → 5 tracker → E0, E1 |
| 4 – **Giữa kỳ** | Tổng hợp + slide giữa kỳ; cache + 5 tracker nếu xong | cache + 5 tracker nếu xong | **Bảng YOLO11m × 5 tracker + oracle** (đủ yêu cầu "≥ 1 pipeline") |
| 5 | Dò ngưỡng (giai đoạn 2) | Dò ngưỡng | Dò ngưỡng |
| 6 | Chạy test-dev 1 lần cho cả 3 × 5, đo FPS e2e, ghép bảng, vẽ heatmap và trade-off (cả nhóm) | | |

Plan này không bao gồm dữ liệu Việt Nam. Nếu sau này bổ sung, phần đánh giá chéo miền dùng lại đúng các script ở mục 5.

---

## 7. Kiểm tra (verification)

- Sau khi tạo D1/D2/D3: vẽ nhãn lên 3 chuỗi và xem bằng mắt. Số ảnh trong `splits/train.txt` phải giống nhau giữa bản YOLO và bản COCO.
- `eval_det.py` chạy trên cache của YOLO11m phải cho mAP50 **xấp xỉ** (±1–2 điểm) kết quả `yolo val`. Dùng điều này để kiểm tra script.
- Oracle × ByteTrack phải cho HOTA và MOTA rất cao (MOTA > 90). Nếu thấp hơn thì lỗi nằm ở D3 hoặc ở bước lọc vùng ignored.
- Chạy lại cùng 1 tổ hợp 2 lần phải ra số giống hệt nhau.
- Kiểm tra đủ 15 + 5 dòng trong `matrix_val_mean.csv` trước khi chạy test-dev.
