# 5. Tracking và đánh giá

## 5.1. Năm tracker

| Tracker | Liên kết dựa trên | ReID | Bù chuyển động camera (GMC) |
|---|---|:-:|:-:|
| SORT | Kalman + IoU, Hungarian | | |
| DeepSORT | Kalman + IoU + đặc trưng ngoại hình | ✓ | |
| ByteTrack | Kalman + IoU, liên kết hai tầng box score cao và thấp | | |
| OC-SORT | Kalman lấy quan sát làm trung tâm, bù quán tính hướng chuyển động | | |
| BoT-SORT | ByteTrack + ReID + GMC | ✓ | ✓ |

Mỗi tracker có hai phiên bản trong `src/trackers/`: bản nhóm tự cài đặt, dùng chung thành phần Kalman, ghép cặp và vòng đời track, và bản dùng thư viện với hậu tố `_lib`. Kiến trúc và cách thêm tracker mới nằm trong [src/trackers/README.md](../../src/trackers/README.md). DeepSORT và BoT-SORT dùng chung trọng số ReID `osnet_x0_25_msmt17.pt` để so sánh được với nhau; riêng mạng OSNet được nạp qua `boxmot`.

Ý nghĩa của từng tracker trong so sánh: SORT là mốc đơn giản nhất. ByteTrack cho thấy lợi ích của việc giữ cả box score thấp, rất quan trọng khi vật thể nhỏ thường có score thấp. OC-SORT xử lý tốt đối tượng bị che lâu và chuyển động phi tuyến. DeepSORT và BoT-SORT cho thấy đặc trưng ngoại hình có giúp ích với đối tượng chỉ vài chục pixel hay không. BoT-SORT còn bù chuyển động camera, phù hợp với drone.

## 5.2. Tham số

**Giai đoạn 1: mặc định, giống hệt cho cả 3 detector.**

| Tracker | Tham số |
|---|---|
| SORT | ngưỡng det vào 0,4; `max_age=30`; `min_hits=3`; `iou_threshold=0.3` |
| DeepSORT | ngưỡng det vào 0,4; `max_age=30`; `n_init=3`; `max_iou_distance=0.7`; `max_cosine_distance=0.2`; `nn_budget=100` |
| ByteTrack | `track_high_thresh=0.5`; `track_low_thresh=0.1`; `new_track_thresh=0.6`; `match_thresh=0.8`; `track_buffer=30` |
| OC-SORT | `det_thresh=0.5`; `max_age=30`; `min_hits=3`; `iou_threshold=0.3`; `delta_t=3`; `inertia=0.2`; `asso_func=iou` |
| BoT-SORT | như ByteTrack + `proximity_thresh=0.5`; `appearance_thresh=0.25`; `with_reid=True`; `cmc_method=sof` |

Thiết lập chung: chỉ đưa vào tracker 5 lớp đánh giá; chạy theo từng lớp (chỉ ghép box cùng lớp); `frame_rate=30`; lọc bỏ track nằm trong vùng ignored trước khi đánh giá.

**Giai đoạn 2: dò ngưỡng det cho từng detector.** Score của ba detector không cùng thang đo, ví dụ score của DETR thường thấp hơn YOLO, nên ngưỡng 0,5 có lợi cho mô hình này và bất lợi cho mô hình khác. Mỗi người dò ngưỡng det cao ∈ {0,3; 0,4; 0,5; 0,6} trên val cho detector của mình (4 giá trị × 5 tracker = 20 lần chạy), chọn theo HOTA. Báo cáo cả hai bảng: mặc định và đã dò. Test-dev chỉ chạy một lần với cấu hình đã chọn.

Thống kê dữ liệu hỗ trợ một số lựa chọn tham số. 7–11% quỹ đạo có đoạn đứt, nên `track_buffer`/`max_age` 30 frame là hợp lý và có thể thử thêm 60. IoU liên tiếp trung vị khoảng 0,9 và chỉ 1,6–4,2% cặp có IoU dưới 0,3, nên ngưỡng IoU 0,3 của SORT hiếm khi là nguyên nhân mất liên kết.

## 5.3. Chỉ số đánh giá detection

Tính bằng `src/evaluation/eval_det.py` (sẽ viết) dùng pycocotools trên DET cache, so với `data/processed/coco/<split>.json`.

| Chỉ số | Ý nghĩa |
|---|---|
| **mAP50-95** (chính) | AP trung bình trên 10 ngưỡng IoU 0,50:0,05:0,95, trung bình trên 10 lớp |
| mAP50, mAP75 | AP ở ngưỡng IoU 0,5 và 0,75 |
| **AP_small**, AP_medium, AP_large | AP theo kích thước, rất quan trọng với dữ liệu drone |
| AP theo lớp | Nhấn mạnh 5 lớp tracking |
| AR@1, AR@100, AR@500 | Recall trung bình theo số detection tối đa, đặt `maxDets=[1,100,500]` vì frame đông |
| Precision, Recall, F1 tại score 0,5 | Điểm làm việc thực tế của tracker |
| Params, GFLOPs, ms/frame, FPS | Cùng GPU, batch 1, khởi động 50 frame, có `torch.cuda.synchronize()` |
| Thời gian train, VRAM đỉnh | Chi phí huấn luyện |

Detection rơi vào vùng ignored (khóa `ignored_regions`) phải được loại trước khi tính, nếu không chúng bị tính là false positive. Để kiểm tra script, mAP50 tính trên DET cache của YOLO phải lệch không quá 1–2 điểm so với `yolo val`.

## 5.4. Chỉ số đánh giá tracking

Dùng TrackEval với `--METRICS HOTA CLEAR Identity`, chạy **từng lớp rồi lấy trung bình 5 lớp** theo cách làm của VisDrone. Script là `src/evaluation/eval_track.py`. Trong mỗi frame, box GT và box tracker có hơn một nửa diện tích nằm trong vùng ignored bị loại, giống `dropObjectsInIgr` của bộ công cụ VisDrone. Trung bình 5 lớp lấy trung bình cộng các chỉ số phần trăm và lấy tổng các chỉ số đếm (IDSW, FP, FN, Frag, MT, ML).

| Nhóm | Chỉ số | Ý nghĩa |
|---|---|---|
| Chính | **HOTA** | Trung bình hình học của DetA và AssA, cân bằng giữa phát hiện và liên kết |
| Chính | **IDF1** | F1 của việc gán đúng ID, nhạy với lỗi liên kết |
| Chính | **MOTA** | 1 − (FN + FP + IDSW) / GT, chịu chi phối mạnh bởi phát hiện |
| Tách HOTA | DetA, AssA, LocA | Chất lượng phát hiện, chất lượng liên kết, độ chính xác vị trí |
| Định danh | IDSW, IDP, IDR, Frag | Số lần đổi ID, precision và recall của ID, số lần đứt quỹ đạo |
| Theo vết | MT (%), ML (%) | Tỉ lệ quỹ đạo được theo dõi ≥ 80% và ≤ 20% thời gian |
| Lỗi thô | FP, FN, MOTP | Dương tính giả, âm tính giả, độ chính xác vị trí |
| Tốc độ | FPS tracker, FPS end-to-end | Riêng tracker và toàn bộ detector + ReID + tracker |

HOTA được chọn làm chỉ số chính vì tách được rõ phần đóng góp của detector (DetA) và của tracker (AssA), đúng với câu hỏi nghiên cứu của dự án.

## 5.5. Oracle

Oracle dùng ground truth làm detection (score bằng 1) rồi chạy cả 5 tracker. Kết quả oracle là giới hạn trên của từng tracker khi detector hoàn hảo. Khoảng cách giữa kết quả thật và oracle cho biết bao nhiêu phần sai số đến từ detector. Đây cũng là bước kiểm tra pipeline: Oracle × ByteTrack phải cho MOTA trên 90, nếu thấp hơn thì lỗi nằm ở ground truth cho TrackEval hoặc ở bước lọc vùng ignored.

## 5.6. Các bảng và biểu đồ kết quả

| # | Nội dung | Kích thước |
|---|---|---|
| 1 | Bảng detection: mAP50-95, mAP50, AP_s/m/l, P/R, Params, FPS | 3 mô hình fine-tune + 3 zero-shot |
| 2 | Ma trận tracking: HOTA, DetA, AssA, MOTA, IDF1, IDSW, MT, ML, FPS_e2e | 15 dòng + 5 dòng oracle |
| 3 | Heatmap HOTA và IDF1 của ma trận 3 × 5 | |
| 4 | HOTA và IDF1 theo 5 lớp cho cấu hình tốt nhất mỗi detector | |
| 5 | Biểu đồ FPS_e2e theo HOTA (đánh đổi tốc độ và chất lượng) | |
| 6 | Biểu đồ mAP theo HOTA (detector tốt hơn có kéo theo tracking tốt hơn) | |
| 7 | T3 so với T3-hr: AP_small, mAP, HOTA, FPS | Nếu chạy T3-hr |

Tất cả bảng ghi ở `results/tables/` với cùng tên cột để ghép thẳng thành bảng tổng: `matrix_val_mean.csv`, `matrix_testdev_mean.csv`.

## 5.7. Các câu hỏi cần phân tích

Thứ hạng tracker có giữ nguyên khi đổi detector hay không. Nếu giữ nguyên, kết luận về tracker là bền vững; nếu đổi, cần giải thích bằng đặc điểm detection của từng mô hình như phân bố score hay tỉ lệ bỏ sót box nhỏ.

DetA bị chi phối bởi detector còn AssA bị chi phối bởi tracker. Với IoU liên tiếp cao như thống kê đã cho thấy, giả thuyết là DetA sẽ khác nhau nhiều giữa các detector, còn AssA khác nhau ít hơn giữa các tracker.

ReID có giúp ích với vật thể chỉ vài chục pixel không. Đặc trưng ngoại hình trích từ crop 20–40 px có thể kém tin cậy, nên DeepSORT có thể không vượt ByteTrack.

GMC có tác dụng khi camera chuyển động không. So sánh BoT-SORT với ByteTrack trên các chuỗi có chuyển động mạnh như `uav0000249_*`.

Khoảng cách giữa kết quả thật và oracle lớn hay nhỏ, và lớp nào có khoảng cách lớn nhất.
