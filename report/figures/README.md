# Ảnh cho báo cáo

Các đường dẫn nguồn dưới đây tính từ `D:/AdvanceCV/uav-tracking`.
Ảnh đã có được chép nguyên bản, chỉ đổi tên khi cần khớp LaTeX.

| Tên trong figures/ | Nguồn repo |
|---|---|
| `05_class_distribution.png` | `outputs/eda/figures/05_class_distribution.png` |
| `06_box_size.png` | `outputs/eda/figures/06_box_size.png` |
| `08_density.png` | `outputs/eda/figures/08_density.png` |
| `train_curve_yolo11m.png` | `results/yolo/T1_yolo11m_640_det/train/training_curves_loss_map.png` |
| `train_curve_frcnn.png` | `results/frcnn/T2_frcnn_r50_v2_det_seed0/figures/train_curve_frcnn_r50.png` |
| `detectors_same_frame.jpg` | `results/comparision/detectors_same_frames/comparison_01_uav0000009_03358_v_frame0000001.jpg` |
| `threshold_sweep_val.png` | `results/figures/threshold_sweep_val.png` |
| `heatmap_hota_testdev.png` | `results/figures/heatmap_hota_testdev.png` |
| `heatmap_idf1_testdev.png` | `results/figures/heatmap_idf1_testdev.png` |
| `oracle_gap.png` | `results/figures/oracle_gap.png` |
| `deta_vs_assa.png` | `results/figures/deta_vs_assa.png` |
| `per_class_hota.png` | `results/figures/per_class_hota.png` |
| `tradeoff_fps_hota.png` | `results/figures/tradeoff_fps_hota.png` |

## Ảnh chờ bổ sung

- `train_curve_rfdetr.png`: loss và mAP DET val của đúng run `T3det_rfdetr_m_576_det`, đánh dấu checkpoint được chọn.
- `tracks/track_01.jpg` đến `tracks/track_06.jpg`: minh họa YOLO11m + BoT-SORT; ghi rõ chuỗi, frame, ngưỡng 0,6 và vệt 30 frame. Tên cảnh trong LaTeX là tiêu chí chọn, có thể sửa theo ảnh thực tế.
- `recall_by_size.png`: AR_small, AR_medium, AR_large từ các dòng `ign` trong `results/tables/det_test-dev.csv`; không dùng bin <8/16/32 px vì chưa có phép đo đó.
- `err_idsw.jpg`: dải frame thể hiện ID đổi, kèm tracker và thời điểm; cần ít nhất hai frame.
- `err_fn_small.png`, `err_fp.png`: lỗi detection có nguồn ảnh/frame, ngưỡng và mô hình. Có ảnh tham khảo trong `results/yolo/T1_yolo11m_640_det/train/errors/`, nhưng phải xác nhận split/cấu hình trước khi dùng làm bằng chứng.
- `he_thong_cuoi_ky.png`: sơ đồ thiết kế hoặc giao diện minh họa hệ thống dự kiến; không ghi là sản phẩm đã hoàn thành.

Macro `\Hinh` tự hiển thị ảnh nếu đúng tên tệp, hoặc khung giữ chỗ nếu chưa có. Không cần sửa macro khi thêm ảnh.
Khi thay khung giữ chỗ bằng ảnh, cập nhật các câu/caption “chờ bổ sung” tương ứng và bổ sung nguồn ảnh.
