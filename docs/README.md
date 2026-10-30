# Tài liệu dự án

## Tài liệu kỹ thuật

Mô tả đầy đủ dự án, từ dữ liệu, tiền xử lý, mô hình đến đánh giá và quy trình làm việc. Nên đọc theo thứ tự.

| # | Tài liệu | Nội dung |
|---|---|---|
| 1 | [Tổng quan](technical/01_tong_quan.md) | Bài toán, mục tiêu, phân công, luồng xử lý, cấu trúc repo, trạng thái |
| 2 | [Dữ liệu](technical/02_du_lieu.md) | VisDrone2019-MOT: định dạng, lớp, và toàn bộ số liệu thống kê |
| 3 | [Tiền xử lý](technical/03_tien_xu_ly.md) | Các bước, quy tắc và căn cứ, định dạng đầu ra, DET cache, augmentation |
| 4 | [Detector](technical/04_detector.md) | RF-DETR, YOLO11m, Faster R-CNN: cấu hình, cách train, thí nghiệm, lỗi thường gặp |
| 5 | [Tracking và đánh giá](technical/05_tracking_danh_gia.md) | 5 tracker, tham số, chỉ số, oracle, bảng kết quả, câu hỏi phân tích |
| 6 | [Quy trình làm việc](technical/06_quy_trinh.md) | Local và server, đồng bộ, chạy job, git, ghi log, tiến độ |

## Kế hoạch và phân công

| Tài liệu | Nội dung |
|---|---|
| [Plan_3Detector_5Tracker.md](Plan_3Detector_5Tracker.md) | Kế hoạch 3 detector × 5 tracker, phân công, tham số, chỉ số |
| [Plan_FineTune_Inference.md](Plan_FineTune_Inference.md) | Danh sách đầy đủ các lần train, inference và đánh giá |
| [Plan_For_Application.md](Plan_For_Application.md) | Kế hoạch ứng dụng demo |
| [Task.md](Task.md) | Mô tả chi tiết từng nhiệm vụ |
| [Phan_cong_3Detector_5Tracker.xlsx](Phan_cong_3Detector_5Tracker.xlsx) | Bảng phân công và bảng điền kết quả |

Một số thông tin trong `Task.md` đã cũ so với tài liệu kỹ thuật. Ví dụ ràng buộc resolution "bội số của 56" của RF-DETR chỉ đúng với các bản cũ; bản 1.11.2 đang dùng yêu cầu bội số của 32. Khi có mâu thuẫn, tài liệu kỹ thuật là bản cập nhật.

## Tài liệu liên quan

| Tài liệu | Nội dung |
|---|---|
| [README chính](../README.md) | Hướng dẫn cài đặt và chạy |
| [data/README.md](../data/README.md) | Mô tả dữ liệu và công cụ tiền xử lý |
| [src/README.md](../src/README.md) | Vai trò từng thư mục mã nguồn |
| [01_eda_visdrone_mot.ipynb](../src/notebooks/01_eda_visdrone_mot.ipynb) | Notebook thống kê dữ liệu |
