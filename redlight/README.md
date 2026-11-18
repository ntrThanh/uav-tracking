# Traffic Vision — giám sát camera và sự kiện vượt đèn đỏ

Ứng dụng nội bộ cho camera cố định. Giao diện tại `/` gồm Tổng quan, Xem camera, Sự kiện vi phạm và Quản lý camera. Trình vẽ vùng nằm tại `/editor` và được mở từ từng camera. Cấu trúc màn hình lấy cảm hứng từ `tCamera-ATM-FE`, chỉ giữ các luồng cần cho đề tài giao thông.

## Chạy thử

Python 3.11 được khuyến nghị. Cài PyTorch phù hợp phần cứng trước, sau đó:

```bash
cd redlight
pip install -r requirements.txt
python scripts/run_web.py --port 8000
```

Mở `http://127.0.0.1:8000`. Muốn truy cập từ máy khác, đặt `REDLIGHT_ADMIN_TOKEN` rồi chạy `--host 0.0.0.0` qua HTTPS reverse proxy. Lệnh `run_web.py` từ chối mở ra ngoài localhost nếu thiếu token. Với HTTPS, đặt thêm `REDLIGHT_HTTPS=1` để cookie đăng nhập có cờ Secure. Chạy **một Uvicorn worker** vì tiến trình camera hiện được giữ trong bộ nhớ của ứng dụng.

## Quy trình cho người vận hành

1. **Thêm camera**: nhập tên, mã, RTSP URL hoặc tải video mẫu; chọn engine phát hiện và bám vết. URL chứa mật khẩu chỉ được lưu trên máy chủ, API chỉ trả địa chỉ đã che thông tin đăng nhập.
2. **Vẽ vùng**: nhấn “Vẽ vùng”, lấy ảnh từ chính nguồn camera, vẽ vạch dừng, chọn hướng xe, khoanh vùng đèn; vùng làn đường là tùy chọn. Lưu cấu hình.
3. **Bật xử lý**: ở Quản lý camera, nhấn “Bật”; xem hình xử lý tại Xem camera. Nếu kết nối hoặc cấu hình lỗi, lý do hiện cạnh camera.
4. **Xem sự kiện**: khi một track đi qua vạch đúng hướng và đèn được phân loại đỏ, hệ thống ghi ảnh toàn cảnh, ảnh vùng xảy ra sự kiện, tọa độ xe, thời điểm và biển số nếu đọc được. Người vận hành mở ảnh rồi chọn “Xác nhận vi phạm” hoặc “Không vi phạm”; trạng thái này được lưu cùng sự kiện.

Hai camera được xử lý bởi hai worker riêng trong cùng tiến trình. Mỗi camera chỉ mở một kết nối video cho xử lý; màn hình xem trực tiếp dùng ảnh mới nhất từ worker đó. Video tải lên được lặp lại để tiện trình diễn.

## Cấu trúc mở rộng engine

`src/operations/engines.py` định nghĩa hợp đồng `Detector` và `Tracker`, registry và factory. Monitor chỉ gọi `detect()` và `update()` nên có thể đăng ký engine mới mà không sửa vòng xử lý. Hiện có YOLO11n, YOLO11m cho detection; ByteTrack và IoU tracker cơ bản cho tracking. Cấu hình engine được lưu theo camera ở `config/cameras/<id>.json`; tham số chung vẫn ở `config/pipeline.json`.

`src/operations/rules.py` chứa quy tắc cắt vạch thuần, có test riêng. `src/operations/monitor.py` quản lý profile, worker, ảnh xem trực tiếp và sự kiện. Mô hình phát hiện biển số ở `weights/plate_yolo11s.pt` được gọi chỉ khi có sự kiện, tránh chạy trên mọi frame. `src/operations/plate.py` có adapter PaddleOCR tùy chọn. Nếu chưa cài PaddleOCR, hệ thống vẫn lưu crop biển số và ghi `ocr_unavailable`, không tạo chuỗi giả.

## Dữ liệu và bảo mật

Profile nguồn ở `config/cameras/` (quyền 0600), video tải lên ở `data/uploads/`, bằng chứng ở `outputs/events/`; các thư mục này được Git ignore. Không đưa RTSP URL hay ảnh bằng chứng vào Git. API được bảo vệ bằng token nếu `REDLIGHT_ADMIN_TOKEN` được đặt. Triển khai công khai cần HTTPS, quản lý bí mật, chính sách lưu/xóa bằng chứng, sao lưu và giám sát tiến trình theo hạ tầng thực tế.

## Giới hạn hiện tại

- Phân loại màu đèn theo ROI và quy tắc hình học chưa được hiệu chuẩn trên video giao thông có nhãn. Sự kiện là **ứng viên cần người kiểm tra**, chưa phù hợp làm căn cứ xử phạt tự động.
- OCR chỉ chạy khi cài adapter PaddleOCR tương thích. Môi trường phát triển hiện chưa có gói này; kiểm thử biển số cần video thực tế và đánh giá độ đúng chuỗi.
- Worker nằm trong một tiến trình và chưa có hàng đợi bền vững, tự khởi động lại sau sự cố hoặc quản lý lưu trữ lâu dài. Chưa có benchmark đa camera, kiểm thử tải RTSP dài hạn, độ trễ p95 hay kiểm thử bảo mật độc lập.

## Kiểm tra

```bash
python -m pytest tests/test_operations.py tests/test_traffic_light.py tests/test_config.py tests/test_geometry.py tests/test_tracker.py tests/test_video.py
```
