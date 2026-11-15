# Kế hoạch viết báo cáo giữa kỳ và định hướng cuối kỳ

Đề tài: Theo vết đa đối tượng trong video drone, so sánh 3 detector × 5 tracker trên VisDrone2019-MOT.
Căn cứ: [requirement.docx](requirement.docx) (mẫu báo cáo của thầy) và toàn bộ kết quả hiện có trong `results/`.
Ngày lập: 10/11/2026.

Tài liệu gồm 10 phần. Phần 1–2 trả lời "báo cáo phải chứng minh điều gì". Phần 3–5 là dàn ý, danh mục bảng và hình. Phần 6–7 là việc phải làm thêm và các lỗi phải tránh để báo cáo chặt chẽ. Phần 8 là tài liệu tham khảo. Phần 9–10 là định hướng cuối kỳ và tiến độ.

---

## 1. Đối chiếu yêu cầu của thầy

| Yêu cầu trong requirement.docx | Cách đáp ứng | Trạng thái |
|---|---|---|
| Lề 2–2–2,5–2 cm, Times New Roman 13, giãn dòng 1,3–1,5, căn đều | Dùng template LaTeX hoặc Word đặt sẵn (mục 3.0) | Cần làm |
| 15–20 trang (tối thiểu 10, tối đa 35), không tính phụ lục mã nguồn | Phân bổ trang ở mục 3, mục tiêu 18 trang | Cần làm |
| Hình, bảng có số, có tiêu đề, được tham chiếu trong văn bản | Mỗi hình/bảng ở mục 4–5 đều ghi rõ đoạn nào tham chiếu nó | Cần làm |
| Không chèn mã nguồn, chỉ pseudo-code | 1 thuật toán pseudo-code cho vòng lặp tracking-by-detection (mục 3, phần 1.1.4) | Cần làm |
| Kết quả số trình bày bằng bảng, không chụp màn hình | Mọi số lấy từ CSV trong `results/tables/`, gõ lại thành bảng | Có số liệu |
| Công thức soạn thảo, không dán ảnh | Kalman, IoU, HOTA, MOTA, IDF1 gõ bằng LaTeX/Equation | Cần làm |
| Nộp code/link repo, file hướng dẫn chạy | Repo GitHub + `README.md`. **README mục 10 "Trạng thái" đang ghi "Train 3 detector: Chưa bắt đầu", phải cập nhật** | Cần sửa |
| Ít nhất 5 ảnh minh họa đầu ra hệ thống | 6 ảnh tracking có ID và vệt quỹ đạo (Hình F11) + ảnh detection đã có | Cần tạo ảnh tracking |
| Ít nhất 2 phương pháp so sánh ở một nhiệm vụ chính | 3 detector và 5 tracker, so sánh trên cùng DET cache | Đã có |
| Mô tả dữ liệu: nguồn, lấy mẫu, quy ước nhãn, phân chia, làm sạch | Mục 1.2, có sẵn số liệu EDA trong `outputs/eda/` | Đã có số liệu |
| Sơ đồ pipeline tổng thể | Hình F1, phải vẽ mới | Cần tạo |
| Ít nhất một bảng so sánh phương pháp | Bảng T5 (detection), T6–T7 (ma trận tracking) | Đã có số liệu |
| Ít nhất một mục phân tích lỗi | Mục 2.3 có 4 tiểu mục phân tích lỗi | Cần viết + 3 hình mới |
| Kết luận về tính ứng dụng thực tiễn | Mục Kết luận dựa trên bảng đánh đổi tốc độ và độ chính xác (T8, F9) | Đã có số liệu |

---

## 2. Xương sống của báo cáo: luận điểm và bằng chứng

Báo cáo chặt chẽ là báo cáo mà mỗi luận điểm đều trỏ được tới một bảng hoặc một hình, và mỗi bảng, hình đều phục vụ một luận điểm. Dưới đây là 7 luận điểm nên đặt làm trục. Cách phát biểu đã được cân theo đúng mức mà số liệu cho phép, không nói quá.

### L1. Nút thắt chính nằm ở khâu phát hiện, không phải khâu liên kết

| | |
|---|---|
| Bằng chứng | Bảng T7 (oracle), Hình F7 (`results/figures/oracle_gap.png`), Hình F8 (`deta_vs_assa.png`) |
| Số liệu | Khi dùng ground truth làm detection, HOTA đạt 84,3–94,9 tùy tracker. Với detector thật, HOTA chỉ còn 42,9–51,7, tức mất khoảng 38–45 điểm. DetA giảm từ 90–97 xuống 39–43, trong khi AssA giảm từ 79–92 xuống 48–63 |
| Cách phát biểu | "Khoảng cách giữa oracle và detector thật lớn hơn nhiều so với khoảng cách giữa tracker tốt nhất và tệ nhất (khoảng 5–9 điểm HOTA), cho thấy chất lượng phát hiện là yếu tố giới hạn chính." Không nói "100% lỗi do detector" vì AssA cũng giảm (detection bị đứt đoạn làm hỏng liên kết) |
| Điểm yếu cần vá | Nên thêm một câu giải thích vì sao AssA cũng giảm: detection bỏ sót làm quỹ đạo bị đứt, nên một phần lỗi liên kết thực chất có gốc từ detector |

### L2. mAP gần như ngang nhau nhưng HOTA khác nhau, nên mAP không đủ để chọn detector cho tracking

| | |
|---|---|
| Bằng chứng | Bảng T5 (detection), Hình F6 (`map_vs_hota.png`), Hình F5 (`threshold_sweep_val.png`) |
| Số liệu | mAP50-95 trên test-dev (đã lọc vùng ignored): YOLO11m 23,17, RF-DETR 22,96, Faster R-CNN 22,45, chênh nhau 0,7 điểm. Trên val thứ hạng lại đảo (Faster R-CNN 24,15 cao nhất). Nhưng với cùng BoT-SORT, RF-DETR thấp hơn khoảng 1,8 HOTA. Ngưỡng tối ưu khác nhau rõ: RF-DETR tốt nhất ở 0,4–0,5 và sụp còn 29,4 HOTA ở 0,8; YOLO sụp còn 17,8 ở 0,8; Faster R-CNN vẫn tăng tới 0,9 |
| Cách phát biểu | "Ba detector không phân biệt được bằng mAP (chênh dưới 1 điểm, thứ hạng đảo giữa val và test-dev), nhưng phân bố score của chúng khác nhau rất nhiều, nên điểm làm việc (ngưỡng) mới quyết định chất lượng tracking." Đây là lý do giao thức dò ngưỡng riêng cho từng detector là bắt buộc |
| Điểm yếu cần vá | Hình F6 hiện có trục x chỉ từ 33,75 đến 35,75 nên nhìn như khác biệt lớn. Phải ghi rõ trong chú thích hoặc vẽ lại với trục x rộng hơn. Nên bổ sung khoảng tin cậy (mục 6, P0-1) để khẳng định "không khác biệt có ý nghĩa" |

### L3. BoT-SORT tốt nhất với mọi detector, và lợi thế đến từ khâu liên kết

| | |
|---|---|
| Bằng chứng | Bảng T6, Hình F3 (`heatmap_hota_testdev.png`), Hình F4 (`heatmap_idf1_testdev.png`), Hình F8 |
| Số liệu | HOTA test-dev đã dò ngưỡng: BoT-SORT 51,67 / 51,68 / 49,85 (YOLO / Faster R-CNN / RF-DETR), hơn ByteTrack 3,9–5,0 điểm. Phần hơn gần như nằm hết ở AssA (63,0 / 63,3 / 60,0 so với 54,7 / 53,7 / 52,9), DetA chỉ hơn 1,4–1,8 |
| Cách phát biểu | "BoT-SORT đứng đầu với cả ba detector, khoảng cách chủ yếu đến từ AssA (+7 đến +10 điểm) chứ không từ DetA." |
| Điểm yếu cần vá | Cần kiểm định theo từng chuỗi (đếm số chuỗi BoT-SORT thắng trong 17 chuỗi test-dev, bootstrap CI), mục 6, P0-1 |

### L4. Bù chuyển động camera quan trọng hơn ReID trên video drone (giả thuyết, cần ablation)

| | |
|---|---|
| Bằng chứng hiện có | DeepSORT (có ReID, không có GMC) gần như bằng SORT: chênh +1,06 / +0,41 / −0,24 HOTA. BoT-SORT (ByteTrack + ReID + GMC) hơn ByteTrack 3,9–5,0. Như vậy ReID đơn lẻ gần như không giúp, phần lợi khả năng cao đến từ GMC |
| Cách phát biểu nếu chưa có ablation | Chỉ được nói "kết quả gợi ý rằng GMC đóng góp chính", kèm câu "chưa tách được riêng tác động của từng thành phần" ở mục hạn chế |
| Cách làm cho chặt (khuyến nghị làm trước khi nộp giữa kỳ) | Ablation BoT-SORT bốn biến thể trên cache detection sẵn có (mục 6, P0-2). Chạy trên CPU, không cần train. Khi có kết quả thì L4 thành luận điểm khẳng định được |
| Lý giải lý thuyết | OSNet trọng số MSMT17 học trên ảnh người chụp ngang, đối tượng VisDrone nhìn từ trên cao và chỉ vài chục pixel, nên đặc trưng ngoại hình kém phân biệt. Ngược lại camera drone di chuyển liên tục làm dự đoán Kalman lệch, GMC sửa đúng lỗi này |

### L5. OC-SORT hưởng lợi nhiều nhất khi detection sạch nhưng không vượt ByteTrack khi detection nhiễu

| | |
|---|---|
| Bằng chứng | Bảng T7: oracle OC-SORT 89,99 > ByteTrack 86,48; với detector thật OC-SORT 43,8–46,5 < ByteTrack 46,0–47,1 |
| Cách phát biểu | OC-SORT dựa vào quan sát (observation-centric re-update, observation momentum), nên khi quan sát chính xác thì rất mạnh, còn khi detection thiếu và nhiễu thì cơ chế này mất tác dụng. ByteTrack tận dụng box score thấp nên bù được phần detection bị bỏ sót. Đây là phân tích "vì sao A tốt hơn B" mà thầy yêu cầu ở mục 2.3 |

### L6. Lỗi tập trung ở đối tượng nhỏ và lớp pedestrian

| | |
|---|---|
| Bằng chứng | Bảng T4 (AP theo lớp), Hình F10 (`per_class_hota.png`), EDA (Hình F2b, kích thước box), Hình F12 mới (recall theo kích thước) |
| Số liệu | AP_small chỉ 8,0–8,4 so với AP_large 39,7–47,7. Test-dev có 47,8% box nhỏ hơn 32² px so với 24,9% ở train. Pedestrian có HOTA thấp nhất (36,7–42,2) so với car (59,7–61,6). Faster R-CNN, với ảnh vào lớn nhất (cạnh dài 1333), cho HOTA pedestrian cao nhất (42,17) |
| Cách phát biểu | Lệch phân phối kích thước giữa train và test-dev cộng với độ phân giải đầu vào thấp (576–640) là nguyên nhân chính của lỗi bỏ sót. Đây là cầu nối sang định hướng cuối kỳ |

### L7. Về ứng dụng thực tế: YOLO11m là lựa chọn cân bằng nhất

| | |
|---|---|
| Bằng chứng | Bảng T8 (tốc độ), Hình F9 (`tradeoff_fps_hota.png`), Bảng T5 cột Params/GFLOPs |
| Số liệu | YOLO11m + ByteTrack: 101 FPS end-to-end, HOTA 47,1. YOLO11m + BoT-SORT: 25,9 FPS, HOTA 51,7. Faster R-CNN + BoT-SORT: cùng HOTA 51,7 nhưng chỉ 9,2 FPS và 745 GFLOPs so với 40,7. Khâu front-end của BoT-SORT (GMC + ReID) là nút thắt tốc độ, chỉ khoảng 30–33 FPS |
| Cách phát biểu | Hai cấu hình đề xuất: YOLO11m + BoT-SORT khi cần độ chính xác ở tốc độ gần thời gian thực, YOLO11m + ByteTrack khi cần chạy thời gian thực trên phần cứng yếu. Faster R-CNN không phù hợp để triển khai dù chính xác ngang |

### Luận điểm phụ về dữ liệu (đưa vào mục 1.2)

RF-DETR huấn luyện trên frame MOT (lấy mẫu stride 2) bị quá khớp theo chuỗi: ở epoch 16 mAP trên train đạt 0,58 nhưng val chỉ 0,178, checkpoint tốt nhất rơi vào epoch 1 (`results/diagnostics/overfit_epoch1.json`, `overfit_epoch16.json`). Nguyên nhân là các frame liên tiếp gần như trùng lặp, 56 chuỗi train thực chất là 56 cảnh. Vì vậy nhóm chuyển sang huấn luyện cả ba detector trên VisDrone2019-DET (6.414 ảnh độc lập), sau khi loại 57 ảnh DET trùng video với MOT val/test-dev để tránh rò rỉ dữ liệu (`data/splits/det_train_excluded.txt`). Mô hình DET cho mAP test-dev 22,4 so với 19,8 của mô hình MOT (Bảng T3). Đây là một quyết định có thực nghiệm chứng minh, nên trình bày rõ vì thầy yêu cầu "chiến lược lấy mẫu" và "các bước làm sạch".

Lưu ý: số trong file diagnostics tính trên tập con (test-dev chỉ 1.332 frame) bằng script chẩn đoán riêng, phải ghi chú như vậy dưới Bảng T3 và không trộn với Bảng T5.

---

## 3. Dàn ý chi tiết báo cáo giữa kỳ

### 3.0. Định dạng

Khuyến nghị LaTeX trên Overleaf, biên dịch XeLaTeX với `fontspec` (Times New Roman) và `babel`/`polyglossia` tiếng Việt, `geometry` đặt `top=2cm, bottom=2cm, left=2.5cm, right=2cm`, `\linespread{1.3}` (tương đương 1,3–1,5 trong Word), `\usepackage{booktabs}` cho bảng, `\usepackage{algorithm2e}` cho pseudo-code, `biblatex` style `ieee`. Nếu dùng Word thì làm Heading 1–3, Caption có đánh số tự động, Cross-reference để tham chiếu hình/bảng.

Thống nhất ngôn ngữ: báo cáo viết tiếng Việt, giữ thuật ngữ tiếng Anh in nghiêng ở lần đầu kèm giải thích (ví dụ *tracking-by-detection*, theo vết dựa trên phát hiện). Dùng dấu phẩy thập phân (51,67) xuyên suốt, kể cả trong bảng. Tên phương pháp viết thống nhất: YOLO11m, Faster R-CNN, RF-DETR-M, SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT. Lời nhận xét dưới bảng/hình viết thành đoạn văn ngắn, lồng số liệu vào câu, không gạch đầu dòng.

### 3.1. Phân bổ trang (mục tiêu khoảng 18 trang nội dung)

| Phần | Trang | Hình | Bảng |
|---|---|---|---|
| Bìa, Tóm tắt | 1 + 0,5 | | |
| Giới thiệu | 1,5 | | |
| 1.1 Phương pháp | 4 | F1 | T1 |
| 1.2 Dữ liệu và tiền xử lý | 2,5 | F2 (a–c) | T2, T3 |
| 2.1 Thiết lập thực nghiệm | 1,5 | | T4' (tham số tracker) |
| 2.2 Thực nghiệm | 3,5 | F3, F4, F5, F11 | T4, T5, T6 |
| 2.3 Phân tích và thảo luận | 3 | F6–F10, F12–F14 | T7, T8 |
| Kết luận và hướng phát triển | 0,75 | | |
| Tài liệu tham khảo | 1,5 | | |
| Phụ lục (không tính) | tùy | ảnh định tính thêm, cấu trúc thư mục | bảng đầy đủ val, per-class |

### 3.2. Nội dung từng mục

**Bìa.** Tên trường, khoa, học phần "Thị giác máy tính nâng cao", tên đề tài, giảng viên, ba thành viên kèm MSSV, lớp, tháng 11/2026. Theo requirement mục B.1.

**Tóm tắt (150–250 từ).** Một đoạn duy nhất có đủ năm ý thầy yêu cầu. Bài toán: theo vết đa đối tượng trong video drone với đối tượng nhỏ, dày và camera chuyển động. Ứng dụng: giám sát giao thông, đếm phương tiện. Dữ liệu: VisDrone2019-MOT, 5 lớp đánh giá. Phương pháp: ma trận 3 detector đại diện ba họ kiến trúc × 5 tracker, cùng DET cache, dò ngưỡng trên val, đánh giá một lần trên test-dev, thêm thí nghiệm oracle. Kết quả chính: cấu hình tốt nhất đạt HOTA khoảng 51,7, oracle cho thấy detector là nút thắt (mất khoảng 40 điểm HOTA), mAP chênh dưới 1 điểm nhưng HOTA chênh tới 1,8, BoT-SORT tốt nhất nhờ AssA, YOLO11m + ByteTrack đạt khoảng 100 FPS.

**Giới thiệu (1,5 trang).** Đoạn 1 nêu động lực: MOT là nền của giám sát giao thông và an ninh; drone mở rộng tầm nhìn nhưng tạo thách thức riêng. Đoạn 2 nêu thách thức cụ thể có số liệu từ EDA (cạnh trung vị 33 px ở test-dev, 47,8% box nhỏ, trung bình 46 box/frame, tối đa 152), cite [1], [2]. Đoạn 3 nêu khoảng trống: các tracker phổ biến được thiết kế và đánh giá chủ yếu trên MOT17/MOT20 với camera cố định, người đi bộ cỡ lớn [27], [28]; chưa rõ chúng hoạt động thế nào và thành phần nào hỏng khi chuyển sang video drone. Đoạn 4 nêu mục tiêu và phạm vi, viết rõ ba câu hỏi nghiên cứu: (Q1) với cùng giao thức, detector nào và tracker nào tốt nhất; (Q2) lỗi đến từ phát hiện hay liên kết; (Q3) cấu hình nào phù hợp triển khai thực tế. Đoạn 5 liệt kê đóng góp dưới dạng câu văn: pipeline đánh giá thống nhất và tái lập được, ma trận 15 tổ hợp + oracle, phân tích lỗi phân rã theo DetA/AssA. Kết thúc bằng một câu nói phạm vi: giữa kỳ là benchmark và phân tích, cuối kỳ là cải tiến và ứng dụng.

**Chương 1. Phương pháp và dữ liệu**

*1.1.1 Pipeline tổng thể (0,5 trang).* Hình F1 kèm đoạn mô tả từng khối: video → detector (3 lựa chọn) → DET cache `frame,-1,x,y,w,h,score,class` với ngưỡng 0,05 → lọc 5 lớp và ngưỡng det → tracker (5 lựa chọn: dự đoán Kalman, GMC tùy chọn, trích đặc trưng ReID tùy chọn, ghép cặp Hungarian, quản lý vòng đời track) → loại vùng ignored → TrackEval. Nhấn mạnh thiết kế "cache một lần, mọi tracker đọc chung" là điều kiện để so sánh công bằng.

*1.1.2 Ba detector (1,5 trang).* Bảng T1 so sánh họ kiến trúc, backbone, số tham số, GFLOPs, có cần NMS hay không, kích thước ảnh vào. Mỗi detector một đoạn: Faster R-CNN R50-FPN v2 là hai giai đoạn, RPN + RoI head, FPN [7], [8], [9]; YOLO11m là một giai đoạn, ba mức feature, NMS [10], [11], [12]; RF-DETR-M là transformer end-to-end, DINOv2 backbone, ghép cặp một-một nên không cần NMS, 300 query [13]–[17]. Mỗi đoạn kết thúc bằng một câu dự đoán lý thuyết về đối tượng nhỏ (ví dụ patch 16 px của ViT bất lợi cho vật thể nhỏ hơn 8 px sau resize) để mục 2.3 quay lại kiểm chứng. Có thể đặt công thức hàm mất mát ghép cặp Hungarian của DETR nếu còn chỗ.

*1.1.3 Năm tracker (1,5 trang).* Bắt đầu bằng phần chung: mô hình Kalman vận tốc hằng với trạng thái $\mathbf{x}=[u,v,s,r,\dot u,\dot v,\dot s]^\top$ (SORT) hoặc $[x,y,w,h,\dot x,\dot y,\dot w,\dot h]^\top$ (BoT-SORT), ma trận chi phí $C_{ij}=1-\mathrm{IoU}(\hat b_i,d_j)$, giải bằng thuật toán Hungarian [22], [23]. Sau đó mỗi tracker một đoạn chỉ nêu cái nó thêm vào: SORT [18] là mốc; DeepSORT [19] thêm khoảng cách cosine đặc trưng ngoại hình và matching cascade; ByteTrack [20] ghép hai tầng, tầng hai dùng box score thấp; OC-SORT [21] thêm ORU, OCM, OCR; BoT-SORT [24] thêm GMC bằng sparse optical flow [32], [33] và ReID, với chi phí kết hợp $\min(d^{IoU}, d^{cos})$ có ngưỡng. Kết thúc bằng bảng nhỏ (gộp trong T1 hoặc T1b) đánh dấu ✓ thành phần mỗi tracker có: Kalman, ghép hai tầng, ReID, GMC. Ghi rõ ReID dùng chung OSNet-x0.25 MSMT17 [25], [26] cho DeepSORT và BoT-SORT để công bằng, và tracker do nhóm tự cài đặt có đối chứng với bản thư viện.

*1.1.4 Pseudo-code (0,3 trang).* Thuật toán 1: vòng lặp tracking-by-detection tổng quát cho một chuỗi, có các nhánh "nếu có GMC thì biến đổi dự đoán", "nếu có ReID thì kết hợp chi phí", "nếu là ByteTrack/BoT-SORT thì ghép tầng hai với box score thấp". Một thuật toán mô tả được cả 5 tracker là cách gọn và đẹp nhất.

*1.2 Dữ liệu và tiền xử lý (2,5 trang).* Nguồn dữ liệu: VisDrone2019-MOT và VisDrone2019-DET [1], [2], quay ở nhiều thành phố Trung Quốc, 7 độ phân giải từ 680×382 đến 3840×2160. Bảng T2 phân chia: số chuỗi, frame, box, quỹ đạo, box/frame, % box nhỏ cho train/val/test-dev (số có sẵn trong README và `outputs/eda/tables/01_split_overview.csv`), thêm dòng DET train 6.414 ảnh. Quy ước nhãn: 12 mã lớp, mã 0 là vùng ignored, mã 11 others bị bỏ, train 10 lớp, đánh giá tracking 5 lớp theo giao thức VisDrone; giải thích ý nghĩa cột score, truncation, occlusion. Hình F2 gồm ba ô (a) phân bố lớp, (b) phân bố kích thước box theo split, (c) mật độ box/frame, lấy từ `outputs/eda/figures/05_class_distribution.png`, `06_box_size.png`, `08_density.png`. Chiến lược lấy mẫu và làm sạch: stride 2 cho frame MOT train và lý do (độ dư thừa giữa frame), loại box có kích thước không hợp lệ và box thuộc lớp 0/11, chuyển đổi sang COCO/YOLO/MOT bằng một bộ công cụ chung `prepare.py`, kiểm tra chéo bằng `check.py`, loại 57 ảnh DET trùng video với val/test-dev. Bảng T3 và đoạn văn về quá khớp theo chuỗi (luận điểm phụ ở mục 2). Kết thúc bằng đoạn nêu sự lệch phân phối: test-dev có tỷ lệ box nhỏ gấp đôi train, dự báo trước khó khăn ở mục 2.3.

**Chương 2. Thực nghiệm và đánh giá**

*2.1 Thiết lập thực nghiệm (1,5 trang).* Phần cứng và phần mềm: GPU huấn luyện (RF-DETR trên 2× NVIDIA L4; YOLO11m và Faster R-CNN trên Kaggle, ghi rõ GPU), GPU đo tốc độ (một NVIDIA L4, batch 1, 905 frame, có khởi động và đồng bộ CUDA), phiên bản PyTorch, torchvision, Ultralytics, rfdetr 1.11.2, TrackEval. Bảng siêu tham số huấn luyện ba detector (có sẵn trong `docs/technical/04_detector.md` mục 4.2). Kịch bản thực nghiệm viết thành ba giai đoạn: E0 đánh giá detection, E1 ma trận tracking với tham số mặc định, E2 dò ngưỡng det trên val ∈ {0,3…0,8} theo HOTA rồi chạy test-dev đúng một lần, cộng E3 oracle. Chỉ số đánh giá: COCO mAP50-95, mAP50, AP_S/M/L, AR [29]; HOTA, DetA, AssA [30]; MOTA [31]; IDF1 [34]; IDSW, MT, ML; FPS. Gõ công thức HOTA và MOTA, IDF1. Ghi rõ cách lấy trung bình 5 lớp (trung bình cộng chỉ số phần trăm, cộng tổng chỉ số đếm), cách loại vùng ignored (box có hơn 50% diện tích trong vùng ignored), và lý do chọn HOTA làm chỉ số chính. Bảng T4' (có thể đưa vào phụ lục nếu thiếu chỗ) tham số mặc định 5 tracker.

*2.2 Thực nghiệm (3,5 trang)*, trình bày theo thứ tự thầy yêu cầu, từng phương pháp rồi đến so sánh.

2.2.1 Kết quả phát hiện: Bảng T5 và Bảng T4 (AP theo lớp), Hình F13 (đường cong huấn luyện ba detector, có sẵn cho YOLO và Faster R-CNN, cần lấy log RF-DETR). Một đoạn nhận xét theo luận điểm L2 phần đầu.

2.2.2 Kết quả tracking với tham số mặc định: chỉ nêu ngắn, đưa bảng đầy đủ vào phụ lục, nhấn mạnh MOTA âm của Faster R-CNN trên val ở ngưỡng 0,5 (−2,86) để dẫn sang lý do phải dò ngưỡng.

2.2.3 Dò ngưỡng trên val: Hình F5 và một đoạn văn theo L2 phần sau.

2.2.4 Ma trận cuối trên test-dev: Bảng T6 (15 dòng) và Hình F3, F4. Đoạn nhận xét theo L3.

2.2.5 Kết quả định tính: Hình F11 (6 ảnh tracking), đây là phần "ít nhất 5 ảnh minh họa đầu ra".

*2.3 Phân tích và thảo luận (3 trang).*

2.3.1 Phân rã lỗi bằng oracle (Bảng T7, Hình F7, F8), luận điểm L1 và L5.

2.3.2 Phân tích lỗi phát hiện: lỗi bỏ sót theo kích thước (Hình F12) và theo lớp (Hình F10, Bảng T4), luận điểm L6. Thêm đoạn về dương tính giả: RF-DETR sinh 1,38 triệu box trên test-dev so với 0,71 triệu của YOLO ở cùng ngưỡng cache 0,05, chứng tỏ phân bố score của nó trải rộng hơn.

2.3.3 Phân tích lỗi liên kết: IDSW và Frag theo tracker, ví dụ một cặp ảnh ID switch (Hình F14), tương quan lợi ích AssA của BoT-SORT với độ lớn chuyển động camera theo chuỗi (mục 6, P1-2), và kết quả ablation BoT-SORT (Bảng T9 nếu làm), luận điểm L4.

2.3.4 Trường hợp tốt và xấu: chọn 2 chuỗi HOTA cao nhất và 2 chuỗi thấp nhất (từ kết quả theo chuỗi, P0-1), mô tả đặc điểm cảnh (độ cao, mật độ, ban đêm, chuyển động camera).

2.3.5 Đánh đổi tốc độ và độ chính xác: Bảng T8, Hình F9, luận điểm L7.

2.3.6 Hạn chế: một seed duy nhất cho mỗi detector; độ phân giải đầu vào ba detector không bằng nhau (576, 640, 800–1333) nên so sánh detector có gắn yếu tố độ phân giải; lưới ngưỡng của Faster R-CNN chạm biên (mục 7); ReID chưa huấn luyện lại trên dữ liệu drone; chưa đánh giá trên dữ liệu ngoài miền (giao thông Việt Nam).

**Kết luận và hướng phát triển (0,75 trang).** Đoạn 1 trả lời lần lượt Q1, Q2, Q3 bằng số. Đoạn 2 kết luận tính ứng dụng thực tiễn (bắt buộc theo checklist): cấu hình đề xuất, tốc độ, hạn chế khi triển khai (đối tượng rất nhỏ, ban đêm). Đoạn 3 nêu hướng cuối kỳ suy ra trực tiếp từ phân tích lỗi (mục 9).

**Tài liệu tham khảo.** IEEE, đánh số theo thứ tự xuất hiện, khoảng 25–35 mục (mục 8).

**Phụ lục.** A: bảng đầy đủ val và test-dev mặc định. B: HOTA theo lớp cho 15 tổ hợp. C: thêm 6–10 ảnh định tính (detection ba mô hình cùng frame trong `results/comparision/detectors_same_frames/`). D: cấu trúc thư mục mã nguồn và hướng dẫn chạy tóm tắt, link repo.

---

## 4. Danh mục bảng

Mỗi bảng ghi nguồn CSV để không ai phải nhập số bằng tay sai. Viết một script nhỏ xuất CSV sang LaTeX (`pandas.DataFrame.to_latex`) để số trong báo cáo khớp tuyệt đối với kết quả.

| Mã | Tên bảng | Nguồn | Cột nên giữ | Tham chiếu ở |
|---|---|---|---|---|
| T1 | So sánh kiến trúc ba detector (và thành phần năm tracker) | `docs/technical/04_detector.md`, `results/tables/model_stats.csv` | Họ, backbone, Params (M), GFLOPs, ảnh vào, NMS | 1.1.2, 1.1.3 |
| T2 | Thống kê và phân chia dữ liệu | `outputs/eda/tables/01_split_overview.csv`, README | Split, số chuỗi, frame, box, quỹ đạo, box/frame, % small, vai trò | 1.2 |
| T3 | Ảnh hưởng nguồn dữ liệu huấn luyện (RF-DETR, MOT frame vs DET) | `results/diagnostics/*.json` | Nguồn train, checkpoint, mAP train/val/test-dev, AP_small | 1.2 |
| T4 | AP50-95 theo lớp trên test-dev | `results/tables/det_test-dev_per_class.csv` (variant `ign`) | 10 lớp × 3 detector, tô đậm max mỗi dòng, đánh dấu 5 lớp tracking | 2.2.1, 2.3.2 |
| T4' | Tham số mặc định 5 tracker | `src/configs/trackers/*.yaml` | Như mục 5.2 của tài liệu kỹ thuật | 2.1 |
| T5 | Kết quả detection trên val và test-dev | `results/tables/det_val.csv`, `det_test-dev.csv` (variant `ign`), `model_stats.csv`, `fps_bench.csv` | mAP50-95, mAP50, AP_S, AP_M, AP_L, AR_500, mAP_track5, P/R@0,5, Params, GFLOPs, FPS_det. Thêm dòng YOLO11m COCO zero-shot (`results/yolo/.../metrics_comparison.csv`, mAP 8,6) để chứng minh fine-tune là cần thiết | 2.2.1 |
| T6 | Ma trận tracking 3 × 5 trên test-dev (ngưỡng đã dò) | `results/tables/matrix_testdev_mean.csv`, các dòng `_thrXX` theo `tuned_val.csv` | Ngưỡng, HOTA, DetA, AssA, MOTA, IDF1, IDSW, MT%, ML%, FPS_e2e. Tô đậm tốt nhất mỗi detector, gạch chân tốt nhì | 2.2.4 |
| T7 | Oracle và khoảng cách với detector thật | `matrix_testdev_mean.csv` dòng `oracle`, `results/figures/oracle_gap.csv` | Tracker, HOTA/DetA/AssA oracle, HOTA tốt nhất của detector thật, chênh lệch | 2.3.1 |
| T8 | Tốc độ và chi phí | `results/tables/fps_bench.csv`, `model_stats.csv` | FPS_det, FPS_tracker, FPS front-end, FPS_e2e cho 15 tổ hợp, hoặc rút gọn 6 tổ hợp tiêu biểu | 2.3.5 |
| T9 (mới, khuyến nghị) | Ablation thành phần BoT-SORT | Sinh từ P0-2 | Biến thể, HOTA, AssA, IDF1, IDSW, FPS | 2.3.3 |
| T10 (mới, khuyến nghị) | So sánh từng cặp theo chuỗi | Sinh từ P0-1 | Cặp so sánh, ΔHOTA trung bình, CI 95%, số chuỗi thắng/17 | 2.2.4, 2.3 |

Quy tắc trình bày bảng: số liệu 1–2 chữ số thập phân thống nhất, đơn vị ở tiêu đề cột, mũi tên ↑/↓ cho biết chỉ số càng cao hay càng thấp càng tốt, chú thích bảng đặt phía trên bảng, nguồn và ghi chú đặt dưới bảng.

---

## 5. Danh mục hình

Tất cả hình trong `results/figures/` hiện có tiêu đề tiếng Anh nằm trong ảnh. Khi đưa vào báo cáo tiếng Việt phải vẽ lại: bỏ tiêu đề trong ảnh (dùng caption của báo cáo), đổi nhãn trục và chú giải sang tiếng Việt, dấu phẩy thập phân, xuất PDF/SVG vector hoặc PNG 300 dpi, cỡ chữ trong hình xấp xỉ cỡ chữ chú thích. Chỉ cần sửa `src/evaluation/make_figures.py` thêm một tham số ngôn ngữ.

| Mã | Nội dung | Trạng thái | Nguồn hoặc cách tạo | Chứng minh |
|---|---|---|---|---|
| F1 | Sơ đồ pipeline tổng thể | **Tạo mới** | Vẽ bằng draw.io hoặc TikZ: video → 3 detector → DET cache → 5 tracker (Kalman, GMC, ReID, ghép cặp, vòng đời) → lọc ignored → TrackEval. Thêm nhánh oracle (GT → tracker) bằng nét đứt | Bắt buộc theo checklist |
| F2 | EDA (a) phân bố lớp, (b) kích thước box theo split, (c) mật độ box/frame | Có sẵn, cần gộp | `outputs/eda/figures/05_class_distribution.png`, `06_box_size.png`, `08_density.png` | Mô tả dữ liệu, lệch phân phối |
| F3 | Heatmap HOTA 3 × 5 trên test-dev | Có sẵn, cần vẽ lại tiếng Việt | `results/figures/heatmap_hota_testdev.png` | L3 |
| F4 | Heatmap IDF1 (và MOTA nếu đủ chỗ) | Có sẵn | `heatmap_idf1_testdev.png`, `heatmap_mota_testdev.png` | L3 |
| F5 | HOTA theo ngưỡng det trên val | Có sẵn, nên vẽ lại với lưới mở rộng tới 0,9 | `threshold_sweep_val.png`, `results/tables/extended/tuned_val_extended.csv` | L2 |
| F6 | mAP so với HOTA | Có sẵn, **phải sửa trục x** | `map_vs_hota.png` | L2 |
| F7 | Oracle so với detector thật | Có sẵn | `oracle_gap.png` | L1 |
| F8 | DetA so với AssA | Có sẵn | `deta_vs_assa.png` | L1, L3 |
| F9 | Đánh đổi FPS và HOTA (trục FPS thang log) | Có sẵn | `tradeoff_fps_hota.png` | L7 |
| F10 | HOTA theo lớp | Có sẵn | `per_class_hota.png` | L6 |
| F11 | **6 ảnh đầu ra tracking** có box, ID, vệt quỹ đạo 30 frame | **Tạo mới** | Script mới `src/inference/render_tracks.py` đọc `results/tracks/<det>/<tracker>/test-dev/<seq>.txt`. Chọn 6 cảnh đa dạng: ngã tư đông, đường cao tốc, ban đêm, camera xoay, người đi bộ dày, độ cao lớn. Dùng cấu hình tốt nhất YOLO11m + BoT-SORT | Yêu cầu ≥ 5 ảnh |
| F12 | Recall theo kích thước box (bin theo cạnh sqrt(wh): <8, 8–16, 16–32, 32–64, >64 px) cho 3 detector ở ngưỡng đã dò | **Tạo mới** | Script mới `src/evaluation/recall_by_size.py`, ghép GT và DET cache bằng IoU ≥ 0,5. Chồng histogram số box GT theo bin ở trục phụ | L6 |
| F13 | Đường cong huấn luyện ba detector (loss và mAP val theo epoch) | Có một phần | `results/yolo/.../training_curves_loss_map.png`, `results/frcnn/.../train_curve_frcnn_r50.png`, log RF-DETR trên server | Thiết lập, chọn checkpoint |
| F14 | Ví dụ lỗi: (a) ID switch ở SORT nhưng BoT-SORT giữ được, (b) bỏ sót đối tượng nhỏ, (c) dương tính giả | **Tạo mới** | Dùng lại `render_tracks.py`, chọn frame có IDSW bằng cách so khớp track với GT. Có thể dùng `results/yolo/.../train/errors/01_FN_small.png`, `02_FP.png`, `03_class_confusion.png` làm (b), (c) nếu đủ rõ | Phân tích lỗi |
| F15 (tùy chọn) | Ba detector trên cùng một frame | Có sẵn | `results/comparision/detectors_same_frames/comparison_04_*.jpg` | Định tính detection |

Ảnh đầu ra hệ thống khi đưa vào báo cáo nên cắt vùng quan tâm và phóng to, vì ảnh drone gốc 2–4K thu nhỏ vào nửa trang thì box nhỏ sẽ không thấy được.

---

## 6. Việc phải làm thêm trước khi nộp giữa kỳ

Tất cả đều chạy được trên cache detection và track đã có trong `results/dets/`, `results/tracks/`, không cần GPU, không cần train lại. Theo quy ước nhóm, viết code ở máy cá nhân, commit, rồi chạy.

### P0, bắt buộc để báo cáo chặt chẽ

**P0-1. Đánh giá theo từng chuỗi và kiểm định thống kê.** Test-dev chỉ có 17 chuỗi, một seed, nên chênh lệch 1–2 điểm HOTA có thể là ngẫu nhiên. Sửa `src/evaluation/eval_track.py` để lưu kết quả TrackEval theo từng chuỗi (TrackEval đã tính sẵn, hiện chỉ lấy `COMBINED_SEQ`). Sau đó viết `src/evaluation/stats.py` tính, với mỗi cặp cần so sánh (BoT-SORT với ByteTrack trên từng detector, YOLO với RF-DETR trên BoT-SORT, v.v.), ΔHOTA trung bình, khoảng tin cậy 95% bằng paired bootstrap theo chuỗi (10.000 lần lấy mẫu lại), và số chuỗi thắng trên 17. Kết quả đưa vào Bảng T10. Từ đó báo cáo được viết theo dạng "BoT-SORT hơn ByteTrack ở a/17 chuỗi, ΔHOTA = x [CI dưới; CI trên]" (điền số thật sau khi chạy) thay vì chỉ nêu trung bình. Đây là thay đổi làm báo cáo chặt chẽ nhất với chi phí thấp nhất. Kết quả theo chuỗi cũng dùng cho mục 2.3.4 (trường hợp tốt và xấu).

**P0-2. Ablation BoT-SORT.** Chạy BoT-SORT trên val và test-dev với bốn biến thể bằng cách đổi cấu hình `with_reid` và `cmc_method`: đầy đủ, bỏ ReID, bỏ GMC (`cmc_method: none`), bỏ cả hai (về gần ByteTrack). Cùng ngưỡng đã dò của từng detector. 4 biến thể × 3 detector × 2 split = 24 lần chạy, mỗi lần vài phút. Kết quả vào Bảng T9, biến luận điểm L4 từ "gợi ý" thành "khẳng định". Nếu chỉ đủ thời gian thì làm với YOLO11m trước.

**P0-3. Ảnh đầu ra tracking (F11) và sơ đồ pipeline (F1).** Điều kiện bắt buộc của thầy.

**P0-4. Vẽ lại hình bằng tiếng Việt, sửa trục F6.**

**P0-5. Cập nhật README** mục 10 "Trạng thái" và mục 7 cho đúng hiện trạng, thêm mục "Tái lập kết quả trong báo cáo" chỉ rõ lệnh sinh từng bảng, hình.

### P1, nên làm để phần phân tích lỗi sâu hơn

**P1-1. Recall theo kích thước (F12).** Chứng minh định lượng L6, chỉ ra ngưỡng kích thước mà detector bắt đầu bỏ sót.

**P1-2. Tương quan chuyển động camera với lợi ích của GMC.** Với mỗi chuỗi test-dev, ước lượng độ lớn chuyển động camera trung bình bằng optical flow thưa (đúng phương pháp `sof` mà BoT-SORT dùng): trung vị độ dịch chuyển của các điểm đặc trưng giữa hai frame liên tiếp. Vẽ scatter trục x là chuyển động camera, trục y là ΔAssA (BoT-SORT đầy đủ trừ BoT-SORT không GMC) cho 17 chuỗi, tính hệ số tương quan Spearman. Nếu tương quan dương rõ, đây là bằng chứng rất thuyết phục cho L4 và là hình đắt giá nhất của mục 2.3.

**P1-3. Ví dụ ID switch (F14a).** Dùng kết quả theo chuỗi tìm chuỗi có chênh IDSW lớn nhất giữa SORT và BoT-SORT.

### P2, nếu còn thời gian

**P2-1.** Thống nhất vấn đề ngưỡng Faster R-CNN (mục 7, lỗi 3).
**P2-2.** Bảng khả năng lặp lại: chạy lại một tổ hợp với tracker bản thư viện (`_lib`) để chứng minh bản tự cài đặt cho kết quả tương đương. Một dòng trong phụ lục là đủ, nhưng thầy sẽ đánh giá cao vì nhóm tự cài đặt tracker.
**P2-3.** Zero-shot cho Faster R-CNN và RF-DETR để T5 có đủ ba dòng zero-shot.

---

## 7. Những lỗi làm báo cáo mất chặt chẽ, cần tránh

1. **Dùng test-dev để chọn tham số.** Giao thức đã đúng (chọn trên val, test-dev chạy một lần). Phải viết rõ câu này trong 2.1 và không đưa các run test-dev khác ngưỡng vào bảng chính như thể là lựa chọn.

2. **Kết luận thứ hạng detector từ chênh lệch mAP dưới 1 điểm.** Thứ hạng mAP đảo giữa val và test-dev. Chỉ được nói ba detector tương đương về mAP trên tập này.

3. **Ngưỡng tối ưu nằm ở biên lưới tìm kiếm.** Với Faster R-CNN, lưới 0,3–0,8 chọn 0,8, nhưng lưới mở rộng (`results/tables/extended/tuned_val_extended.csv`) cho thấy HOTA val còn tăng tới 0,9 (49,83 với BoT-SORT). Đồng thời đã có run test-dev ở 0,85 và 0,6 trong `extended/`. Cách xử lý trung thực: bảng chính giữ giao thức lưới 0,3–0,8 cho cả ba detector; trong 2.2.3 ghi một câu rằng Faster R-CNN đạt tối ưu ở biên, lưới mở rộng trên val chọn 0,9; nếu muốn dùng kết quả mở rộng thì chạy test-dev đúng một lần với 0,9 (ngưỡng do val chọn), không chọn 0,85 vì 0,85 không phải ngưỡng val chọn. Ghi điều này vào mục hạn chế.

4. **So sánh detector khi độ phân giải đầu vào khác nhau.** Faster R-CNN dùng 800–1333, YOLO 640, RF-DETR 576. Đây là cấu hình mặc định của từng họ và là lựa chọn hợp lý, nhưng phải nói rõ là yếu tố gây nhiễu, nhất là khi giải thích Faster R-CNN tốt với pedestrian.

5. **Trộn số từ các script đánh giá khác nhau.** Số mAP trong `results/diagnostics/` (tập con, script chẩn đoán), trong `results/yolo/.../comparison/` (DET test-dev) và trong `results/tables/det_*.csv` (MOT, pycocotools trên cache) không cùng giao thức. Bảng chính chỉ dùng `results/tables/`. Các số khác dùng ở chỗ riêng, có ghi chú.

6. **Báo cáo MOTA làm chỉ số chính.** MOTA bị chi phối bởi FP/FN và có thể âm. Giải thích một lần vì sao chọn HOTA, sau đó luôn đọc HOTA trước, IDF1 sau, MOTA để tham khảo.

7. **Nói "ReID không có tác dụng" khi ReID chưa được huấn luyện cho dữ liệu drone.** Phát biểu đúng là "ReID huấn luyện trên ảnh người (MSMT17) không mang lại lợi ích rõ rệt trên VisDrone".

8. **Hình không được nhắc trong văn bản, hoặc chú thích quá ngắn.** Mỗi chú thích hình nên tự đọc được: nêu cái gì, trên split nào, cấu hình nào, đọc ra điều gì.

9. **FPS không nêu điều kiện đo.** Ghi GPU, batch, độ phân giải, có hoặc không tính thời gian đọc ảnh, số frame đo.

10. **Gán quá mức nguyên nhân.** Phân biệt "kết quả cho thấy" (có ablation hoặc kiểm định) và "chúng tôi cho rằng" (lý giải chưa kiểm chứng).

---

## 8. Tài liệu tham khảo

Định dạng IEEE. Cột "Dùng cho" chỉ chỗ cần cite. Các mục đánh dấu (*) cần kiểm tra lại thông tin xuất bản trên Google Scholar trước khi nộp, vì là preprint hoặc phần mềm.

| # | Tài liệu | Dùng cho |
|---|---|---|
| [1] | P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, H. Ling, "Detection and tracking meet drones challenge," *IEEE TPAMI*, vol. 44, no. 11, pp. 7380–7399, 2022. | Dữ liệu, giới thiệu |
| [2] | L. Wen et al., "VisDrone-MOT2019: The vision meets drone multiple object tracking challenge results," *ICCV Workshops*, 2019. | Giao thức đánh giá 5 lớp |
| [3] | D. Du et al., "The unmanned aerial vehicle benchmark: Object detection and tracking," *ECCV*, 2018. (UAVDT) | Giới thiệu, bối cảnh dữ liệu drone |
| [4] | S. Liu, X. Li, H. Lu, Y. He, "Multi-object tracking meets moving UAV," *CVPR*, 2022. | Giới thiệu, chuyển động camera drone |
| [5] | W. Luo et al., "Multiple object tracking: A literature review," *Artificial Intelligence*, vol. 293, 2021. | Giới thiệu, tracking-by-detection |
| [6] | T.-Y. Lin et al., "Microsoft COCO: Common objects in context," *ECCV*, 2014. | Trọng số khởi tạo, mAP |
| [7] | S. Ren, K. He, R. Girshick, J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," *NeurIPS*, 2015. | Faster R-CNN |
| [8] | T.-Y. Lin et al., "Feature pyramid networks for object detection," *CVPR*, 2017. | FPN |
| [9] | Y. Li, S. Xie, X. Chen, P. Dollár, K. He, R. Girshick, "Benchmarking detection transfer learning with vision transformers," arXiv:2111.11429, 2021. | Faster R-CNN v2 của torchvision |
| [10] | J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You only look once: Unified, real-time object detection," *CVPR*, 2016. | Họ YOLO |
| [11] | G. Jocher, J. Qiu, "Ultralytics YOLO11," 2024. [Online]. github.com/ultralytics/ultralytics (*) | YOLO11 |
| [12] | R. Khanam, M. Hussain, "YOLOv11: An overview of the key architectural enhancements," arXiv:2410.17725, 2024. (*) | Kiến trúc YOLO11 |
| [13] | N. Carion et al., "End-to-end object detection with transformers," *ECCV*, 2020. | DETR, ghép cặp Hungarian |
| [14] | X. Zhu et al., "Deformable DETR: Deformable transformers for end-to-end object detection," *ICLR*, 2021. | RF-DETR kế thừa |
| [15] | Q. Chen et al., "LW-DETR: A transformer replacement to YOLO for real-time detection," arXiv:2406.03459, 2024. (*) | RF-DETR kế thừa |
| [16] | M. Oquab et al., "DINOv2: Learning robust visual features without supervision," *TMLR*, 2024. | Backbone RF-DETR |
| [17] | I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, N. Peri, "RF-DETR: Neural architecture search for real-time detection transformers," arXiv:2511.09554, 2025. (*) | RF-DETR |
| [18] | A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft, "Simple online and realtime tracking," *ICIP*, 2016. | SORT |
| [19] | N. Wojke, A. Bewley, D. Paulus, "Simple online and realtime tracking with a deep association metric," *ICIP*, 2017. | DeepSORT |
| [20] | Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," *ECCV*, 2022. | ByteTrack |
| [21] | J. Cao, J. Pang, X. Weng, R. Khirodkar, K. Kitani, "Observation-centric SORT: Rethinking SORT for robust multi-object tracking," *CVPR*, 2023. | OC-SORT |
| [22] | R. E. Kalman, "A new approach to linear filtering and prediction problems," *J. Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960. | Kalman |
| [23] | H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Research Logistics Quarterly*, vol. 2, pp. 83–97, 1955. | Ghép cặp |
| [24] | N. Aharon, R. Orfaig, B.-Z. Bobrovsky, "BoT-SORT: Robust associations multi-pedestrian tracking," arXiv:2206.14651, 2022. | BoT-SORT, GMC |
| [25] | K. Zhou, Y. Yang, A. Cavallaro, T. Xiang, "Omni-scale feature learning for person re-identification," *ICCV*, 2019. | OSNet |
| [26] | L. Wei, S. Zhang, W. Gao, Q. Tian, "Person transfer GAN to bridge domain gap for person re-identification," *CVPR*, 2018. | MSMT17 |
| [27] | A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler, "MOT16: A benchmark for multi-object tracking," arXiv:1603.00831, 2016. | Giới thiệu, khoảng trống |
| [28] | P. Dendorfer et al., "MOT20: A benchmark for multi object tracking in crowded scenes," arXiv:2003.09003, 2020. | Giới thiệu, khoảng trống |
| [29] | (dùng lại [6]) chỉ số COCO AP/AR | Chỉ số detection |
| [30] | J. Luiten et al., "HOTA: A higher order metric for evaluating multi-object tracking," *IJCV*, vol. 129, pp. 548–578, 2021. | HOTA, DetA, AssA |
| [31] | K. Bernardin, R. Stiefelhagen, "Evaluating multiple object tracking performance: The CLEAR MOT metrics," *EURASIP J. Image and Video Processing*, 2008. | MOTA, MOTP |
| [32] | B. D. Lucas, T. Kanade, "An iterative image registration technique with an application to stereo vision," *IJCAI*, 1981. | GMC bằng optical flow |
| [33] | J. Shi, C. Tomasi, "Good features to track," *CVPR*, 1994. | GMC, chọn điểm đặc trưng |
| [34] | E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi, "Performance measures and a data set for multi-target, multi-camera tracking," *ECCV Workshops*, 2016. | IDF1 |
| [35] | J. Luiten, A. Hoffhues, "TrackEval," 2020. [Online]. github.com/JonathonLuiten/TrackEval | Công cụ đánh giá |

Dành cho cuối kỳ (cite khi làm): F. C. Akyon, S. O. Altinuc, A. Temizel, "Slicing aided hyper inference and fine-tuning for small object detection," *ICIP*, 2022 (SAHI); X. Zhu et al., "TPH-YOLOv5," *ICCV Workshops*, 2021; G. D. Evangelidis, E. Z. Psarakis, "Parametric image alignment using enhanced correlation coefficient maximization," *IEEE TPAMI*, 2008 (ECC); Y. Zhao et al., "DETRs beat YOLOs on real-time object detection," *CVPR*, 2024 (RT-DETR); Y. Du et al., "StrongSORT: Make DeepSORT great again," *IEEE TMM*, 2023.

Nếu muốn có một dòng "đặt kết quả của nhóm trong bối cảnh", có thể so sánh với kết quả đã công bố trên VisDrone2019-MOT test-dev (ví dụ trong [1], [2], [4]) nhưng phải ghi rõ khác biệt giao thức (họ báo cáo MOTA/IDF1 bằng toolkit VisDrone, nhóm báo cáo HOTA bằng TrackEval), và kiểm tra số trực tiếp trong bài gốc.

---

## 9. Định hướng cuối kỳ

### 9.1. Nguyên tắc

Giữa kỳ là bài toán "đo và giải thích". Cuối kỳ nên là "cải tiến có kiểm chứng", và mỗi cải tiến phải xuất phát từ một lỗi đã được đo ở giữa kỳ. Như vậy hai báo cáo nối thành một câu chuyện liền mạch, và báo cáo cuối kỳ có thể giữ nguyên chương 1, chỉ mở rộng chương 2 và thêm chương ứng dụng. Giữ nguyên giao thức (cùng split, cùng cache, chọn trên val, test-dev một lần, bootstrap theo chuỗi) để mọi cải tiến so sánh trực tiếp được với bảng giữa kỳ.

### 9.2. Các hướng, xếp theo mức ưu tiên

| Mã | Hướng | Xuất phát từ lỗi nào | Thí nghiệm | Chỉ số chứng minh | Chi phí |
|---|---|---|---|---|---|
| C1 | **Tăng khả năng phát hiện đối tượng nhỏ** | L1, L6: detector là nút thắt, AP_small khoảng 8, test-dev có 48% box nhỏ | (a) Suy luận cắt ảnh SAHI với YOLO11m, không cần train lại, thử kích thước mảnh 640 với chồng lấn 0,2. (b) Tăng độ phân giải: YOLO11m 1280 hoặc RF-DETR-M 1024 (cấu hình `rfdetr_m_1024.yaml` đã có). Mỗi biến thể chỉ đổi một yếu tố | AP_small, recall theo kích thước (F12), DetA, HOTA, FPS | (a) thấp, (b) trung bình (train lại 1 mô hình khoảng 3–6 giờ trên L4) |
| C2 | **Cải tiến khâu liên kết cho camera drone** | L4, kết quả ablation P0-2 | (a) So sánh phương pháp GMC: sparse optical flow, ECC, ORB. (b) Fine-tune ReID OSNet trên crop đối tượng VisDrone MOT train (ID có sẵn), đánh giá bằng mAP/Rank-1 ReID và AssA. (c) `track_buffer` 30 so với 60 | AssA, IDF1, IDSW, Frag, FPS | Thấp đến trung bình |
| C3 | **Tổng quát hóa sang giao thông Việt Nam** | Hạn chế đã nêu, giả thuyết H4 trong kế hoạch nghiên cứu | Quay hoặc thu thập 10–20 đoạn video góc cao ở Việt Nam, gán nhãn bán tự động: chạy cấu hình tốt nhất, sửa nhãn trong CVAT. Đánh giá zero-shot và sau fine-tune nhẹ. Thêm lớp motor | HOTA, mAP trên tập VN, chênh lệch so với VisDrone | Cao (gán nhãn tốn thời gian) |
| C4 | **Ứng dụng trình diễn** | L7: đã xác định cấu hình triển khai | Web app Gradio hoặc Streamlit: tải video, chọn detector và tracker, hiển thị video kết quả, đếm phương tiện qua vạch, bản đồ nhiệt mật độ, thống kê theo thời gian. Đánh giá độ chính xác đếm trên vài chuỗi test-dev có GT | Sai số đếm (%), FPS trên GPU và CPU | Trung bình |

Khuyến nghị cho cuối kỳ: **C1 + C2 là bắt buộc** (phần nghiên cứu, mỗi cái ít nhất một bảng ablation), **C4 làm ở mức gọn** (đáp ứng "tính ứng dụng thực tiễn" và để demo khi bảo vệ), **C3 chỉ làm nếu còn thời gian**, ở quy mô nhỏ và trình bày như một thử nghiệm sơ bộ.

### 9.3. Câu hỏi nghiên cứu cho cuối kỳ

Q4: Tăng độ phân giải hiệu dụng (cắt ảnh hoặc tăng kích thước đầu vào) cải thiện HOTA bao nhiêu, và với chi phí tốc độ bao nhiêu? Kỳ vọng DetA tăng rõ, AssA tăng theo vì quỹ đạo ít đứt hơn.

Q5: Trong BoT-SORT, thành phần nào đóng góp vào AssA trên video drone, và ReID huấn luyện trên dữ liệu drone có thay đổi kết luận đó không?

Q6 (nếu làm C3): Hiệu năng giảm bao nhiêu khi chuyển sang giao thông Việt Nam, và lớp nào giảm mạnh nhất?

### 9.4. Bảng và hình dự kiến cho cuối kỳ

Bảng ablation phát hiện đối tượng nhỏ (baseline, SAHI, độ phân giải cao, kết hợp) với AP_small, DetA, HOTA, FPS. Bảng ablation liên kết (GMC × ReID × buffer). Biểu đồ Pareto HOTA–FPS cập nhật có thêm các cấu hình mới. Bảng so sánh với kết quả giữa kỳ có ΔHOTA và CI 95%. Ảnh trước và sau cải tiến trên cùng frame. Ảnh chụp giao diện ứng dụng (đưa vào phụ lục theo mẫu của thầy).

---

## 10. Tiến độ và phân công đề xuất

Mốc tính theo tuần kể từ hôm nay (10/11/2026). Điều chỉnh theo hạn nộp thực tế.

| Tuần | Giữa kỳ | Người |
|---|---|---|
| 1 | P0-1 (kết quả theo chuỗi + bootstrap), P0-2 (ablation BoT-SORT) | Tiến (đánh giá tracking), Thi (tracker) |
| 1 | P0-3 sơ đồ pipeline và ảnh tracking, P0-4 vẽ lại hình | Thành |
| 1 | Viết 1.1 và 1.2 | Thi (phần tracker), Thành (detector, dữ liệu) |
| 2 | P1-1, P1-2, P1-3 | Thành, Tiến |
| 2 | Viết 2.1, 2.2, 2.3, kết luận, tóm tắt | Cả nhóm, Thành tổng hợp |
| 2 | Rà soát theo mục 1 và mục 7, cập nhật README, nộp | Cả nhóm |

| Giai đoạn sau giữa kỳ | Cuối kỳ | Người |
|---|---|---|
| Tuần 3–4 | C1 (SAHI với YOLO11m trước, sau đó RF-DETR 1024) | Thành, Tiến |
| Tuần 3–4 | C2 (ablation GMC, fine-tune ReID) | Thi |
| Tuần 5 | C4 ứng dụng | Tiến |
| Tuần 5–6 | C3 nếu kịp, viết báo cáo cuối kỳ, slide, video demo | Cả nhóm |

Tiêu chí hoàn thành giữa kỳ: đủ 13 dòng trong bảng đối chiếu ở mục 1 ở trạng thái hoàn thành, mỗi luận điểm L1–L7 có ít nhất một bảng hoặc hình được tham chiếu trong văn bản, không vi phạm mục nào trong danh sách ở mục 7, và báo cáo nằm trong 15–20 trang.
