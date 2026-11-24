# Tài liệu ôn vấn đáp đồ án thị giác máy tính nâng cao

Tài liệu này tổng hợp theo mã nguồn, cấu hình, báo cáo và bảng kết quả hiện có trong dự án UAV Tracking, cập nhật ngày 24/11/2026. Nên học theo thứ tự: bài toán và giao thức; dữ liệu; kiến trúc ba detector; năm tracker; chỉ số; kết quả; giới hạn và ứng dụng.

Các số phần trăm dưới đây là điểm phần trăm khi so sánh hai kết quả, trừ khi nói rõ là tỉ lệ tương đối.

## 1. Câu trả lời mở đầu trong khoảng hai phút

Đề tài nghiên cứu bài toán theo vết nhiều đối tượng trên video UAV bằng phương pháp tracking by detection. Mỗi frame được detector xác định hộp bao, lớp và điểm tin cậy; tracker liên kết các hộp ở các frame liên tiếp thành quỹ đạo có ID.

Nhóm so sánh ba kiến trúc phát hiện đại diện cho ba hướng: Faster R-CNN ResNet-50 FPN v2 hai giai đoạn, YOLO11m một giai đoạn và RF-DETR Medium dựa trên transformer; kết hợp lần lượt với SORT, DeepSORT, ByteTrack, OC-SORT và BoT-SORT. Tổng cộng có 15 tổ hợp.

Dữ liệu là VisDrone2019; detector chính huấn luyện trên 6.414 ảnh DET train đã lọc trùng nguồn video, chọn checkpoint trên DET val 548 ảnh. Tracker chọn ngưỡng trên MOT val 2.846 frame, rồi báo cáo trên MOT test-dev 6.635 frame.

Detector dự đoán 10 lớp; đánh giá tracking chỉ trên năm lớp pedestrian, car, van, truck và bus. Chỉ số chính là HOTA, xem thêm IDF1, MOTA và FPS.

Trên MOT test-dev, BoT-SORT cho HOTA cao nhất với cả ba detector: 51,68 với Faster R-CNN, 51,67 với YOLO11m và 49,85 với RF-DETR-M. Chênh lệch 0,01 giữa hai tổ hợp đầu quá nhỏ để khẳng định bên nào tốt hơn nói chung.

Nếu ưu tiên tốc độ theo phép đo xử lý trên NVIDIA L4, YOLO11m + ByteTrack đạt 101,0 FPS và HOTA 47,09; nếu ưu tiên ID, YOLO11m + BoT-SORT đạt HOTA 51,67 và 25,9 FPS. Khi cấp hộp ground truth thay detector cho tracker, HOTA oracle đạt 84,30–94,93.

Khoảng cách lớn này gợi ý detection là một nút thắt quan trọng, nhưng không thể quy toàn bộ chênh lệch là lỗi riêng detector vì thay đầu vào làm đổi điều kiện ghép cặp. Bài toán khó do đối tượng nhỏ, đông, che khuất và góc nhìn camera di động.

## 2. Bài toán, ký hiệu và phạm vi

Detection là tìm các đối tượng trong một ảnh, mỗi dự đoán có hộp b = (x, y, w, h), lớp c và score s. Tracking là gán một ID ổn định cho cùng đối tượng qua thời gian.

Multiple object tracking (MOT) phải xử lý đồng thời nhiều đối tượng, bao gồm xuất hiện mới, biến mất, che khuất, giao cắt, thay đổi tỷ lệ và camera chuyển động. Tracking by detection đặt detector trước tracker; tracker không tự khôi phục vật thể không tạo ra tín hiệu trong ảnh.

ID chỉ có ý nghĩa bên trong một video hoặc một phiên camera, không phải nhận dạng danh tính đời thực.

Ba câu hỏi nghiên cứu: kiến trúc detector nào phù hợp ảnh drone nhiều vật thể nhỏ; thứ hạng tracker có giữ khi thay detector không; phần phát hiện và phần liên kết ảnh hưởng kết quả đến mức nào.

Đây là so sánh cấu hình triển khai hoàn chỉnh, không phải thí nghiệm cô lập kiến trúc, vì ba detector có độ phân giải, phép tăng cường và cách tối ưu khác nhau.

Trong nghiên cứu chính, nhãn detector gồm 10 lớp. Năm lớp chấm MOT là pedestrian, car, van, truck, bus. Motor có trong huấn luyện detection nhưng không nằm trong phép tính tracking chuẩn của báo cáo. Không được nói kết quả HOTA này chứng minh hệ thống theo vết xe máy tốt.

Chức năng nhận biển số, phát hiện vượt đèn đỏ và giao diện desktop là nhánh ứng dụng, phải phân biệt với kết quả benchmark VisDrone.

## 3. Bộ dữ liệu và cách chia dữ liệu

VisDrone2019-MOT cung cấp chuỗi frame với box và track ID. VisDrone2019-DET cung cấp ảnh tĩnh với box, không cung cấp quỹ đạo. Dữ liệu drone có nhiều bối cảnh đô thị, độ cao, góc nhìn và ánh sáng khác nhau.

Dữ liệu MOT trước xử lý có 80 chuỗi, 33.682 frame, 1.527.447 box ở 10 lớp và 10.393 quỹ đạo. MOT train gồm 56 chuỗi, 24.201 frame, 1.105.516 box, 7.342 track; val gồm 7 chuỗi, 2.846 frame, 114.132 box, 758 track; test-dev gồm 17 chuỗi, 6.635 frame, 307.799 box, 2.293 track.

Trung bình tương ứng 45,7; 40,1; 46,4 box mỗi frame. Có 89.835 vùng ignored trên cả ba split.

Nhãn MOT mỗi dòng có frame, track_id, x, y, w, h, score, class, truncation, occlusion. Hộp dùng góc trái trên và chiều rộng, chiều cao theo pixel ảnh gốc. Score trong ground truth là cờ hợp lệ khi đánh giá, không phải xác suất của detector.

Truncation cho biết đối tượng bị cắt ở biên. Occlusion 0 là không che, 1 là che một phần, 2 là che nhiều. ID được bảo toàn theo chuỗi, không nhất thiết có ý nghĩa xuyên chuỗi.

Mã lớp 0 là ignored region, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others. Detector học 1–10; loại 0 và 11.

YOLO và RF-DETR dùng chỉ số nội bộ 0–9, khi ghi cache phải cộng một; Faster R-CNN dùng 0 cho background và 1–10 cho lớp thật.

Thí nghiệm chính huấn luyện detector trên DET train. Có 57 ảnh DET trùng mã video nguồn với MOT val/test-dev, trong đó 45 trùng val và 12 trùng test-dev. Những ảnh này bị loại, còn 6.414 ảnh train. DET val có 548 ảnh để chọn checkpoint.

MOT val dùng để dò ngưỡng tracker; MOT test-dev dành cho kết quả cuối. DET test-dev có 1.610 ảnh là tập ảnh tĩnh riêng, chỉ dùng thí nghiệm detection bổ trợ. Tuyệt đối không trộn mAP trên DET test-dev với mAP trên MOT test-dev vì ảnh và giao thức khác nhau.

Không chia ngẫu nhiên từng frame MOT giữa train và val vì frame sát nhau gần như trùng, tạo rò rỉ cảnh.

EDA ban đầu khảo sát một biến thể huấn luyện bằng frame MOT. Biến thể này lấy frame lẻ, stride 2, cho 12.114 frame train và 553.391 box, giữ 99,6% quỹ đạo. Val/test-dev vẫn giữ mọi frame để tracking. Đây không phải tập huấn luyện của ma trận kết quả cuối.

Thí nghiệm chẩn đoán RF-DETR trên nguồn MOT cho thấy mAP train tăng 39,12 lên 58,02 từ epoch 1 đến 16 trong khi mAP val giảm 23,32 xuống 17,79. Các frame MOT dư thừa theo thời gian và chỉ thuộc 56 chuỗi, nên nhóm chọn DET để tăng đa dạng cảnh.

Những kết quả chẩn đoán trên tập con không thay thế các bảng chính.

## 4. Phân phối dữ liệu và ý nghĩa đối với mô hình

Train MOT mất cân bằng lớp mạnh. Car có 505.301 box, chiếm 45,7%; pedestrian 234.305, 21,2%; motor 102.819, 9,3%; people 94.396, 8,5%; van 46.940, 4,2%; bicycle 40.255, 3,6%; truck 30.498, 2,8%; tricycle 28.338, 2,6%; awning-tricycle 13.011, 1,2%; bus 9.653, 0,9%. Lớp car nhiều hơn bus khoảng 52 lần.

Trên val, pedestrian 28,4%, car 27,9%, bus chỉ 264 box (0,2%). Trên test-dev, pedestrian 34,5%, car 32,4%; như vậy class prior thay đổi giữa các split. Điểm AP/HOTA của lớp hiếm có thể dao động mạnh; mAP trung bình theo lớp trao trọng số ngang nhau cho bus dù số box thấp.

Theo chuẩn COCO, small có diện tích nhỏ hơn 32² pixel, medium từ 32² đến dưới 96², large từ 96² trở lên. Tỉ lệ small ở MOT train là 24,9%, val 19,2%, test-dev 47,8%; test-dev nhỏ hơn rõ rệt.

Cạnh tương đương căn diện tích có trung vị 45,8 pixel ở train, 52,4 ở val, 33,4 ở test-dev. Khi đưa ảnh về cạnh dài 576, 27,9% box train và 24,3% box test-dev có cạnh ngắn dưới 8 pixel; về 1024, còn 4,3% và 4,7%.

Đây là thống kê mô phỏng resize, không phải khẳng định một mô hình tại độ phân giải đó có recall tương ứng. Chi tiết vài pixel làm định vị chính xác khó, IoU rất nhạy với lệch hộp.

Ví dụ hộp 10×10 dịch 2 pixel theo cả hai trục có IoU 64/136, khoảng 0,47; một lỗi nhỏ đã khiến ghép ở IoU 0,5 thất bại.

Mật độ cao: frame tối đa 180 box train, 116 val, 152 test-dev; khoảng 6–8% frame có trên 100 box. Mật độ này làm tăng box chồng lấn, che khuất và nguy cơ NMS xóa đúng đối tượng.

Số query RF-DETR 300 về mặt sức chứa lớn hơn số box ground truth tối đa, nhưng không bảo đảm tất cả query tìm đúng vật thể. Với NMS của YOLO và Faster R-CNN phải xem xét max_det, ngưỡng score và IoU.

Detector YOLO đặt max_det=1.000 khi cache để tránh cắt mất box ở cảnh đông.

Tỉ lệ không che/che một phần/che nhiều trên train lần lượt 54,3%/40,3%/5,3%; val 40,9%/52,4%/6,6%; test-dev 47,9%/42,6%/9,5%. Che khuất làm detection rớt score, gây quỹ đạo đứt hoặc đổi ID. Khoảng 7–11% quỹ đạo tùy split có đoạn mất quan sát rồi xuất hiện lại.

Kích thước và vị trí cũng liên hệ với phối cảnh: phần trên ảnh thường xa và vật thể nhỏ, phần dưới gần và vật thể lớn. Vì vậy lật dọc hoặc quay 90° không hẳn là augmentation hợp lý cho ảnh drone nhìn xiên.

IoU của cùng đối tượng ở hai frame sát nhau có trung vị khoảng 0,897 trên train, 0,873 val và 0,875 test-dev; phần có IoU dưới 0,3 lần lượt chỉ 1,6%, 4,2%, 3,2%.

Điều này lý giải vì sao ghép theo chuyển động và IoU vẫn có hiệu quả lớn nếu detector cung cấp box đều đặn. Val có chuyển động mạnh hơn train; camera di động vẫn có thể làm dự đoán chuyển động sai, nhất là ở đoạn quay nhanh.

Sai khác ảnh xám trung bình hai frame kề nhau là 4,67/255 trong mẫu EDA, là căn cứ cho stride 2 ở biến thể MOT.

Vùng ignored là nơi nhãn chưa đầy đủ, không phải vùng nền âm. MOT train có 62.369 vùng, val 3.994, test-dev 23.472; ở test-dev, 85,8% frame có ít nhất một vùng. Quy trình không tô đen vùng này trên ảnh.

Khi chấm, box có hơn 50% diện tích nằm trong hợp vùng ignored bị loại cho cả ground truth và dự đoán theo quy ước dự án. Nếu coi detection trong vùng chưa gán nhãn là false positive, độ chính xác sẽ thấp sai.

Có 34 box nhãn vượt mép ảnh 1–2 pixel được clip; không có box đối tượng cạnh bằng 0 hoặc âm.

## 5. Tiền xử lý và giao thức để so sánh công bằng

Pipeline offline đọc nhãn một lần, lấy kích thước ảnh theo chuỗi, clip box về biên ảnh, bỏ box sau clip có cạnh dưới 1 pixel, loại lớp 11, giữ vùng ignored riêng.

Nó xuất COCO cho Faster R-CNN và đánh giá, RF-DETR theo COCO với tên ảnh phẳng và liên kết tượng trưng, YOLO với tâm và kích thước chuẩn hóa cùng class 0–9, và MOTChallenge cho TrackEval. Bản MOTChallenge giữ toàn bộ frame, kể cả các frame không có detection.

Khi train DET, không dùng stride 2; khi train MOT thử nghiệm, chỉ train lấy frame lẻ. COCO giữ class 1–10; RF-DETR và YOLO ánh xạ lại đúng khi suy luận. Các định dạng được kiểm tra số ảnh, số box, class, tọa độ, liên kết ảnh và xem bằng mắt.

Pipeline online thuộc từng thư viện. Faster R-CNN resize giữ tỉ lệ, cạnh ngắn 800 và cạnh dài tối đa 1.333, chuẩn hóa kiểu ImageNet. YOLO11m letterbox về 640×640, chuẩn hóa giá trị pixel, dùng augmentation Ultralytics gồm mosaic, HSV, scale, translate, lật ngang; đóng mosaic 10 epoch cuối.

RF-DETR-M suy luận ở 576; khi train có đa tỉ lệ ảnh vuông khoảng 416–736 pixel và lật ngang 0,5. Không nên nói cả ba cùng độ phân giải. Khác biệt input là một phần của cấu hình và có thể ảnh hưởng đáng kể đến AP small.

Để tái dùng cùng dự đoán cho năm tracker, mỗi detector suy luận một lần ở score tối thiểu 0,05 và lưu file theo chuỗi, mỗi dòng frame, -1, x, y, w, h, score, class trên ảnh gốc. Ngưỡng 0,05 giữ các box score thấp để ByteTrack/BoT-SORT dùng tầng ghép thứ hai.

Mọi tracker của cùng detector đọc cùng cache nên không có sai khác do suy luận lại. Trước tracking, chỉ lấy năm lớp đánh giá, ghép cặp từng lớp, không nối ID giữa hai lớp khác nhau.

Bộ nhớ tracker được reset khi bắt đầu video mới; frame trống vẫn phải được đưa qua tracker để tăng tuổi track. Sau tracking, lọc vùng ignored rồi chấm. Khóa checkpoint bằng DET val, ngưỡng bằng MOT val; không dùng MOT test-dev để chọn ngưỡng.

## 6. Nền tảng lý thuyết của detector

Trong detection, precision = TP/(TP+FP), recall = TP/(TP+FN), còn IoU(A,B) = diện tích giao / diện tích hợp. Một detection được ghép với một ground truth cùng lớp ở ngưỡng IoU quy định.

AP là diện tích dưới đường precision–recall sau quy ước nội suy của COCO. mAP50-95 là trung bình AP trên 10 ngưỡng IoU 0,50 đến 0,95 bước 0,05, rồi trung bình trên lớp. mAP50 dễ hơn mAP50-95; AP small, medium, large lọc ground truth theo diện tích ảnh gốc.

Điểm score không phải xác suất đã hiệu chỉnh giống nhau giữa các kiến trúc.

Backbone trích đặc trưng hình ảnh; neck kết hợp đặc trưng các độ phân giải; head chuyển đặc trưng thành lớp, box và score. CNN dùng phép tích chập cục bộ, downsampling mở rộng receptive field nhưng có thể mất tín hiệu vật thể nhỏ.

FPN kết hợp đặc trưng ngữ nghĩa mức cao với chi tiết mức thấp. Transformer dùng cơ chế attention để kết hợp thông tin theo vị trí và nội dung.

Một giai đoạn dự đoán trực tiếp trên bản đồ đặc trưng; hai giai đoạn trước tiên sinh đề xuất vùng rồi phân loại/tinh chỉnh; DETR xem detection như dự đoán một tập đối tượng và học ghép một-một.

## 7. Faster R-CNN ResNet-50 FPN v2

Kiến trúc hai giai đoạn của torchvision gồm ResNet-50 làm backbone, FPN kết hợp đặc trưng nhiều tầng, Region Proposal Network (RPN), RoIAlign và RoI head. ResNet-50 dùng các khối residual để học biểu diễn sâu mà gradient dễ truyền hơn.

FPN tạo các bản đồ ở các mức kích thước, đưa vật thể nhỏ tới mức phù hợp. RPN trượt trên feature map, dự đoán objectness và biến đổi box tương đối với anchor.

Các proposal được lọc, chọn RoI phù hợp ở FPN và RoIAlign lấy đặc trưng kích thước cố định bằng nội suy, tránh sai lệch do lượng tử hóa cứng của RoIPool. RoI head dự đoán lớp và hồi quy box lần hai. Khi suy luận, NMS loại nhiều proposal trùng.

Đầu phân loại có 11 đầu ra: nền và 10 lớp VisDrone.

Loss huấn luyện gồm bốn phần: RPN classification/objectness, RPN box regression, RoI classification và RoI box regression. RPN trả lời vùng nào có vật, còn RoI head trả lời vật thuộc lớp nào và box chính xác ra sao.

Ưu điểm là độ phân giải 800–1.333 và FPN giữ chi tiết cho vật thể nhỏ; nhược điểm là hai bước và ảnh lớn gây nhiều phép tính.

Mô hình được khởi tạo từ COCO, train 24 epoch theo cấu hình chạy: batch 4, SGD lr 0,005, momentum 0,9, weight decay 10^-4, AMP, seed 0, giảm lr sau epoch 16 và 22. Checkpoint DET val tốt nhất epoch 17.

Báo cáo có một chú thích hình nhắc 25 epoch đường cong; nên trả lời theo run_config 24 epoch và checkpoint 17, đồng thời nói rõ nếu bị hỏi về khác biệt ghi chép.

Theo bảng thống kê mô hình, Faster R-CNN có 43,3 triệu tham số, khoảng 745,5 GFLOPs/frame với quy ước 2 FLOPs cho phép nhân-cộng trên phân phối độ phân giải benchmark, tốc độ detector 13,8 FPS trên L4. Đây không phải GFLOPs ở cùng input với hai mô hình còn lại.

Trên MOT test-dev mAP50-95 là 22,45, mAP50 48,58, AP small 8,02, AP medium 28,32, AP large 39,72. Trên MOT val mAP50-95 24,15, cao nhất trong ba detector tại split đó.

## 8. YOLO11m

YOLO11m là detector một giai đoạn: backbone lấy đặc trưng, neck kết hợp các mức và head trực tiếp dự đoán box, lớp tại nhiều stride, thường 8, 16, 32.

Bản dùng trong dự án có khối C3k2 và C2PSA; C3k2 là biến thể cross stage partial để trao đổi đặc trưng với chi phí vừa phải, C2PSA thêm attention vùng/đặc trưng.

Đầu detection hiện đại của Ultralytics là anchor-free theo nghĩa không cần bộ hộp anchor với kích thước cố định như RPN; điểm lưới là mốc để dự đoán khoảng cách đến bốn cạnh. Loss gồm thành phần hộp dựa trên IoU, classification và Distribution Focal Loss để học phân phối khoảng cách biên.

Gán positive mẫu xét đồng thời tín hiệu lớp và chất lượng vị trí. Sau suy luận, NMS loại các hộp trùng; cache dùng NMS IoU 0,7, max_det 1.000, score tối thiểu 0,05. Input 640 với letterbox giữ tỉ lệ nhưng làm đối tượng nhỏ hơn trên ảnh gốc.

Mô hình khởi tạo từ yolo11m.pt COCO, train trên DET train 6.414 ảnh, 80 epoch, batch 16, SGD lr đầu 0,01, lịch cosine, warmup 3 epoch, weight decay 5×10^-4, AMP và seed 0. Mosaic đóng 10 epoch cuối; patience 20. Checkpoint tốt nhất epoch 60 theo DET val.

Môi trường train ghi Tesla T4 và Ultralytics 8.4.197. Có 20,1 triệu tham số, 40,7 GFLOPs/frame và detector 118,3 FPS trên L4. MOT test-dev mAP50-95 23,17, mAP50 46,54, AP small 8,10, AP medium 28,53, AP large 42,53; mAP năm lớp tracking 35,71.

Trên DET test-dev tĩnh, thí nghiệm phụ sau gộp sáu lớp chung COCO cho mAP 8,61 trước fine-tune và 26,05 sau fine-tune; không so trực tiếp hai số này với mAP MOT 10 lớp.

## 9. RF-DETR Medium

RF-DETR-M đi theo nhánh DETR, Deformable DETR và LW-DETR. Backbone là vision transformer tiền huấn luyện DINOv2; ảnh được chia patch và đưa vào encoder để học đặc trưng.

Với độ phân giải 576 và patch 16, một box có cạnh chỉ vài pixel có thể chứa rất ít tín hiệu; đây là lý do nghi ngờ khó khăn ở vật thể nhỏ, nhưng không được khẳng định mạng không thể nhận vật nhỏ hơn patch.

Đặc trưng backbone được chiếu sang không gian decoder. Các object query học được đại diện các vị trí ứng viên trong tập dự đoán; decoder dùng điểm tham chiếu không gian và deformable cross-attention để lấy thông tin từ một số điểm phù hợp ở nhiều mức thay vì attention dày toàn ảnh.

Query là biểu diễn cho một dự đoán trong frame, không phải ID qua thời gian. Output giới hạn khoảng 300 detection theo cấu hình; không dùng NMS chuẩn vì huấn luyện Hungarian matching một-một giữa dự đoán và ground truth.

Ghép Hungarian trong huấn luyện chọn mỗi ground truth cho nhiều nhất một query, các query khác là no-object. Loss gồm phân loại, L1 box, GIoU box và giám sát decoder trung gian.

L1 giúp tọa độ gần ground truth, GIoU xử lý chất lượng chồng lấn và cả hộp không giao nhau; phân loại xử lý lớp/no-object. Điểm khác Faster R-CNN là không sinh proposal/RoI bằng RPN, còn khác YOLO là đầu dự đoán tập với ghép một-một.

Paper RF-DETR nêu tìm kiếm kiến trúc NAS có chia sẻ trọng số để cân bằng độ trễ và độ chính xác; nhóm chỉ tinh chỉnh cấu hình Medium có sẵn, không chạy lại NAS.

Cấu hình ma trận chính là rfdetr_m_576_det: DET train 6.414 ảnh, DET val 548 ảnh, 40 epoch tối đa, early stopping patience 8, batch 8 mỗi GPU × 2 L4 = effective batch 16, không tích lũy gradient.

AdamW lr 10^-4, lr encoder 1,5×10^-4, weight decay 10^-4, cosine xuống min factor 0,01, warmup tuyến tính 1 epoch, seed 0.

Hình báo cáo đánh dấu checkpoint tốt nhất epoch 23, nhưng phần văn bản bảng hyperparameter chưa xác nhận epoch; khi trả lời nên nói hình ghi epoch 23 và kiểm tra log gốc nếu thầy yêu cầu truy xuất chính xác.

RF-DETR có 33,4 triệu tham số, 78,4 GFLOPs/frame, detector 32,8 FPS trên L4. MOT test-dev mAP50-95 22,96, mAP50 48,25, AP small 8,35, AP medium 28,48, AP large 47,65; AP large và AR@500 cao nhất trong ba cấu hình.

RF-DETR không tự làm MOT dù có object query: query không được ràng buộc mang cùng định danh ở frame sau. Vẫn cần một tracker.

Nếu bị hỏi vì sao mAP tổng gần YOLO mà HOTA thấp hơn, phải nêu mAP đo ranking qua nhiều ngưỡng và 10 lớp, còn tracking đánh giá năm lớp ở một điểm score cụ thể với lỗi liên kết tích lũy; khác phân bố score, recall, chất lượng box theo lớp và độ ổn định thời gian đều có thể ảnh hưởng.

Chưa có phân tích nhân quả tách riêng từng yếu tố.

## 10. Nền tảng toán học của tracker

Kalman filter là bộ ước lượng trạng thái tuyến tính có nhiễu. Trạng thái x và hiệp phương sai P được dự đoán theo x̂(t|t−1)=F x(t−1), P(t|t−1)=F P(t−1) Fᵀ+Q. Khi có detection z, innovation y=z−H x̂, innovation covariance S=H P Hᵀ+R, Kalman gain K=P Hᵀ S^-1, rồi cập nhật x=x̂+Ky và P=(I−KH)P.

F mô tả chuyển động gần vận tốc hằng; Q là nhiễu mô hình, R là nhiễu đo. Khi thiếu detection, tracker chỉ dự đoán, độ bất định tăng. Camera di động làm giả định vận tốc ảnh hằng kém đúng nếu không bù chuyển động.

IoU giữa box track dự đoán và detection tạo chi phí c=1−IoU. Ghép Hungarian tìm phân công một-một có tổng chi phí thấp nhất dưới các ràng buộc/gating. Việc ghép đúng ID khó hơn chỉ phát hiện đúng tại một frame, vì một sai lầm có thể kéo dài.

Quản lý vòng đời gồm tentative track mới, xác nhận sau đủ hits, lost khi chưa có detection, xóa khi quá max_age hoặc track_buffer. Ngưỡng score, IoU, số frame cho phép mất và điều kiện tạo ID quyết định đánh đổi FP với FN/IDSW.

ReID trích embedding ngoại hình từ crop đối tượng; cosine distance = 1−(a·b)/(||a||||b||) đo độ khác. Mahalanobis distance của vị trí dự đoán tính cả hiệp phương sai, giúp gating cặp quá xa. Cần phân biệt ReID học nhận diện ngoại hình với OCR đọc biển số: đây là hai mô-đun khác nhau.

Trong dự án DeepSORT và BoT-SORT dùng OSNet-x0.25 tiền huấn luyện MSMT17. Nguồn này chủ yếu người đi bộ góc nhìn mặt đất, còn VisDrone có cả phương tiện crop rất nhỏ, gây lệch miền.

## 11. SORT

SORT dùng Kalman vận tốc hằng, IoU và Hungarian, không ReID, không GMC, không ghép box score thấp thành hai tầng. Trạng thái thường là tâm u,v, diện tích s, tỷ lệ r và vận tốc của u,v,s: [u,v,s,u̇,v̇,ṡ], với r giả định gần hằng.

Nó dự đoán box từng track, ghép với detection bằng IoU, cập nhật khi khớp, tăng tuổi khi mất, khởi tạo track mới từ detection chưa ghép. Cấu hình hiện có: det_thresh 0,4 mặc định, max_age 30, min_hits 3, ngưỡng IoU 0,3. Ưu điểm nhanh, đơn giản, làm baseline rõ.

Nhược điểm dễ tráo ID khi box giao nhau, che khuất hoặc camera dịch chuyển làm IoU giảm. Trên test-dev với ngưỡng đã chọn, HOTA là 45,32 YOLO, 45,10 Faster R-CNN, 43,11 RF-DETR.

## 12. DeepSORT

DeepSORT mở SORT bằng đặc trưng ngoại hình. Nó dùng dự đoán Kalman và gating Mahalanobis để loại ghép không phù hợp chuyển động, rồi cosine distance của embedding để chọn ID; matching cascade ưu tiên track vừa được quan sát gần đây, còn các cặp chưa khớp có thể ghép bằng IoU.

Các track giữ một kho embedding giới hạn để nhận lại sau che khuất. Cấu hình: det_thresh 0,4, max_age 30, n_init 3, max_iou_distance 0,7, max_cosine_distance 0,2, nn_budget 100, OSNet-x0.25 MSMT17.

DeepSORT có thêm chi phí cắt crop và chạy ReID; với ảnh drone từ trên cao, crop quá nhỏ khiến embedding có thể không ổn định. HOTA test-dev 46,38 YOLO, 45,51 Faster R-CNN, 42,87 RF-DETR.

Nó tốt hơn SORT ở hai detector đầu nhưng kém 0,24 điểm ở RF-DETR; vì vậy không nói ReID luôn có lợi.

## 13. ByteTrack

ByteTrack dựa trên quan sát quan trọng: box của vật thật khi che khuất thường có score thấp, nếu lọc sớm sẽ đứt ID. Nó tách detection thành nhóm high score và low score.

Tầng một ghép track đang có với nhóm high; tầng hai ghép track chưa khớp với nhóm low bằng hình học. Low score chỉ cứu track cũ, không tạo track mới, nên giảm quỹ đạo giả. Detection còn lại đủ score và thỏa new_track_thresh mới được khởi tạo.

Cấu hình: high 0,5, low 0,1, new 0,6, match 0,8, track_buffer 30. Trong quét ngưỡng, high thay bằng tau; new track tăng cùng tau theo quy tắc trong script, low giữ 0,1. Điều này có nghĩa quét tau thay đổi cả điều kiện khởi tạo track, không đơn thuần lọc file đầu vào.

Nếu cache từ 0,5 sẽ làm mất toàn bộ giá trị của tầng low.

ByteTrack không cần mạng ReID nên nhanh. HOTA test-dev 47,09 YOLO, 46,64 Faster R-CNN, 45,97 RF-DETR. YOLO + ByteTrack đạt khoảng 101,0 FPS xử lý ghép detector/tracker trên L4. Dù BoT-SORT cao HOTA hơn, ByteTrack là lựa chọn phù hợp khi cần ngân sách thời gian cho OCR, đọc video và UI.

Score thấp cũng có FP; nếu detector tạo nhiều box nhiễu thì tầng hai có thể ghép sai, vì vậy hiệu quả phụ thuộc đặc tính đầu vào.

## 14. OC-SORT

OC-SORT lấy observation làm trung tâm để giảm lệch do dự đoán Kalman qua các frame mất box.

OCM đưa hướng chuyển động tính từ các quan sát vào chi phí ghép; OCR dùng quan sát gần nhất để thử phục hồi cặp track; ORU dùng các quan sát ảo để cập nhật lại bộ lọc khi vật xuất hiện sau đoạn mất, nhằm giảm sai lệch trạng thái.

Các bước này đặc biệt có ý nghĩa khi chuyển động phi tuyến hoặc thiếu detection. Cấu hình: det_thresh 0,5, max_age 30, min_hits 3, IoU 0,3, delta_t 3, inertia 0,2, association function IoU. HOTA test-dev 46,51 YOLO, 46,22 Faster R-CNN, 43,78 RF-DETR.

Ở oracle, OC-SORT đạt 89,99, cao hơn ByteTrack 86,48, nhưng với detector thật ByteTrack lại hơn OC-SORT trong cả ba tổ hợp. Đây là ví dụ rõ việc thứ hạng phụ thuộc chất lượng detection; không kết luận cơ chế OC-SORT kém về nguyên lý.

## 15. BoT-SORT

BoT-SORT kết hợp ý tưởng ghép hai tầng score của ByteTrack với Kalman theo tâm, chiều rộng, chiều cao và vận tốc tương ứng, ReID và bù chuyển động camera GMC.

Trạng thái tám chiều mô tả [cx,cy,w,h,vcx,vcy,vw,vh]; cách này tránh một số vấn đề của biểu diễn diện tích/tỷ lệ khi hình dạng box đổi.

GMC trong dự án dùng sparse optical flow: tìm điểm đặc trưng, theo chúng qua frame, ước lượng biến đổi chuyển động toàn cảnh rồi hiệu chỉnh vị trí/độ bất định track trước khi ghép.

Cơ chế này xử lý chuyển động ảnh do camera, không tự ổn định video tuyệt đối và không tự giải quyết một vạch dừng đứng yên trong thế giới thực. ReID thêm tín hiệu ngoại hình khi hình học mơ hồ.

BoT-SORT kết hợp khoảng cách IoU có score và khoảng cách cosine, với gating proximity và appearance; nếu cặp xa về hình học thì ngoại hình không được tùy tiện nối.

Cấu hình: high 0,5, low 0,1, new 0,6, match 0,8, buffer 30, proximity 0,5, appearance 0,25, with_reid true, cmc_method sof; ReID là OSNet-x0.25 MSMT17. Trong ma trận cuối, ngưỡng high tối ưu trên val là 0,6 cho YOLO, 0,8 cho Faster R-CNN và 0,5 cho RF-DETR. HOTA test-dev 51,67, 51,68 và 49,85.

So ByteTrack cùng detector, BoT-SORT tăng 3,88–5,04 HOTA; AssA tăng 7,12–9,67 điểm, song chi phí ReID/GMC làm FPS giảm mạnh. Không được nói riêng GMC làm tăng chính xác bao nhiêu: chưa có ablation tắt riêng GMC và ReID. Hai tracker còn khác mô hình trạng thái và chi phí ghép.

## 16. Chọn ngưỡng và cách đọc ma trận kết quả

Ma trận chính lấy ngưỡng cao tau từ lưới 0,3 đến 0,8, bước 0,1, chọn riêng cho từng cặp detector–tracker bằng HOTA trên MOT val. Nếu có nhiều cấu hình bằng nhau, phải theo quy tắc script chọn ngưỡng, không thay bằng test-dev.

Các ngưỡng cuối: YOLO/SORT 0,5; DeepSORT 0,4; ByteTrack 0,6; OC-SORT 0,5; BoT-SORT 0,6. Faster R-CNN tương ứng 0,6; 0,8; 0,8; 0,6; 0,8. RF-DETR tương ứng 0,4; 0,4; 0,6; 0,4; 0,5.

Khảo sát mở rộng Faster R-CNN + BoT-SORT đến 0,9 đạt HOTA val 49,83, chứng tỏ 0,8 chỉ tốt nhất trong lưới chính, không phải tối ưu toàn cục; bảng test-dev chính vẫn giữ lưới đã khóa.

Score giữa mô hình không cùng thang và không nhất thiết đã hiệu chỉnh, nên không dùng một ngưỡng tuyệt đối để kết luận công bằng.

Kết quả test-dev theo thứ tự SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT. Với YOLO11m, HOTA lần lượt 45,32; 46,38; 47,09; 46,51; 51,67. IDF1 là 55,76; 57,48; 58,38; 57,34; 64,90. MOTA là 45,08; 42,75; 46,12; 46,40; 48,15. FPS end-to-end tính theo các khối xử lý là 101,1; 25,6; 101,0; 92,3; 25,9.

BoT-SORT chỉ có 498 IDSW so với ByteTrack 1.521.

Với Faster R-CNN, HOTA theo cùng thứ tự là 45,10; 45,51; 46,64; 46,22; 51,68. IDF1 là 55,00; 55,97; 57,44; 56,56; 64,76. MOTA 34,30; 40,00; 35,63; 35,79; 37,38. FPS 13,4; 9,8; 13,3; 13,1; 9,2. BoT-SORT có 821 IDSW.

MOTA của DeepSORT cao hơn BoT-SORT tại detector này dù HOTA/IDF1 thấp hơn; do các chỉ số đánh giá lỗi khác nhau, không có xếp hạng tuyệt đối duy nhất.

Với RF-DETR-M, HOTA là 43,11; 42,87; 45,97; 43,78; 49,85. IDF1 là 51,03; 51,25; 56,59; 52,01; 61,10. MOTA 36,06; 34,66; 45,21; 37,57; 40,45. FPS 31,3; 17,9; 31,5; 30,4; 16,2. ByteTrack dẫn MOTA, BoT-SORT dẫn HOTA và IDF1.

Nếu được hỏi detector nào thắng, trả lời phụ thuộc metric: trên MOT val mAP Faster R-CNN cao nhất; trên MOT test-dev YOLO mAP 10 lớp và năm lớp cao nhất; trong HOTA với BoT-SORT hai mô hình Faster R-CNN và YOLO gần như hòa; nếu xét tốc độ thì YOLO vượt trội.

HOTA theo lớp với BoT-SORT: pedestrian 36,74 YOLO, 42,17 Faster R-CNN, 37,70 RF-DETR; car 61,04, 61,56, 59,72; van 44,55, 46,51, 44,18; truck 51,93, 51,77, 50,60; bus 64,08, 56,39, 57,05. Bus có ít mẫu nên điểm cao chưa chứng minh khả năng tổng quát tốt. Pedestrian thấp vì nhỏ và nhiều che khuất.

Trong báo cáo có một chú thích hình ghi “motor” khi mô tả năm lớp tracking; dựa trên dữ liệu CSV và giao thức, lớp đúng là bus.

## 17. Ý nghĩa các chỉ số tracking và lỗi thường gặp khi giải thích

HOTA ở một ngưỡng định vị alpha có dạng căn bậc hai của DetA nhân AssA; HOTA cuối là trung bình trên 19 mức alpha từ 0,05 tới 0,95. DetA = TP/(TP+FP+FN) dưới quy tắc matching của HOTA, khác precision thông thường có mẫu số TP+FP.

AssA đo mức nhất quán liên kết của các cặp đúng qua thời gian, về bản chất xem TP liên kết so với FP/FN liên kết. DetA không hoàn toàn chỉ do detector vì tracker có thể làm mất hoặc tạo track; AssA cũng bị ảnh hưởng khi detector bỏ box.

Không lấy căn tích DetA và AssA đã trung bình ở bảng để kiểm tra HOTA cuối, vì thứ tự trung bình và lấy căn khác nhau.

MOTA = 1−(FN+FP+IDSW)/GT theo tổng frame; nhạy rất mạnh với FN và FP, có thể âm nếu lỗi vượt tổng ground truth. MOTA cao chưa bảo đảm ID ổn định. IDF1 = 2 IDTP/(2 IDTP+IDFP+IDFN), đo chất lượng ghép danh tính toàn chuỗi.

IDSW là số lần một đối tượng ground truth bị gán sang ID khác; fragmentation là số đoạn đứt liên kết. MT là quỹ đạo được theo dõi ít nhất 80% chiều dài, ML là không quá 20%; báo cáo dùng trung bình phần trăm qua năm lớp.

MOTP/LocA phản ánh vị trí của những cặp đã ghép, không thay AP. Khi đánh giá TrackEval, tổng hợp các chuỗi trong từng lớp rồi trung bình cộng chỉ số phần trăm qua năm lớp; các chỉ số đếm như FP, FN, IDSW cộng lại.

Không lấy trung bình từng frame HOTA rồi gọi là HOTA dataset.

Oracle thay detection bằng ground truth box, score bằng 1, nhưng không đưa ground truth ID cho tracker. Tracker vẫn phải khởi tạo, ghép và quản lý track. HOTA oracle theo SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT là 85,28; 84,30; 86,48; 89,99; 94,93.

BoT-SORT oracle DetA 97,52, AssA 92,47, IDF1 95,53, MOTA 99,40; chưa đạt 100 vì tracker vẫn có lỗi và quy tắc xác nhận/xuất track. Khoảng cách với detector thật tốt nhất theo từng tracker là 39,96; 37,92; 39,39; 43,48; 43,25 điểm HOTA.

Đây là mốc tham chiếu cho tiềm năng, không phải phân rã nhân quả độc lập lỗi detection và association.

## 18. Tốc độ, chi phí và đánh đổi triển khai

Benchmark tốc độ dùng một NVIDIA L4, batch 1, FP32, cùng 905 frame từ ba chuỗi MOT test-dev mật độ khác nhau. Detector warmup 50 frame, đồng bộ CUDA trước/sau đo, ba lượt lấy trung vị. FPS detector: Faster R-CNN 13,76, RF-DETR 32,82, YOLO 118,27.

FPS tracker chỉ đo ghép cặp có thể rất cao nhưng không đồng nghĩa toàn hệ thống nhanh. Ví dụ YOLO + BoT-SORT phần liên kết 378,0 FPS, nhưng khi tính ReID/GMC phần tracker đầy đủ chỉ 33,1 FPS, ghép với detector được 25,9 FPS.

YOLO + ByteTrack phần tracker đầy đủ khoảng 692,4 FPS, hệ kết hợp khoảng 101,0 FPS. Công thức tổng thời gian trung bình là t_e2e=t_det+t_front+t_assoc, FPS=N/t_tổng, không được cộng FPS các khối.

Các con số FPS ở bảng không tính đọc/giải mã video, ghi file, hiển thị, OCR, plate detector, tương tác UI và độ trễ hàng đợi. Không suy diễn 101 FPS trên L4 thành 101 FPS trên CPU hay Jetson.

RF-DETR 576 có GFLOPs cao hơn YOLO 640 nhưng vẫn ít hơn Faster R-CNN ở ảnh 800–1.333; chi phí thực tế còn phụ thuộc kernel, bộ nhớ, hậu xử lý và phần cứng.

Với mục tiêu giám sát có OCR, lựa chọn detector/tracker là một đánh đổi tổng ngân sách xử lý; nên đo toàn ứng dụng trên đúng camera và máy.

## 19. Các nguồn lỗi và cách bảo vệ kết luận trước phản biện

AP small test-dev của ba detector chỉ khoảng 8,02–8,35, trong khi AP large khoảng 39,72–47,65. AR small tương ứng YOLO 19,83, Faster R-CNN 22,75, RF-DETR 23,00; AR large 56,90, 57,84, 65,55. Vì vậy vấn đề lớn là bỏ sót và định vị đối tượng nhỏ, đặc biệt khi test-dev có 47,8% box small.

Mất box làm tracker thiếu quan sát, dẫn đến FN, fragmentation và có thể đổi ID. Dương tính giả có thể đến từ bóng cây, kết cấu mặt đường; lỗi liên kết thường xuất hiện khi hai đối tượng giao cắt hoặc tạm che nhau.

RF-DETR có nhiều box trong cache ở score 0,05 hơn YOLO trên cùng MOT test-dev (1.383.694 so 710.254 sau lọc ignored), nhưng không được đồng nhất nhiều box với nhiều false positive khi chưa ghép ground truth. Ở score 0,5, precision/recall của YOLO là 73,83/54,19; RF-DETR 64,98/52,69 theo báo cáo.

Không kết luận kiến trúc nào “thắng tuyệt đối” từ chênh 0,01 HOTA vì chưa có khoảng tin cậy theo chuỗi hoặc nhiều seed. Chỉ một seed huấn luyện cho mỗi detector; val có bảy chuỗi, bus rất hiếm; lớp và kích thước lệch giữa split.

Không khẳng định BoT-SORT tăng nhờ riêng GMC hoặc ReID khi chưa chạy tắt từng thành phần. Không khẳng định 576 là nguyên nhân duy nhất RF-DETR kém ở HOTA vì kiến trúc, optimizer, score và checkpoint cũng khác.

Muốn thử giả thuyết phải giữ cùng detector/cache và chỉ đổi một yếu tố; ví dụ chạy RF-DETR 1024 và so AP small cùng FPS, hoặc tắt GMC rồi tắt ReID trong BoT-SORT ở cùng ngưỡng.

Có thể thử tiling/SAHI để giữ vật nhỏ nhưng cần kiểm tra hộp trùng, mép lát và chi phí. Mọi cải tiến chọn trên val, test-dev chỉ đánh giá một lần sau khi khóa.

## 20. Ứng dụng desktop hiện có và ranh giới với nghiên cứu

Ứng dụng desktop Tkinter nhận video file, RTSP hoặc USB camera, cho chọn detector, tracker và ngưỡng; quản lý một session mỗi lần.

Luồng chính là phát hiện phương tiện → tracking → phát hiện biển số trong vùng xe theo chu kỳ → OCR → lưu track, quỹ đạo và ảnh minh chứng → tra cứu và xuất Excel. App dùng runtime riêng dưới tracking_app/engine, không gọi trực tiếp src/engine hay src/trackers khi chạy.

Model catalog của app có YOLO11n COCO ONNX cho camera đường phố, YOLO11m VisDrone ONNX cho video drone và YOLO11s ONNX cho biển số. RF-DETR và Faster R-CNN chưa được chuyển ONNX và chưa có trong catalog app. App mặc định dùng ByteTrack, plate_interval 5 frame và score biển số 0,25.

Tracking nghiên cứu năm lớp không đồng nghĩa app đã được benchmark tracking/biển số trên video giao thông thực tế.

OCR mặc định dùng hai mô hình PP-OCRv5 mobile đã export ONNX: text detector xác định vùng chữ trong crop biển số, bộ nhận dạng Latin chuyển thành chuỗi. Crop biển số có thể phóng 2 lần; chế độ auto thử ảnh gốc đã phóng rồi mới CLAHE nếu thất bại.

Lưu consensus theo track bằng cửa sổ 12 lần đọc; một frame chỉ cho một phiếu, muốn đổi chuỗi đã chọn cần tối thiểu 3 lần đọc mới và tổng confidence của chuỗi mới vượt 1,2 lần chuỗi cũ.

Điều này giảm đổi qua lại do OCR nhiễu nhưng không biến text sai thành đúng chắc chắn. Quỹ đạo trong app dùng tâm cạnh dưới của box theo pixel ảnh gốc; đây không phải tọa độ GPS. Khi đổi track ID, app không tự nối quỹ đạo cũ và mới.

Báo cáo còn nêu mục tiêu phát hiện vượt đèn đỏ: cấu hình vạch dừng, hướng xe, pha đèn; xác định quỹ đạo cắt vạch khi đèn đỏ, ghi timestamp và minh chứng.

VisDrone không có nhãn biển số, ký tự, trạng thái đèn hoặc sự kiện vi phạm, nên HOTA/mAP trong benchmark không đánh giá được chức năng đó.

Nếu camera là drone chuyển động, vạch dừng cần đăng ký theo cảnh hoặc tọa độ ổn định, GMC của tracker chưa đủ để định nghĩa vạch cố định trong thế giới.

Trả lời đúng thực trạng: ứng dụng có tracking, biển số và OCR theo README; phần vượt đèn đỏ trong báo cáo là định hướng/tính năng cần kiểm chứng riêng, không phải kết quả đã được định lượng bởi benchmark.

## 21. Những câu hỏi thầy có thể hỏi và câu trả lời ngắn

Vì sao chọn VisDrone? Vì đây là dữ liệu UAV nhiều đối tượng nhỏ, đông, nhiều chuyển động camera, có nhãn MOT và giao thức đánh giá phù hợp với bài toán. Hạn chế là miền dữ liệu Trung Quốc, không trực tiếp đại diện giao thông Việt Nam hay biển số.

Vì sao detector train 10 lớp còn tracker chấm 5 lớp? Detector học toàn bộ lớp đối tượng hữu ích; giao thức MOT của dự án chỉ đánh giá pedestrian, car, van, truck, bus. Mã lớp cần giữ chuẩn 1–10 ở cache, rồi lọc năm lớp trước tracking.

Motor không nằm trong HOTA đã báo cáo.

Vì sao train trên DET thay MOT train? Ảnh MOT rất dư thừa theo chuỗi; thử RF-DETR trên MOT thấy train mAP tăng nhưng val mAP giảm mạnh. DET tăng đa dạng bối cảnh. Đồng thời 57 ảnh DET có mã video trùng MOT val/test-dev được loại để giảm rò rỉ.

Vì sao cache ở 0,05? Để dùng chung prediction và bảo toàn box score thấp cho tầng hai ByteTrack/BoT-SORT. Ngưỡng vận hành được chọn sau trên val, không thể khôi phục box đã bị cắt ở bước cache.

Vì sao RF-DETR không dùng NMS? Huấn luyện bằng ghép một-một giữa query và ground truth nhằm cho mỗi vật một dự đoán. Nhưng việc không dùng NMS không bảo đảm không có false positive, và query không mang ID ổn định theo thời gian.

Vì sao Faster R-CNN chậm? Ảnh vào lớn, backbone/FPN, RPN, RoIAlign và RoI head hai giai đoạn; 745,5 GFLOPs/frame trong phép đo của nhóm. Việc chậm là ở cấu hình này trên L4, không khẳng định mọi triển khai Faster R-CNN đều chậm như nhau.

ByteTrack khác SORT ở đâu? Cả hai dựa vào hình học và Kalman, nhưng ByteTrack ghép hai tầng theo score, dùng box score thấp để cứu track cũ, có trạng thái track và điều kiện tạo track riêng. SORT lọc một ngưỡng rồi ghép IoU một tầng.

DeepSORT khác BoT-SORT ở đâu? Cả hai có ReID, nhưng DeepSORT dùng cascade ngoại hình/chuyển động; BoT-SORT kết hợp score cao/thấp, GMC, mô hình trạng thái khác và cost IoU–appearance. Vì nhiều thành phần khác nhau nên không thể quy toàn bộ chênh HOTA cho ReID.

Vì sao HOTA quan trọng hơn một mình MOTA? HOTA cân bằng độ chính xác phát hiện và liên kết; MOTA chủ yếu bị FP/FN chi phối. Ví dụ RF-DETR + ByteTrack có MOTA 45,21 cao hơn BoT-SORT 40,45 nhưng HOTA thấp hơn 45,97 so 49,85 và IDF1 thấp hơn 56,59 so 61,10.

Vì sao oracle chưa bằng 100? Tracker vẫn phải tạo, xác nhận và liên kết ID từ box ground truth; oracle không cấp ID. Đầu vào lý tưởng chỉ bỏ lỗi detection, không bỏ lỗi tracking hay hậu xử lý.

Điểm yếu lớn nhất là gì? AP small chỉ khoảng 8 và test-dev có gần một nửa box small; FN làm tracking đứt. Chênh oracle 38–43 điểm HOTA cho thấy dư địa lớn. Cần tránh khẳng định mọi chênh oracle thuần do detector.

Nếu phải chọn cấu hình triển khai? Với giới hạn thời gian, YOLO11m + ByteTrack có HOTA 47,09 và 101 FPS thành phần. Khi ưu tiên giữ ID, YOLO11m + BoT-SORT HOTA 51,67 và 25,9 FPS. Lựa chọn phải đo lại cả OCR, video I/O và phần cứng triển khai.

Nếu thầy hỏi cải tiến tiếp?

Tăng độ phân giải hoặc tiling để tăng AP small, đo lại tốc độ và hộp trùng; tinh chỉnh ReID đúng miền UAV; ablation GMC/ReID; đánh giá nhiều seed và khoảng tin cậy theo chuỗi; thu thập dữ liệu giao thông Việt Nam có nhãn biển số, đèn, vi phạm để kiểm chứng ứng dụng.

## 22. Những con số nên nhớ ngay trước buổi vấn đáp

Đề tài có 3 detector × 5 tracker = 15 tổ hợp; detector học 10 lớp, tracking chấm 5 lớp. MOT train/val/test-dev: 24.201/2.846/6.635 frame, 56/7/17 chuỗi. Ma trận cuối train detector trên DET train 6.414 ảnh sau khi loại 57 ảnh trùng nguồn; DET val 548 ảnh. Tỉ lệ small MOT train/val/test-dev: 24,9%/19,2%/47,8%.

Car chiếm 45,7% train, bus 0,9%, mất cân bằng khoảng 52 lần. Ba input: Faster R-CNN 800–1.333, YOLO 640, RF-DETR 576. Ba mAP50-95 MOT test-dev: Faster R-CNN 22,45, YOLO 23,17, RF-DETR 22,96. Ba HOTA BoT-SORT: 51,68; 51,67; 49,85. YOLO + ByteTrack 47,09 HOTA và 101,0 FPS; YOLO + BoT-SORT 51,67 HOTA và 25,9 FPS.

BoT-SORT oracle 94,93 HOTA. Benchmark FPS trên L4 không gồm video I/O, UI, OCR.

## 23. Nguồn trong dự án để đối chiếu khi cần

Báo cáo tổng hợp nằm tại report/main.tex. Bảng kết quả gốc nằm tại results/tables/det_val.csv, results/tables/det_test-dev.csv, results/tables/matrix_val_mean.csv, results/tables/matrix_testdev_mean.csv và results/tables/fps_bench.csv. Thống kê EDA nằm tại outputs/eda/tables/ và tài liệu mô tả tại docs/technical/02_du_lieu.md. Quy tắc chuyển đổi nhãn nằm trong data/tools/visdrone.py và data/tools/prepare.py.

Cấu hình RF-DETR chính là src/configs/detectors/rfdetr_m_576_det.yaml; YOLO trong results/yolo/T1_yolo11m_640_det/run_config.yaml; Faster R-CNN trong results/frcnn/T2_frcnn_r50_v2_det_seed0/run_config.json. Cấu hình tracker ở src/configs/trackers/. Ứng dụng desktop được mô tả tại tracking_app/README.md, catalog ở tracking_app/models.json, logic OCR consensus ở tracking_app/ocr/consensus.py.
