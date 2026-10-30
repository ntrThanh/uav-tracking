# 6. Quy trình làm việc

## 6.1. Nguyên tắc: code ở local, chạy trên server

Mã nguồn được viết và quản lý bằng git ở máy cá nhân. Server GPU chỉ dùng để chạy: đồng bộ code từ local lên, chạy job nền, rồi kéo kết quả về. Không sửa code trực tiếp trên server, vì những thay đổi đó sẽ không được ghi lại trong git và dễ bị ghi đè ở lần đồng bộ sau.

| | Local | Server |
|---|---|---|
| Vai trò | Viết code, quản lý git, xem kết quả | Chạy tiền xử lý, train, suy luận, đánh giá |
| Thư mục | Thư mục clone repo `uav-tracking` | `~/AI_LAB/ntrthanh/rfdetr/` |
| Dữ liệu gốc | Không cần | `data/raw/` |
| Kết quả lớn | Chỉ kéo về bảng, biểu đồ | `runs/`, `results/`, `data/processed/` |

## 6.2. Server

| Thông tin | Giá trị |
|---|---|
| Địa chỉ | `128.168.137.198`, user `vpcuser` |
| GPU | 2 × NVIDIA L4, 23 GB mỗi card |
| CPU, RAM | 32 nhân, 156 GB |
| Ổ đĩa | 500 GB, còn khoảng 149 GB sau khi có dữ liệu |
| Thư mục dự án | `~/AI_LAB/ntrthanh/rfdetr/` |
| Môi trường | conda `rfdetr` (Python 3.11) tại `~/miniconda3/envs/rfdetr` |

Môi trường `rfdetr` hiện có các gói phân tích trong `requirements.txt`. Để train cần cài thêm RF-DETR và PyTorch:

```bash
conda activate rfdetr
pip install -e models/rf-detr        # kéo theo torch, torchvision, pytorch-lightning
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 6.3. Đồng bộ code và kết quả

Đẩy code từ local lên server (không đụng tới dữ liệu và kết quả trên server):

```bash
rsync -av --exclude .git --exclude __pycache__ \
      --exclude data/raw --exclude data/processed --exclude outputs --exclude runs --exclude results \
      ./ 128.168.137.198:AI_LAB/ntrthanh/rfdetr/
```

Kéo kết quả nhẹ từ server về local:

```bash
rsync -av 128.168.137.198:AI_LAB/ntrthanh/rfdetr/outputs/ outputs/
rsync -av 128.168.137.198:AI_LAB/ntrthanh/rfdetr/results/tables/ results/tables/
```

Hai lệnh này sẽ được đóng gói thành script trong `src/scripts/`.

## 6.4. Chạy job dài trên server

Job train kéo dài nhiều giờ phải chạy nền với `nohup` để không bị dừng khi mất kết nối SSH:

```bash
cd ~/AI_LAB/ntrthanh/rfdetr
nohup ~/miniconda3/envs/rfdetr/bin/python -m src.training.train_rfdetr \
      --config src/configs/detectors/rfdetr_medium.yaml > logs/T3_rfdetr_m_576.log 2>&1 &
tail -f logs/T3_rfdetr_m_576.log
```

Chọn GPU bằng `CUDA_VISIBLE_DEVICES=0` hoặc `1` khi hai người cùng train trên server. Kiểm tra GPU đang rảnh bằng `nvidia-smi` trước khi chạy.

## 6.5. Chạy notebook trên server

Notebook được chạy không giao diện bằng nbconvert rồi kéo bản đã chạy về:

```bash
cd ~/AI_LAB/ntrthanh/rfdetr/src/notebooks
~/miniconda3/envs/rfdetr/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 01_eda_visdrone_mot.ipynb
```

Notebook dùng đường dẫn tương đối `../../data/raw` và `../../outputs/eda`, nên phải nằm trong `src/notebooks/`.

## 6.6. Tải dữ liệu

Link Google Drive chính thức của VisDrone thường báo "Quota exceeded" khi tải không đăng nhập. Cách tải ổn định là dùng `rclone` với tài khoản Google của chính mình:

```bash
rclone backend copyid <remote>: 1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh ./   # train
rclone backend copyid <remote>: 1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu ./   # val
rclone backend copyid <remote>: 14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu ./   # test-dev
for z in VisDrone2019-MOT-*.zip; do unzip -tq "$z" && unzip -q "$z"; done
```

Nếu làm việc trên cùng server thì không cần tải lại, dữ liệu đã có sẵn ở `~/AI_LAB/ntrthanh/rfdetr/data/raw/`.

## 6.7. Quy ước git

| Quy ước | Nội dung |
|---|---|
| Nhánh | `main` luôn chạy được; mỗi việc lớn làm trên nhánh riêng như `feat/rfdetr-train`, `feat/trackers` |
| Commit | Mô tả ngắn gọn bằng tiếng Anh, dòng đầu dưới 72 ký tự |
| Không đưa lên git | Dữ liệu, trọng số (`*.pt`, `*.pth`), `runs/`, `results/dets`, `results/tracks`, video |
| Đưa lên git | Code, cấu hình, `data/splits/`, bảng kết quả `results/tables/*.csv`, biểu đồ |
| Tiền xử lý | Chỉ sửa trong `data/tools/visdrone.py` và `prepare.py`, chạy lại `check.py` trước khi commit |

## 6.8. Ghi log thí nghiệm

Mỗi lần train hoặc chạy tracker ghi một dòng vào `results/runs_log.csv`:

```
run_id, ngày, người chạy, mô hình, dữ liệu, tham số chính, GPU, thời gian, mAP / HOTA / IDF1 / FPS, ghi chú
```

Mỗi thư mục run lưu kèm file cấu hình đã dùng, phiên bản thư viện (`pip freeze`) và commit hash của repo để tái lập được kết quả. Seed cố định bằng 0. Chạy lại cùng một tổ hợp hai lần phải ra cùng kết quả.

## 6.9. Quy tắc dùng split

Train chỉ dùng để huấn luyện. Val dùng để chọn checkpoint, chọn ngưỡng và dò tham số tracker. Test-dev chỉ chạy **một lần** cho cấu hình cuối cùng và không được điều chỉnh gì dựa trên kết quả test-dev. Vi phạm quy tắc này làm kết quả test-dev mất giá trị khách quan.

## 6.10. Tiến độ

| Tuần | Thành (RF-DETR) | Thi (Faster R-CNN) | Tiến (YOLO11m) |
|---|---|---|---|
| 1 | Repo, dữ liệu, tiền xử lý dùng chung, `eval_det.py` | Code 5 tracker và cấu hình | Baseline zero-shot, ground truth TrackEval, `eval_track.py`, oracle × 5 tracker |
| 2 | Train RF-DETR | Train Faster R-CNN | Train YOLO11m |
| 3 | Train (tiếp) | Train (tiếp) | Cache val, 5 tracker, đánh giá |
| 4, **giữa kỳ** | Tổng hợp, slide giữa kỳ | Cache và 5 tracker nếu xong | Bảng YOLO11m × 5 tracker + oracle |
| 5 | Dò ngưỡng | Dò ngưỡng | Dò ngưỡng |
| 6 | Chạy test-dev 3 × 5, đo FPS, ghép bảng, heatmap, biểu đồ đánh đổi (cả nhóm) | | |

Mốc giữa kỳ chỉ yêu cầu một pipeline hoàn chỉnh (YOLO11m × 5 tracker + oracle trên val).

## 6.11. Danh sách kiểm tra

- [x] Tải và kiểm tra VisDrone2019-MOT
- [x] Notebook thống kê dữ liệu
- [x] Tiền xử lý dùng chung, `check.py` đạt
- [x] Clone RF-DETR 1.11.2
- [ ] Cài torch và `rfdetr` trên server
- [ ] `src/evaluation/eval_det.py` (pycocotools)
- [ ] Code 5 tracker và `src/configs/trackers/`
- [ ] Ground truth TrackEval theo lớp, `eval_track.py`, oracle × 5 tracker
- [ ] Train T3 (RF-DETR 576), T1 (YOLO11m), T2 (Faster R-CNN)
- [ ] Cache detection val và test-dev cho 3 detector
- [ ] Ma trận 3 × 5 trên val, dò ngưỡng
- [ ] T3-hr (RF-DETR 1024)
- [ ] Chạy test-dev một lần, ghép bảng, vẽ biểu đồ
