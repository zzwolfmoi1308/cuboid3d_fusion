# CVAT 3D Fusion Suite (Pre-label & QA)

Chrome Extension (Manifest V3) kết hợp Python Local Agent (`127.0.0.1:8766`) hỗ trợ **Pre-label 3D Cuboid đa model**, **QA Audit**, **Chiếu khung 3D lên Camera 2D realtime**, và **Quản lý Profile theo nhiều Guideline/Task** trên CVAT (100% chạy nội bộ, không gửi dữ liệu ra ngoài).

---

## 1. Kiến trúc & Tính năng chính

### Hệ thống gồm 2 phần
1. **Chrome Extension (`manifest.json`, `sidepanel/`, `src/`)**
   - **Tab `Pre-label`**:
     - Tăng size điểm & độ sáng Point Cloud 3D, tô màu điểm theo nhãn Cuboid, hiện khoảng cách tới xe Ego, phím tắt `[` / `]` chuyển nhanh Camera 2D.
     - **Chiếu Cuboid 3D lên Camera 2D realtime** (nạp `calib.json` 1 lần và tự lưu).
     - **Pre-label đa model** theo 1 frame hoặc toàn bộ dải frame, xem trước danh sách Track theo mức độ tin cậy (`HIGH`, `REVIEW`, `UNCERTAIN_CLASS`, `MISSED?`), tự động lưu **Snapshot JSON** trước khi chèn vào CVAT (hỗ trợ `Ctrl+Z`, không tự bấm Save).
   - **Tab `QA Audit`**: Quét lỗi Metadata (9 luật R1–R9: kích thước, phạm vi khoảng cách, bước nhảy, góc quay, thuộc tính...) và quét **PCD** (kiểm tra chạm mặt đất, mật độ điểm trong box).
   - **Tab `Size & Check`**: Tra cứu bảng kích thước tiêu chuẩn và checklist theo Profile.
   - **Tab `Issue`**: Tổng hợp & format nhanh báo cáo lỗi gửi Mentor/QA.
   - **Tab `Settings`**: Quản lý Profile đa Guideline, import Guideline từ file (`.docx`, `.pdf`, `.txt`, `.md`, `.csv`) và map nhãn AI ↔ nhãn CVAT.

2. **Python Fusion Agent (`agent/`, cổng `127.0.0.1:8766`)**
   - Kết hợp 4 nguồn tín hiệu độc lập:
     - **[A] CenterPoint-Pillar** (`0.20m`, `centerpoint_pillar02_nus.pth`)
     - **[B] PointPillars-FPN** (`0.25m`, `pointpillars_fpn_nus.pth`)
     - **[C] Cụm hình học LiDAR** (GroundGrid + DBSCAN + Search-based L-Shape OBB)
     - **[D] Camera 2D YOLO11s** (`yolo11s.pt`, batch full ảnh + high-res crop)
   - Ước lượng chuyển động Ego bằng **LiDAR ICP Odometry** và ghép **Track** xuyên chuỗi frame.

---

## 2. Yêu cầu hệ thống

- **Trình duyệt**: Google Chrome / Edge (hỗ trợ Manifest V3 & Side Panel).
- **Python**: 3.10 – 3.12 (khuyến nghị Anaconda / venv).
- **GPU**: Khuyến nghị NVIDIA GPU hỗ trợ CUDA (đã tối ưu chạy FP32 trên dòng tầm trung như Quadro P2000; vẫn có thể chạy fallback trên CPU nhưng sẽ chậm hơn).

---

## 3. Hướng dẫn cài đặt & khởi chạy

### Bước 1: Clone repository
```bash
git clone https://github.com/zzwolfmoi1308/cuboid3d_fusion.git
cd cuboid3d_fusion
```

### Bước 2: Cài đặt môi trường cho Python Fusion Agent
```bash
cd agent
pip install -r requirements.txt
```
Cài PyTorch có CUDA phù hợp với GPU của máy (ví dụ CUDA 12.1, tương thích tốt cả GPU đời cũ `sm_61` như P2000):
```bash
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
```
*(Nếu chạy thuần CPU, cài bản PyTorch CPU tiêu chuẩn: `pip install torch torchvision`.)*

Đảm bảo thư mục `agent/models/` đã có đủ 3 file weights:
- `centerpoint_pillar02_nus.pth`
- `pointpillars_fpn_nus.pth`
- `yolo11s.pt`

### Bước 3: Chạy Fusion Agent
- **Trên Windows**: Chạy trực tiếp file `agent/run_agent.bat` (script tự giải phóng cổng `8766` cũ và khởi động `server.py`).
- **Hoặc chạy tay qua terminal**:
  ```bash
  cd agent
  python server.py
  ```
- Kiểm tra nhanh bằng cách mở trình duyệt vào `http://127.0.0.1:8766/health` — nếu trả về `"ok": true` là Agent đã sẵn sàng.

### Bước 4: Cài đặt Chrome Extension
1. Mở Chrome và truy cập `chrome://extensions/`.
2. Bật **Developer mode** (Chế độ dành cho nhà phát triển) ở góc trên bên phải.
3. Chọn **Load unpacked** (Tải tiện ích đã giải nén) → trỏ vào thư mục gốc `cuboid3d_fusion/` (nơi chứa `manifest.json`).
4. Ghim icon **CVAT 3D Fusion Suite** lên thanh công cụ Chrome.

---

## 4. Hướng dẫn sử dụng nhanh

1. **Mở Job 3D trên CVAT** (mặc định hỗ trợ `http://localhost:8080/*` và `https://cvat.edufun.ai/*`; nếu dùng domain khác, bấm nút **Cấp quyền** ở góc trên Side Panel).
2. Bấm icon Extension để mở **Side Panel**. Kiểm tra badge `agent` đã chuyển xanh (`127.0.0.1:8766`).
3. **Nạp `calib.json`** (nếu muốn chiếu 3D→2D và bật kiểm class bằng Camera 2D):
   - Trong mục *Chiếu Cuboid lên Camera 2D*, chọn thư mục/file `calib.json` hoặc dán đường dẫn local rồi bấm **Nạp** (chỉ cần làm 1 lần).
4. **Chọn Profile & khớp nhãn**:
   - Chọn Profile phù hợp ở dropdown hoặc bấm **Upload Guideline**.
   - Bấm **Tự khớp nhãn** để ánh xạ class của AI sang đúng tên label trong Job CVAT hiện tại.
5. **Chạy Pre-label**:
   - Bấm **Thử 1 frame** để kiểm tra nhanh chất lượng nhận diện trên frame hiện tại.
   - Chỉnh phạm vi frame (`From → To`), chọn kiểu đối tượng (`Track` hoặc `Shape`) rồi bấm **Chạy toàn phạm vi**.
   - Lọc theo tag (`HIGH`, `REVIEW`, `UNCERTAIN_CLASS`, `MISSED?`), bỏ chọn các track sai rồi bấm **Chèn vào CVAT**.
   - Kiểm tra lại trên CVAT (có thể `Ctrl+Z` nếu cần) trước khi nhấn **Save** trên CVAT.
6. **Chạy QA Audit**:
   - Chuyển sang tab **QA Audit** → bấm **Quét QA Metadata** và/hoặc **Quét PCD** để rà soát lỗi trước khi nộp job.

---

## 5. Các giới hạn nhận diện hiện tại (Known Limitations — cần team lưu ý & nâng cấp)

Hiện tại 2 detector 3D (`CenterPoint-Pillar`, `PointPillars-FPN`) sử dụng weights pretrained trên **nuScenes**, kết hợp **YOLO11s** pretrained trên **COCO** (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`). Do đó một số lớp đối tượng sau **chưa nhận diện tốt** và luôn cần kiểm tra/sửa tay hoặc nâng cấp thêm:

| Đối tượng / Tình huống | Tình trạng hiện tại | Nguyên nhân kỹ thuật |
| :--- | :--- | :--- |
| **`Barrier` (Rào chắn)** | Dễ bỏ sót, nhận nhầm hoặc gộp nhiều barrier liền nhau thành 1 box dài; đôi khi sai hướng trục 90°. | COCO YOLO11s **không có class `Barrier`** (`CAM_SKIP` bỏ qua bước xác nhận camera). Model LiDAR nuScenes chỉ quen barrier kiểu nhựa/bê tông chuẩn; rào chắn mỏng, dải phân cách liên tiếp hoặc đặt sát nhau dễ bị DBSCAN gộp cụm. Cần chọn đúng **Quy ước trục Barrier** (`length_is_x` / `width_is_x`) trong UI. |
| **`Traffic_cone` (Cọc tiêu)** | Tỷ lệ sót cao ở khoảng cách trung bình/xa, dễ nhầm với cột nhỏ hoặc bụi cây. | Không có trong COCO YOLO11s (`CAM_SKIP`), kích thước rất nhỏ (`0.25–0.65m`), ít điểm LiDAR (`MIN_FG_POINTS = 3`). Mặc định đang tắt nếu Profile không bật. |
| **`Trailer` & `Construction_vehicle`** | Dễ bị nhận nhầm sang `Truck` hoặc `Bus`, hoặc bị đánh dấu `DETECTOR_DISAGREE` / `UNCERTAIN_CLASS`. | COCO YOLO11s không phân biệt riêng 2 lớp này (thường chỉ ra `Truck`), nên pipeline phải đặt ngưỡng score LiDAR cao hơn (`0.48`) và phụ thuộc mạnh vào hình học 3D. |
| **`Bicycle` vs `Motorcycle` (khi bật tách 2 bánh)** | Dễ nhầm lẫn giữa xe đạp và xe máy, đặc biệt khi ở xa hoặc bị người lái che khung xe. | Điểm LiDAR ở khoảng cách >25m rất thưa; độ chính xác phụ thuộc lớn vào bước crop Camera 2D YOLO11s. |
| **Người đi bộ / Xe 2 bánh ở xa hoặc bị che khuất** | Hay bị gán `LOW_VISIBILITY` hoặc sót track. | Khi số điểm LiDAR `< 4–5` điểm và camera không nhìn rõ (đêm tối, ngược sáng, góc khuất), bộ lọc sẽ loại để tránh false positive. |
| **Cụm hình học `MISSED?` (DBSCAN)** | Có thể sinh box thừa ở cột điện, biển báo, bụi cây khi không có Camera 2D xác nhận. | Cụm `geo` chỉ phân loại dựa trên kích thước bao ngoài khi detector chính bỏ sót; khuyến nghị luôn bật Camera 2D + `calib.json` chuẩn khi dùng. |

### Gợi ý hướng nâng cấp cho team
- **Cải thiện `Barrier` & `Traffic_cone`**: Bổ sung model 2D open-vocabulary / fine-tuned YOLO (thêm class `barrier`, `traffic_cone`, `trailer`, `construction_vehicle`) vào `agent/camera_check.py`, và tách cụm barrier nối tiếp theo chiều dài chuẩn trong `agent/pipeline.py`.
- **Fine-tune / Domain adaptation**: Fine-tune `CenterPoint` / `PointPillars` trên dữ liệu LiDAR thực tế của dự án (đặc biệt khi khác số tia LiDAR hoặc vị trí gắn cảm biến so với nuScenes).
- **Cải thiện Tracking**: Nâng cấp Kalman Filter / dự đoán vận tốc trong `agent/tracker.py` cho các đối tượng di chuyển nhanh hoặc bị che khuất dài (>4 frame).

---

## 6. Kiểm thử (Tests)

- **QA Rules (Node.js)**:
  ```bash
  node tests/run_qa_tests.mjs
  ```
- **Fusion Pipeline (Python / pytest)**:
  ```bash
  pytest tests/test_fusion_pipeline.py
  ```

---

## 7. Cấu trúc thư mục

```text
cuboid3d_fusion/
├── manifest.json                 # Cấu hình Chrome Extension MV3
├── icons/                        # Icon extension
├── sidepanel/                    # Giao diện Side Panel (HTML/CSS/JS)
├── src/
│   ├── background.js             # Service worker
│   ├── content_bridge.js         # Cầu nối extension <-> trang CVAT
│   ├── page_bridge.js            # Tương tác trực tiếp CVAT Canvas/ObjectState
│   ├── qa/                       # Luật QA metadata, PCD worker, Profile & Guideline importer
│   ├── fix/                      # V2 Safe Assist actions
│   └── issue/                    # Format báo cáo lỗi cho Mentor
├── agent/
│   ├── server.py                 # FastAPI server (127.0.0.1:8766)
│   ├── pipeline.py               # Pipeline kết hợp 4 nguồn tín hiệu
│   ├── centerpoint_pillar.py     # Detector 3D chính
│   ├── pointpillars_fpn.py       # Detector 3D phụ
│   ├── camera_check.py           # Kiểm chứng 2D YOLO11s + giải nén context_image
│   ├── geometry.py               # GroundGrid, L-Shape, chuẩn hóa kích thước & trục Barrier
│   ├── odometry.py               # LiDAR ICP Odometry
│   ├── tracker.py                # Ghép track đa frame
│   ├── pcd_io.py                 # Đọc/parse file PCD
│   ├── requirements.txt          # Thư viện Python
│   ├── run_agent.bat             # Script khởi chạy nhanh trên Windows
│   └── models/                   # Trọng số CenterPoint, PointPillars-FPN, YOLO11s
└── tests/                        # Unit test cho QA & Fusion pipeline
```
