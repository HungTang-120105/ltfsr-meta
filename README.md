# Long-tail Few-shot Recognition with Meta-Learning (LTFSR)

## Mục tiêu

- Giải quyết vấn đề không cân bằng dữ liệu (long-tail distribution) trong bài toán nhận dạng ít mẫu (few-shot).
- Khắc phục thiên kiến của Softmax truyền thống đối với các lớp ít mẫu (tail classes).
- Xây dựng mô hình có khả năng học nhanh các lớp mới với hiệu suất cao trên tất cả các lớp, đặc biệt là tail classes.

## Phương pháp

### 1. Phân tích Vấn đề
- **Vấn đề chính:** Dữ liệu thực tế luôn bị lệch (long-tail). Bộ phân loại Softmax bị thiên kiến vì độ lớn vectơ trọng số tỷ lệ với số lượng mẫu, khiến vùng quyết định của tail classes bị thu hẹp.

### 2. Giải pháp Chính
- **Prototype Learning:** Thay thế Softmax bằng học nguyên mẫu, tính khoảng cách Euclidean từ mẫu đến tâm cụm của mỗi lớp.
- **Meta-Learning:** Sử dụng episodic training để mô phỏng kịch bản few-shot, giúp mô hình học cách học (learn to learn).
- **Meta-Weighting:** Gán trọng số tự động cho các mẫu quan trọng, ưu tiên các lớp tail bị lỗi cao.
- **MetaSAug:** Sinh thêm biến thể đặc trưng cho lớp hiếm dựa trên thông tin từ lớp phổ biến.
- **Representation Learning:** Sử dụng SupCon (Supervised Contrastive Learning) để tạo ra cụm dữ liệu tách biệt rõ rệt.
- **Prototype Refinement:** Cập nhật tâm cụm linh hoạt dựa trên query set thay vì chỉ lấy trung bình tĩnh.

## Đóng góp

- Tóm tắt luồng tư duy toàn diện từ phân tích vấn đề đến giải pháp thực thi.
- Đề xuất kiến trúc mô hình tích hợp: Prototype Head + Meta-Learning + MetaSAug + SupCon.
- Lộ trình phát triển từng giai đoạn rõ ràng để kiểm soát và đánh giá hiệu quả.

## Dataset

### Các dataset sử dụng
- **ImageNet-LT:** Phiên bản long-tail của ImageNet với 1,000 lớp, phân bố theo quy luật lũy thừa.
- **Places-LT:** Phiên bản long-tail của Places365 dataset (scene recognition).
- **CelebA-Spoof:** Dataset gian lận nhận dạng khuôn mặt với phân bố không cân bằng.

### Quy tắc phân chia lớp
- **Head classes:** > 100 mẫu/lớp
- **Medium classes:** 20-100 mẫu/lớp
- **Tail classes:** < 20 mẫu/lớp

### Protocol Episodic Training
- **Support set size:** 5 mẫu/lớp (5-way-5-shot)
- **Query set size:** 15 mẫu/lớp (5-way-15-query)
- **Episode composition:** Lấy mẫu cân bằng từ Head/Medium/Tail classes

## Các bước thực hiện

### Giai đoạn 1: Baseline
- Xây dựng baseline (ResNet + Cross-Entropy) để đo lường độ lệch trên tail classes.
- Đánh giá ban đầu trên Many/Med/Few accuracy.

### Giai đoạn 2: Meta-Learning & Prototype Head
- Triển khai khung Meta-Learning với episodic training.
- Thay thế Softmax bằng Prototype Head.
- Tích hợp Meta-Weighting để ưu tiên mẫu quan trọng.

### Giai đoạn 3: Nâng cao biểu trưng
- Tích hợp SupCon cho Representation Learning.
- Thêm MetaSAug để sinh dữ liệu cho tail classes.
- Áp dụng Prototype Refinement động.

### Giai đoạn 4: Tối ưu hóa & Đánh giá
- Tinh chỉnh siêu tham số (learning rate, margin, embedding dimension).
- So sánh đầy đủ với baseline và các phương pháp khác.
- Ablation study từng thành phần.

## Cấu trúc Thư mục

```
ltfsr-meta/
├── README.md                    # Tài liệu dự án
├── Summary.md                   # Tóm tắt luồng công việc
├── requirements.txt             # Các thư viện cần thiết
├── setup.py                     # Setup script
├── config/
│   ├── config_baseline.yaml     # Cấu hình baseline
│   ├── config_meta_learning.yaml # Cấu hình meta-learning
│   └── config_full_model.yaml   # Cấu hình mô hình đầy đủ
├── data/
│   ├── prepare_datasets.py      # Script chuẩn bị dataset
│   └── dataloader.py            # Custom DataLoader cho episodic training
├── models/
│   ├── backbone.py              # ResNet backbone
│   ├── prototype_head.py        # Prototype Learning head
│   ├── meta_learner.py          # Meta-Learning module
│   └── full_model.py            # Model tích hợp
├── training/
│   ├── trainer.py               # Training loop
│   ├── loss_functions.py        # SupCon, meta-weighting losses
│   └── meta_aug.py              # MetaSAug implementation
├── evaluation/
│   ├── metrics.py               # Many/Med/Few Accuracy, G-mean
│   ├── evaluate.py              # Evaluation script
│   └── ablation_study.py        # Ablation study script
├── utils/
│   ├── logger.py                # Logging utility
│   ├── checkpoint.py            # Save/load checkpoints
│   └── visualization.py         # Visualization tools
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_prototype_learning.ipynb
│   ├── 03_meta_learning.ipynb
│   └── 04_full_model_results.ipynb
└── results/
    ├── baseline/
    ├── meta_learning/
    └── full_model/
```

## Validation

### Chỉ số đánh giá
- **Many Accuracy:** Độ chính xác trên head classes (nhiều mẫu).
- **Medium Accuracy:** Độ chính xác trên medium classes.
- **Few Accuracy:** Độ chính xác trên tail classes (ít mẫu).
- **G-mean:** Trung bình hình học của Many/Med/Few Accuracy (chỉ số cân bằng).

### Phương pháp đánh giá
- So sánh tổng thể (Overall Accuracy, G-mean) với baseline.
- Ablation study: từng module (Prototype, Meta-Learning, SupCon, MetaSAug, Refinement).
- Phân tích biểu đồ kết quả trên từng nhóm lớp (Head/Medium/Tail).

## Hướng dẫn Chạy

### 1. Chuẩn bị môi trường
```bash
git clone <repo_url>
cd ltfsr-meta
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Chuẩn bị Dataset
```bash
python data/prepare_datasets.py --dataset ImageNet-LT --data_dir /path/to/data
```

### 3. Chạy Baseline
```bash
python training/trainer.py --config config/config_baseline.yaml --output_dir results/baseline
```

### 4. Chạy Meta-Learning
```bash
python training/trainer.py --config config/config_meta_learning.yaml --output_dir results/meta_learning
```

### 5. Chạy Mô hình Đầy đủ
```bash
python training/trainer.py --config config/config_full_model.yaml --output_dir results/full_model
```

### 6. Đánh giá & Ablation Study
```bash
python evaluation/evaluate.py --checkpoint results/full_model/best_model.pth --dataset ImageNet-LT
python evaluation/ablation_study.py --dataset ImageNet-LT
```
