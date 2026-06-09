# 01 — Hướng dẫn chạy (How to run)

Hướng dẫn chạy toàn bộ thí nghiệm Long-Tail trên CIFAR-100-LT: từ chuẩn bị dữ liệu
→ train các phương pháp → các kỹ thuật tái dùng checkpoint → đọc kết quả.

> Đây là file số **01** trong thư mục `guides/`. Các hướng dẫn sau (chuẩn bị Kaggle,
> phân tích kết quả, thêm phương pháp mới…) sẽ là `02_…`, `03_…`.

---

## 1. TL;DR (đường đi nhanh nhất)

1. Cài môi trường + chuẩn bị dữ liệu (mục 3).
2. Mở **`notebooks/run_all_methods.ipynb`** → Run All → train hết các method, ra `outputs/comparison.csv`.
3. Mở **`notebooks/phase0_reuse.ipynb`** → Run All → ensemble / fusion / τ-norm trên các checkpoint vừa train.
4. Đọc `outputs/comparison.csv` và các hình `outputs/*.png`.

> Lần đầu nên **chạy thử nhanh**: trong cell config đặt `MAX_TRAIN_SAMPLES = 2000` và
> `EPOCHS = 5` để chắc pipeline chạy, rồi mới đặt lại full (`MAX_TRAIN_SAMPLES = None`,
> `EPOCHS = 200`).

---

## 2. Bản đồ thư mục cần biết

| Đường dẫn | Vai trò |
|---|---|
| `notebooks/run_experiment.ipynb` | Chạy **một** method (đặt `METHOD`, Run All) |
| `notebooks/run_all_methods.ipynb` | Chạy **tất cả** method + vẽ biểu đồ so sánh (**dùng cái này là chính**) |
| `notebooks/phase0_reuse.ipynb` | Tái dùng checkpoint đã train: ensemble, tier-fusion, τ-norm |
| `data/prepare_datasets.py` | Tạo dataset CIFAR-100-LT |
| `src/` | Toàn bộ code (datasets, models, trainers, evaluation) |
| `outputs/<method>/` | Kết quả từng lần chạy (metrics, checkpoint, hình) |
| `outputs/comparison.csv` | Bảng so sánh tổng hợp mọi method |

---

## 3. Chuẩn bị (chạy local)

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

# Tạo dataset CIFAR-100-LT (chỉ làm 1 lần)
python data/prepare_datasets.py --dataset cifar100-lt --data_dir ./data --overwrite

# Kiểm tra dataset
python data/validate_cifar_lt.py --data_dir ./data/CIFAR-100-LT
```

Sau bước này phải có thư mục `data/CIFAR-100-LT/` gồm `train/`, `test/`, `class_counts.json`.

---

## 4. GIAI ĐOẠN 1 — Train tất cả phương pháp (`run_all_methods.ipynb`)

Mở notebook, chỉ cần sửa **cell config (cell 2)**, rồi **Run All**.

### 4.1. Các tham số quan trọng trong cell config

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `METHODS` | Danh sách method sẽ train, theo thứ tự | `["baseline","balanced_softmax","decoupling","supcon","meta","cmo"]` |
| `PRETRAINED` / `IMAGE_SIZE` | `False`/`32` = từ scratch (setup chính). `True`/`224` = ImageNet (bảng phụ) | `False` / `32` |
| `EPOCHS` | Số epoch (from-scratch cần dài) | `200` |
| `MAX_TRAIN_SAMPLES` | Giới hạn ảnh để smoke test (`None` = full) | `None` |
| `CRT_EPOCHS`, `CRT_LR` | Giai đoạn cRT của decoupling/supcon | `10`, `0.1` |
| `PRETRAIN_EPOCHS`, `PRETRAIN_LR`, `TEMPERATURE` | SupCon | `200`, `0.5`, `0.07` |
| `AUG_ALPHA`, `MIX_PROB` | Augmentation cho `mixup`/`cutmix`/`cmo` | `1.0`, `0.5` |
| `N_WAY`, `K_SHOT`, `N_QUERY`, … | Meta-learning | 5 / 5 / 15 |

### 4.2. Ý nghĩa từng method

| `METHOD` | Sửa ở tầng | Mô tả ngắn |
|---|---|---|
| `baseline` | — | ResNet + cross-entropy (mốc tham chiếu) |
| `balanced_softmax` | loss | Cộng log class-prior vào logits |
| `decoupling` | classifier | Train feature → cRT lại head cân bằng (BN đã được đóng băng đúng cách) |
| `supcon` | representation | SupCon + cRT |
| `meta` | bonus | Episodic ProtoNet (báo cáo trên trục few-shot) |
| `cmo` | data | **Giai đoạn 1**: Balanced-Softmax + CutMix thiên tail (Context-rich Minority Oversampling) |
| `mixup` / `cutmix` | data | Biến thể augmentation khác (thêm vào `METHODS` nếu muốn so sánh) |

Mỗi method ghi kết quả vào `outputs/<method>/`. Cell cuối sinh
`comparison.csv`, `comparison_metrics.png` và các `overlay_*.png` (mỗi measure 1 hình,
gộp mọi method).

---

## 5. GIAI ĐOẠN 0 — Tái dùng checkpoint (`phase0_reuse.ipynb`)

**Chạy SAU** `run_all_methods.ipynb` (cần các `best_model.pt` đã có). Không train lại,
chỉ inference. Sửa cell config rồi Run All.

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `REUSE_METHODS` | Các run dùng để ensemble | `["baseline","balanced_softmax","decoupling","supcon"]` |
| `BEST_SINGLE` | Model mạnh nhất, dùng cho tier-fusion và τ-norm | `"balanced_softmax"` |
| `FUSION_WEIGHTS` | Trọng số prototype theo tầng (`many/medium/few`) | `{many:0.0, medium:0.3, few:0.8}` |
| `TAU_VALUES` | Các giá trị τ để thử | `[0.5, 0.75, 1.0]` |

Các kỹ thuật chạy: **ensemble** (+TTA), **tier_fusion** (head dùng classifier, tail dùng
prototype), **τ-normalization**. Mỗi kỹ thuật ghi thêm một dòng vào `outputs/` và cập nhật
lại `comparison.csv` + `comparison_metrics.png`.

> Muốn ensemble gồm cả model `cmo`: thêm `"cmo"` vào `REUSE_METHODS` (và đặt
> `BEST_SINGLE = "cmo"` nếu cmo là model mạnh nhất).

---

## 6. Chạy trên Kaggle (GPU)

1. Upload thư mục `CIFAR-100-LT/` làm **Kaggle Dataset** (xuất hiện ở `/kaggle/input/...`),
   hoặc đặt `BUILD_DATASET = True` để notebook tự tạo vào `/kaggle/working`.
2. Upload repo làm Dataset/Utility, hoặc `!git clone` trong cell đầu — notebook tự tìm `src/`.
3. Mở notebook, **bật GPU**, set `DATA_DIR` (vd `/kaggle/input/cifar-100-lt/CIFAR-100-LT`)
   và `OUTPUT_DIR = "/kaggle/working"`, rồi **Run All**.

---

## 7. Đọc kết quả

Trong `outputs/`:

```
outputs/
├── comparison.csv                 # bảng tổng hợp mọi method (sắp theo balanced_accuracy)
├── comparison_metrics.png         # bar chart gộp các method theo từng metric
├── overlay_val_accuracy.png …     # mỗi measure 1 hình, các method là các đường
└── <method>/
    ├── metrics.json               # metric cuối (acc, balanced_acc, macro_f1, g_mean, many/med/few)
    ├── metrics.csv                # lịch sử theo epoch
    ├── best_model.pt              # checkpoint tốt nhất
    └── *.png                      # confusion matrix, t-SNE …
```

**Thước đo cần nhìn (long-tail):** ưu tiên `balanced_accuracy` và `few_shot_accuracy`,
không chỉ `accuracy`. (Test set cân bằng nên `accuracy == balanced_accuracy`.)

---

## 8. Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `No train/ folder under ...` | Sai `DATA_DIR`, hoặc chưa chạy `prepare_datasets.py` |
| `phase0` báo "skip (no checkpoint)" | Chưa chạy `run_all_methods.ipynb` để sinh `best_model.pt` |
| Lỗi load checkpoint / mismatch trọng số | `PRETRAINED` ở phase0 không khớp lúc train (phải cùng `False`) |
| Hết VRAM | Giảm `BATCH_SIZE`, hoặc dùng `MAX_TRAIN_SAMPLES` để test |
| SupCon/CMO rất chậm trên CPU | Bình thường — chạy GPU, hoặc giảm epoch khi smoke test |
| Kết quả thấp bất thường ở từng method | Xem lại `metrics.csv` theo epoch; đảm bảo `EPOCHS=200`, không còn ở chế độ smoke |

---

## 9. Thứ tự chạy khuyến nghị (tổng kết)

```
prepare_datasets.py
        │
        ▼
run_all_methods.ipynb        (Giai đoạn 1: train hết, gồm cmo)
        │  → outputs/<method>/, comparison.csv
        ▼
phase0_reuse.ipynb           (Giai đoạn 0: ensemble + tier_fusion + τ-norm)
        │  → cập nhật comparison.csv + comparison_metrics.png
        ▼
Đọc outputs/comparison.csv  → tinh chỉnh (FUSION_WEIGHTS, MIX_PROB, AUG_ALPHA, τ) nếu cần
```
