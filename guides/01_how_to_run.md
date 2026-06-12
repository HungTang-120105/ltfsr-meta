# 01 — Hướng dẫn chạy (How to run)

Hướng dẫn chạy toàn bộ thí nghiệm Long-Tail trên CIFAR-100-LT: từ chuẩn bị dữ liệu
→ train các phương pháp → các kỹ thuật tái dùng checkpoint → đọc kết quả.

> Đây là file số **01** trong thư mục `guides/`. Các hướng dẫn sau (chuẩn bị Kaggle,
> phân tích kết quả, thêm phương pháp mới…) sẽ là `02_…`, `03_…`.

---

## 1. TL;DR (đường đi nhanh nhất)

1. Cài môi trường + chuẩn bị dữ liệu (mục 3). **Làm trước tất cả.**
2. Mở **`notebooks/run_all_methods.ipynb`** → Run All → train hết các method, ra `outputs/comparison.csv`.
3. Mở **`notebooks/phase0_reuse.ipynb`** → Run All → ensemble / fusion / τ-norm trên các checkpoint vừa train.
4. (Song song được với 2–3) Mở **`notebooks/phase2_clip_adapt.ipynb`** → Run All → Tip-Adapter + LIFT (track VLM, điểm cao nhất).
5. (Sau bước 2, cần `cmo`) Mở **`notebooks/phase3_knowledge_sources.ipynb`** → Run All → nghiên cứu: LLM/DINOv2/diffusion/mixup + GLA + fusion.
6. Đọc `outputs/comparison.csv`, `comparison_vlm.csv`, `knowledge_sources.csv` và các hình `outputs/*.png`.

> Lần đầu nên **chạy thử nhanh**: trong cell config đặt `MAX_TRAIN_SAMPLES = 2000` và
> `EPOCHS = 5` để chắc pipeline chạy, rồi mới đặt lại full (`MAX_TRAIN_SAMPLES = None`,
> `EPOCHS = 200`).

---

## 2. Bản đồ thư mục cần biết

| Đường dẫn | Vai trò |
|---|---|
| `notebooks/run_experiment.ipynb` | Chạy **một** method (đặt `METHOD`, Run All) |
| `notebooks/run_all_methods.ipynb` | Chạy **tất cả** method + vẽ biểu đồ so sánh (**dùng cái này là chính**) |
| `notebooks/phase0_reuse.ipynb` | Tái dùng checkpoint đã train: ensemble, tier-fusion, τ-norm, CLIP fusion |
| `notebooks/phase2_clip_adapt.ipynb` | Adapt CLIP đóng băng: Tip-Adapter + LIFT (track VLM, **độc lập checkpoint vision**) |
| `notebooks/phase3_knowledge_sources.ipynb` | Nghiên cứu: LLM/DINOv2/diffusion/mixup + GLA + fusion (**cần checkpoint `cmo`**) |
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
| `VAL_FRACTION` | Tỉ lệ tách **validation** từ train để **chọn checkpoint**; test chỉ dùng báo cáo cuối | `0.1` |
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

## 5b. TRACK VLM — Adapt CLIP đóng băng (`phase2_clip_adapt.ipynb`)

**Quan trọng:** notebook này **chỉ cần dataset**, KHÔNG cần checkpoint vision → có thể chạy
**song song** với `run_all_methods.ipynb` (ở một session/máy khác), hoặc bất cứ lúc nào sau khi
có dữ liệu. Cell đầu tự `pip install open_clip_torch` (cần Internet trên Kaggle).

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `CLIP_MODEL` | Backbone CLIP (`ViT-B-32` nhẹ; `ViT-B-16`/`ViT-L-14` cao hơn, nặng hơn) | `"ViT-B-32"` |
| `LIFT_EPOCHS`, `LIFT_LR`, `LIFT_BOTTLENECK` | Huấn luyện adapter LIFT (train trên feature đã cache → vài giây/epoch) | `50`, `1e-3`, `64` |
| `TIPF_EPOCHS`, `TIPF_LR` | Fine-tune Tip-Adapter-F | `20`, `1e-3` |
| `MAX_TRAIN_SAMPLES` | Giới hạn ảnh để smoke test (`None` = full) | `None` |

Ghi `outputs/{clip_only, tip_adapter, tip_adapter_f, lift}/metrics.json`, cập nhật
`comparison.csv` và xuất **bảng VLM riêng** `comparison_vlm.csv`. Chi tiết: `guides/03_clip_adaptation.md`.

---

## 6. Chạy trên Kaggle (GPU)

1. Upload thư mục `CIFAR-100-LT/` làm **Kaggle Dataset** (xuất hiện ở `/kaggle/input/...`),
   hoặc đặt `BUILD_DATASET = True` để notebook tự tạo vào `/kaggle/working`.
2. Upload repo làm Dataset/Utility, hoặc `!git clone` trong cell đầu — notebook tự tìm `src/`.
3. Mở notebook, **bật GPU**, set `DATA_DIR` (vd `/kaggle/input/cifar-100-lt/CIFAR-100-LT`)
   và `OUTPUT_DIR = "/kaggle/working"`, rồi **Run All**.

### 6.1. Nạp checkpoint giữa các session (RẤT QUAN TRỌNG)

`/kaggle/working` là **tạm thời**: hết session là **mất**. Vì vậy checkpoint do
`run_all_methods` (session A) train ra **không tự có** ở session chạy `phase0_reuse`. Cách
chuyển checkpoint sang notebook sau:

1. Ở session A: sau khi train xong, bấm **"Save Version"** (Commit) → `/kaggle/working` được
   lưu lại thành **output của notebook đó** (gồm `outputs/<method>/best_model.pt`).
2. Ở session B (`phase0_reuse`): bấm **"Add Input"** → chọn notebook A (hoặc một Dataset bạn
   tạo từ output đó). Nó mount **read-only** tại `/kaggle/input/<tên>/...`.
3. Trong cell config của `phase0_reuse`, đặt:
   ```python
   OUTPUT_DIR = Path("/kaggle/working")      # nơi GHI kết quả mới (ghi được)
   CKPT_SOURCE = "/kaggle/input/<tên-notebook-A>/outputs"   # nơi ĐỌC checkpoint cũ
   ```
   Cell **"3b. Import checkpoints"** sẽ **copy** các thư mục `outputs/<method>/` từ `CKPT_SOURCE`
   vào `OUTPUT_DIR`, rồi phần còn lại chạy như cũ. (Để `CKPT_SOURCE = None` khi chạy cùng
   session hoặc chạy local — checkpoint đã nằm sẵn trong `OUTPUT_DIR`.)

> Vì sao phải tách ĐỌC/GHI: `/kaggle/input` chỉ đọc, không ghi `metrics.json` mới vào đó được;
> nên ta copy checkpoint sang `/kaggle/working` (ghi được) trước khi tái dùng.

### 6.2. Gộp kết quả chạy song song

Khi nhánh A và nhánh B (`phase2`) chạy ở 2 session khác nhau, mỗi bên có `comparison.csv` riêng
(thiếu dòng của bên kia). Để có bảng đầy đủ: Commit cả hai → ở một notebook bất kỳ, "Add Input"
cả hai output, copy hết `outputs/<*>/` về chung một `OUTPUT_DIR` (giống cơ chế `CKPT_SOURCE` ở
trên — lặp lại cho từng nguồn), rồi chạy `compare_runs(OUTPUT_DIR)` để dựng lại `comparison.csv`.

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

**Về validation (mới):** mỗi method được train trên train' (≈90% train, vẫn giữ profile
long-tail; tail giữ nguyên — chỉ head/medium góp ảnh cho val), chọn epoch tốt nhất theo
**val** rồi mới chấm trên **test**. Điều này khắt khe hơn (không còn chọn model trên test).
`phase0_reuse.ipynb` cũng chọn `τ` trên val. Vì cách chọn checkpoint đã đổi, **hãy chạy lại
`run_all_methods.ipynb`** nếu checkpoint cũ của bạn được train theo lối chọn-trên-test.

---

## 8. Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `No train/ folder under ...` | Sai `DATA_DIR`, hoặc chưa chạy `prepare_datasets.py` |
| `phase0` báo "skip (no checkpoint)" | Chưa chạy `run_all_methods.ipynb` để sinh `best_model.pt` |
| `phase2`/CLIP lỗi tải weight hoặc `open_clip` | Chưa bật **Internet** trên Kaggle (cần tải CLIP ~350MB); cell đầu đã tự `pip install` |
| `phase3` báo thiếu `cmo`, hoặc lỗi tải DINOv2/LLM | Chưa có `outputs/cmo/best_model.pt` (đối chứng) → chạy `run_all_methods`/nạp checkpoint; và bật **Internet** (CLIP+DINOv2+LLM). LLM tải chậm → đổi `LLM_MODEL` nhỏ hơn |
| Lỗi load checkpoint / mismatch trọng số | `PRETRAINED` ở phase0 không khớp lúc train (phải cùng `False`) |
| Hết VRAM | Giảm `BATCH_SIZE`, hoặc dùng `MAX_TRAIN_SAMPLES` để test |
| SupCon/CMO rất chậm trên CPU | Bình thường — chạy GPU, hoặc giảm epoch khi smoke test |
| Kết quả thấp bất thường ở từng method | Xem lại `metrics.csv` theo epoch; đảm bảo `EPOCHS=200`, không còn ở chế độ smoke |

---

## 9. Thứ tự chạy: cái nào TUẦN TỰ, cái nào SONG SONG

### 9.1. Bản đồ phụ thuộc

```
                         prepare_datasets.py          ← BẮT BUỘC chạy đầu tiên (1 lần)
                         (tạo data/CIFAR-100-LT/)
                                  │
            ┌─────────────────────┴───────────────────────────┐
            ▼                                                  ▼
   run_all_methods.ipynb  (NHÁNH A, train lâu)        phase2_clip_adapt.ipynb (NHÁNH B)
   → outputs/<method>/best_model.pt                   Tip-Adapter / LIFT
            │   (gồm cmo = control cho Phase 3)        (CHỈ cần dataset → độc lập,
            │                                            chạy SONG SONG với nhánh A)
      ┌─────┴───────────────┐
      ▼                     ▼
 phase0_reuse.ipynb    phase3_knowledge_sources.ipynb   (NHÁNH C, nghiên cứu)
 (ensemble/tier/        LLM / DINOv2 / diffusion / mixup
  τ-norm/CLIP fusion)   + GLA + fusion  (cần cmo checkpoint)
      │                     │                  │
      └─────────────────────┴──────────────────┘
                            ▼
   Gộp outputs/ → comparison.csv + comparison_vlm.csv + knowledge_sources.csv
```

### 9.2. Quy tắc

- **TUẦN TỰ (bắt buộc theo thứ tự):**
  1. `prepare_datasets.py` → trước mọi thứ.
  2. `run_all_methods.ipynb` → **rồi mới** `phase0_reuse.ipynb` **và** `phase3_knowledge_sources.ipynb`
     (phase0 cần `best_model.pt`; phase3 cần checkpoint **`cmo`** làm đối chứng).

- **SONG SONG (không phụ thuộc nhau):**
  - **Nhánh B** (`phase2_clip_adapt`) chỉ cần dataset → chạy đồng thời với **nhánh A**
    (`run_all_methods`) ở session khác. `phase2` nhanh (vài phút), `run_all_methods` lâu (6 method × 200 epoch).
  - Sau khi nhánh A xong, **`phase0_reuse` và `phase3` độc lập nhau** → chạy song song được (cả hai
    chỉ đọc checkpoint, không sửa). `phase3` cần **Internet** (tải CLIP + DINOv2 + LLM).
  - **Trong** `run_all_methods`, 6 method cũng độc lập với nhau. Muốn chạy song song nhiều method,
    mở nhiều session rồi dùng **một trong hai cách** (cả hai giờ cùng giao thức val-split → số khớp
    nhau và khớp `run_all_methods`):
  - **Trong** `run_all_methods`, 6 method cũng độc lập với nhau. Muốn chạy song song nhiều method,
    mở nhiều session rồi dùng **một trong hai cách** (cả hai giờ cùng giao thức val-split → số khớp
    nhau và khớp `run_all_methods`):
    - mỗi session một bản `run_all_methods.ipynb` với `METHODS` rút gọn (vd session 1:
      `["baseline","balanced_softmax","cmo"]`; session 2: phần còn lại), hoặc
    - mỗi session một `run_experiment.ipynb` đặt một `METHOD` khác nhau (xem mục 10).
    Sau đó gộp `outputs/` lại.

- **Lưu ý khi chạy song song trên Kaggle:** mỗi session có `/kaggle/working` **riêng**, không thấy
  output của nhau. Sau khi xong, tải các thư mục `outputs/<method>/` từ mọi session về **chung một
  chỗ**, rồi chạy lại cell cuối (`compare_runs`) — hoặc `phase0_reuse` cell 9 / `phase2` cell 7 —
  để dựng lại `comparison.csv` đầy đủ. (Chạy tuần tự trên cùng một máy thì không cần bước gộp này.)

### 9.3. Đường đi tối giản (1 máy, tuần tự)

```
prepare_datasets.py
        ▼
run_all_methods.ipynb     (train hết, gồm cmo)        → outputs/<method>/, comparison.csv
        ▼
phase0_reuse.ipynb        (ensemble + tier_fusion + τ-norm + CLIP fusion; BEST_SINGLE="cmo")
        ▼
phase2_clip_adapt.ipynb   (Tip-Adapter + LIFT, track VLM điểm cao nhất)
        ▼
phase3_knowledge_sources.ipynb  (nghiên cứu: LLM/DINOv2/diffusion/mixup + GLA + fusion; cần cmo)
        ▼
Đọc comparison.csv + comparison_vlm.csv + knowledge_sources.csv
```

> **Mẹo tiết kiệm GPU:** nhánh A (`run_all_methods`) là phần nặng duy nhất. Nếu đã có sẵn các
> `outputs/<method>/best_model.pt` (đặc biệt `cmo`), bỏ qua nhánh A và chạy thẳng phase0/phase2/phase3
> (nạp checkpoint theo **mục 6.1** nếu ở session Kaggle khác).

---

## 10. `run_experiment.ipynb` là gì? (chạy nhanh MỘT method)

Notebook "xem nhanh" để chạy **đúng một** method trong một lần: đặt `METHOD` (một trong
`baseline | balanced_softmax | decoupling | supcon | meta`) rồi **Run All**. Pipeline:
*configure → load data → train 1 method → evaluate → visualise*, ghi vào `outputs/<METHOD>/`
(training curve, confusion matrix, t-SNE).

`METHOD` nhận đủ **8** lựa chọn: `baseline | balanced_softmax | decoupling | supcon | meta |
mixup | cutmix | cmo`.

Dùng khi nào: muốn soi kỹ **một** method (debug, xem t-SNE/đường học, thử hyperparameter
nhanh) mà không phải chạy cả 6 method như `run_all_methods`.

> ✅ **Cùng giao thức với `run_all_methods`:** notebook này giờ cũng tách `VAL_FRACTION` để **chọn
> checkpoint theo balanced-acc trên val**, test chỉ dùng báo cáo cuối (xem mục 7). Vì vậy một lần
> chạy ở đây **so trực tiếp được** với bảng tổng hợp của `run_all_methods`. Bảng báo cáo chính thức
> vẫn nên lấy từ `run_all_methods.ipynb` (chạy đủ method một lượt), còn `run_experiment` là công cụ
> dò nhanh từng method — số liệu hai bên nhất quán.
