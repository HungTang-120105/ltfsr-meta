# 05 — Chạy trên dataset mới (đa dataset), ví dụ CUB-200-LT

Mục tiêu thiết kế: **đổi dataset chỉ cần đổi *cách đọc data*, không sửa logic phương pháp.** Mọi
notebook đọc dữ liệu qua **ImageFolder layout chuẩn** + 2 file metadata. Tài liệu này mô tả layout
đó và cách thêm một dataset long-tail mới (đã làm sẵn cho **CUB-200-LT**).

---

## 1. Layout chuẩn mà mọi notebook mong đợi

```
<DATASET>-LT/
  train/class_000/ ... class_{C-1}/      # ảnh train (đã subsample long-tail)
  test/class_000/  ... class_{C-1}/      # ảnh test (đánh giá)
  class_counts.json    # {"0": count, ..., "C-1": count}  — số ảnh train mỗi lớp
  class_names.json     # ["tên lớp 0", ..., "tên lớp C-1"]  — theo đúng thứ tự label
```

- Thư mục `class_{i:03d}` ⇒ nhãn ImageFolder = `i`. `class_names[i]` là tên đọc được của lớp `i`
  (dùng cho CLIP zero-shot / LLM prompt / khởi tạo head LIFT).
- `class_counts.json` cho biết hồ sơ long-tail (head→tail).

Hai hàm đọc (trong `src/datasets/cifar_lt.py`):
- `load_class_counts(DATA_DIR)` → `{int: int}`
- `load_class_names(DATA_DIR)` → `list[str]` (đọc `class_names.json`; nếu thiếu → mặc định CIFAR-100).

> Đây là **toàn bộ** chỗ "khác nhau" giữa các dataset. Phần phương pháp (CLIP/DINOv2/LIFT/fusion)
> nhận `feature + nhãn + tên lớp` nên **không cần đổi**.

## 2. CUB-200-LT đã được tạo sẵn

Script: **`data/prepare_cub_lt.py`** — chuyển CUB-200-2011 (~60 ảnh/lớp) thành long-tail theo đúng
công thức mũ của CIFAR-100-LT: dùng **toàn bộ ảnh**, tách test cân bằng, subsample phần còn lại.

```bash
python data/prepare_cub_lt.py --imbalance_factor 10 --overwrite
# → data/CUB-200-LT/  (200 lớp, head=50, tail=5, train≈3868, test=2000 cân bằng, + class_names.json)
```

Cơ chế (mới): dùng **toàn bộ ~60 ảnh/lớp**, tách **test cân bằng** (`--test_per_class`, mặc định
10/lớp), phần còn lại (~50/lớp) làm pool long-tail. Mặc định **`max_images=50`, IF=10** → head 50,
tail 5, train ≈ 3868, test 2000.

Lưu ý đặc thù CUB:
- Pool ~50 train/lớp ⇒ với IF=10 head=50/tail=5. Tăng `--imbalance_factor` (vd 20) cho đuôi gắt hơn.
  Trình bày như **"CUB-200-LT, fine-grained, IF=10"** — không so trực tiếp IF với CIFAR.
- Test **cân bằng** (10/lớp) ⇒ `accuracy == balanced_accuracy` như CIFAR. (Muốn test lớn hơn:
  tăng `--test_per_class`, nhưng train pool giảm.)
- **Ngưỡng shot-group cho CUB (head 50):** dùng **`MANY_THRESHOLD=20`, `FEW_THRESHOLD=10`**
  (many 78 / medium 66 / few 56 — cân bằng). *Không* dùng 15/6 (lệch: many 102 / few 9).
- Tên lớp là **tên chim tiếng Anh** (vd "Black footed Albatross") ⇒ CLIP/LLM hiểu được (khác iNat
  dùng danh pháp Latin).

## 3. Cách chạy track foundation (Phase 2 / Phase 3) trên CUB-200-LT

Trong cell **config** của `phase2_clip_adapt.ipynb` hoặc `phase3_knowledge_sources.ipynb`, đổi:

```python
DATA_DIR = PROJECT_DIR / "data" / "CUB-200-LT"
MANY_THRESHOLD = 20        # CUB head 50 -> 20 / 10 (CIFAR là 100 / 20)
FEW_THRESHOLD = 10
USE_CMO = False            # chưa có checkpoint từ-đầu cho CUB (xem mục 4)
```

`NUM_CLASSES` và `CLASS_NAMES` được **tự suy từ dataset** trong cell data — không cần sửa. Rồi
**Run All**. Mọi expert (CLIP zero-shot, LIFT, DINOv2-LIFT, diffusion/mixup, GLA, fusion, ablation)
chạy y như CIFAR, chỉ khác dữ liệu nguồn.

## 4. Track từ-đầu (Hồi 1) trên dataset mới — lưu ý

`run_all_methods.ipynb` (từ-đầu) được tuned cho ảnh **32×32**. Trên CUB (ảnh lớn) cần:
- đặt `IMAGE_SIZE` ≥ 64 (để `build_transforms` **Resize** trước khi crop; ở 32 nó crop một mẩu nhỏ);
- biết rằng từ-đầu trên CUB (fine-grained, ít ảnh) sẽ **yếu** — điều này *củng cố* thông điệp "cần
  tri thức ngoài". Nếu cần một đối chứng `cmo` cho Phase 3, train riêng một `cmo` CUB rồi đặt
  `CMO_DIR`; hoặc dùng `clip_zeroshot` làm mốc sàn.

## 5. Thêm một dataset khác nữa
Viết một script tạo layout ở **mục 1** (như `prepare_cub_lt.py`): subsample train thành long-tail,
copy test, ghi `class_counts.json` + `class_names.json`. Xong là mọi notebook chạy được — chỉ đổi
`DATA_DIR` + ngưỡng shot-group trong config.
