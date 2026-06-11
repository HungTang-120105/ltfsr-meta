# 03 — Thích nghi CLIP cho đuôi dài: Tip-Adapter + LIFT

Hướng "fancy" thứ hai cho thuyết trình, mạnh hơn `vlm_fusion`. Thay vì *trộn* CLIP với
model thị giác, ta **thích nghi (adapt) chính CLIP** vào dữ liệu lệch của ta — nhưng **đóng
băng backbone** và chỉ học vài tham số nhỏ. Chạy trên feature CLIP đã trích sẵn nên rất nhẹ,
hợp Kaggle.

> File số **03** trong `guides/`. Xem `01_how_to_run.md` (chạy chung) và `02_vlm_fusion.md` (CLIP fusion).

---

## 1. Vì sao không train thêm từ đầu?

Leaderboard from-scratch dừng ở `cmo` ~0.47 bal-acc — đó gần như trần của việc train một mạng
nhỏ từ đầu trên CIFAR-100-LT (IF=100), đặc biệt ở đuôi (5 ảnh/lớp). Khi đã có một foundation
model như **CLIP**, hướng mạnh nhất *không* phải làm vision tốt hơn mà là **mượn tri thức của
CLIP rồi tinh chỉnh nhẹ** cho bài toán của ta.

Cả hai phương pháp dưới đây đều chạy trên **feature CLIP đã đóng băng** (trích đúng **một lần**
cho mỗi split bằng `encode_clip_features`), nên train chỉ là vài giây/epoch.

## 2. Hai phương pháp

| | Học gì? | Ý tưởng | Kỳ vọng bal-acc |
|---|---|---|---|
| `clip_only` | không | CLIP zero-shot (so khớp tên lớp) | ~0.63–0.66 |
| `tip_adapter` | **không** (training-free) | phân loại bằng **truy hồi**: ảnh test giống ảnh train nào trong "bộ nhớ" → bầu theo lớp đó, trộn với zero-shot | ~0.68–0.72 |
| `tip_adapter_f` | ~5M tham số cache | như trên nhưng tinh chỉnh khoá cache vài epoch (Balanced Softmax) | ~0.71–0.75 |
| `lift` | **<1% tham số** (adapter nhỏ + head) | adapter residual + cosine head **khởi tạo từ feature chữ** của tên lớp, train với loss logit-adjusted | ~0.76–0.83 |

### Tip-Adapter (retrieval-augmented, *không train*)
Lập một **cache**: khoá = feature CLIP của ảnh train, giá trị = nhãn one-hot. Ảnh test được
phân loại bằng độ tương đồng với các khoá (một kiểu *k-NN mềm*), rồi cộng với điểm zero-shot
của CLIP. Hai số `alpha` (mức tin cache) và `beta` (độ sắc) **chọn trên val** theo balanced acc.

### LIFT (lightweight fine-tuning)
Thông điệp gốc của LIFT (ICML 2024): *"fine-tune nặng làm hại đuôi"*. Ta giữ tinh thần đó:
- **Adapter residual** `x + gate·MLP(x)` với `gate` khởi tạo **0** → lúc đầu model **đúng bằng
  CLIP zero-shot**, chỉ rời khỏi đó khi train cải thiện được đuôi.
- **Cosine head khởi tạo từ feature chữ** → ngay cả lớp 5 ảnh cũng bắt đầu từ một prototype có
  nghĩa (do CLIP hiểu *tên* lớp), thay vì khởi tạo ngẫu nhiên.
- **Loss logit-adjusted** (tái dùng `BalancedSoftmaxLoss`) để đuôi không bị head lấn át.

## 3. Cài đặt thêm
```bash
pip install open_clip_torch
```
Trên Kaggle: bật Internet để tải weight CLIP (~350MB cho ViT-B/32).

## 4. Cách chạy
1. Mở `notebooks/phase2_clip_adapt.ipynb`.
2. (Tùy chọn) smoke test: đặt `MAX_TRAIN_SAMPLES = 2000`, `LIFT_EPOCHS = 5` rồi Run All.
3. Chạy đầy đủ: `MAX_TRAIN_SAMPLES = None`, `LIFT_EPOCHS = 50`. Run All.
4. Notebook ghi `outputs/{clip_only,tip_adapter,tip_adapter_f,lift}/metrics.json`, cập nhật
   `comparison.csv` và xuất bảng VLM riêng `comparison_vlm.csv`.

Muốn điểm cao hơn (nặng hơn): đổi `CLIP_MODEL = "ViT-B-16"` hoặc `"ViT-L-14"`.

## 5. Cách trình bày (chính danh)
- Để 4 dòng này trong **bảng riêng "external knowledge (VLM)"**, cạnh `vlm_fusion`, tách khỏi
  leaderboard from-scratch — vì chúng dùng tri thức ngoài.
- Mạch kể đắt giá: **zero-shot → truy hồi (Tip-Adapter) → tinh chỉnh nhẹ (Tip-Adapter-F) →
  adapter + semantic init (LIFT)**: mỗi bước thêm rất ít tham số nhưng đuôi khá lên rõ rệt.
- Câu chốt: *"Không cần train mạng lớn từ đầu — đóng băng một foundation model rồi học <1%
  tham số với một loss nhận biết đuôi là đủ đóng khoảng cách ở các lớp hiếm."*
- Demo: lấy một ảnh lớp hiếm, in dự đoán `clip_only` (có thể sai) vs `lift` (đúng), và in
  `gate` của adapter + `few_shot_accuracy` tăng dần qua 4 phương pháp.

## 6. Lưu ý
- Mọi phương pháp dùng **cùng** feature đã trích (cùng thứ tự mẫu) nên so sánh công bằng.
- Cache của Tip-Adapter lệch về head (head nhiều khoá hơn) → ta chọn `alpha`/`beta` trên
  **balanced** val và train `tip_adapter_f` bằng Balanced Softmax để giữ đuôi trong tầm ngắm.
- `train_counts` lấy từ **train sau khi tách val** (không phải `class_counts.json` đầy đủ), để
  loss khớp đúng phân phối đang train.
- Đây là biến thể LIFT ở **mức feature** (adapter trên feature đóng băng) cho nhẹ trên Kaggle;
  thông điệp "đóng băng backbone + học <1% tham số + loss logit-adjusted" vẫn đúng.
