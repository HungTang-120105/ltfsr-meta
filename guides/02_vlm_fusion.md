# 02 — Vision-Language fusion (CLIP): "ảnh cho lớp thường, chữ cho lớp hiếm"

Phương pháp "fancy" để thuyết trình: ghép model thị giác train-from-scratch với một
**CLIP đóng băng** (nhận diện qua *tên lớp*, zero-shot, không train). Chạy hoàn toàn ở
**inference**, tái dùng checkpoint đã train.

> File số **02** trong `guides/`. Xem `01_how_to_run.md` cho phần chạy chung.

---

## 1. Ý tưởng (một câu)

Model nhỏ giỏi lớp **nhiều ảnh** nhưng thua lớp **5 ảnh**. CLIP hiểu *nghĩa của tên lớp*
và **không thấy** dữ liệu lệch của ta → mạnh đều ở mọi lớp. Trộn xác suất hai chuyên gia,
trọng số `alpha` (mức tin CLIP) **chọn trên val** → kết quả thường vượt cả hai.

Không phải "đoán mẫu là hiếm hay không" (không biết trước tương lai): `alpha` là **một số
chung**, tinh chỉnh trên val; ta chỉ trộn điểm của hai chuyên gia cho **mọi lớp** rồi argmax.

## 2. Cài đặt thêm

```bash
pip install open_clip_torch
```
Trên Kaggle: bật Internet để tải weight CLIP (~350MB), hoặc thêm Kaggle Dataset chứa weight.

## 3. Cách chạy

1. Chạy `run_all_methods.ipynb` trước (cần checkpoint thị giác).
2. Mở `phase0_reuse.ipynb`, ở cell config đặt **`BEST_SINGLE = "cmo"`** (model thị giác mạnh nhất).
3. Run All. Mục **"8. Vision-Language fusion (CLIP)"** sẽ in 3 dòng vào `outputs/` và bảng so sánh:
   - `vision_only` — chỉ model thị giác.
   - `clip_only` — chỉ CLIP zero-shot.
   - `vlm_fusion` — trộn, `alpha` chọn trên val.

Đổi CLIP lớn hơn cho số cao hơn (nặng hơn): sửa `model_name='ViT-B-16'` hoặc `'ViT-L-14'`
trong cell fusion.

## 4. Cách trình bày (chính danh)

- Để `vlm_fusion` ở **bảng riêng "có tri thức ngoài (VLM)"**, tách khỏi leaderboard
  from-scratch — vì nó dùng kiến thức ngoài, không cùng setup.
- Thông điệp: *"5 ảnh không đủ để học; ta mượn tri thức ngôn ngữ từ một VLM đóng băng. Một
  trọng số trộn đơn giản (chọn trên val) đóng được khoảng cách ở đuôi."*
- Demo: lấy một ảnh lớp hiếm → in dự đoán của vision (sai) vs CLIP (đúng) vs fusion.

## 5. Kỳ vọng (CIFAR-100-LT, IF=100)

| | balanced_acc | few-shot (tail) |
|---|---|---|
| cmo (thị giác tốt nhất) | ~0.47 | ~0.30 |
| clip_only (ViT-B/32) | ~0.65 | ~0.60 |
| **vlm_fusion** | **~0.66–0.70** | **~0.60+** |

## 6. Lưu ý
- Căn mẫu: cả vision và CLIP đều đọc **cùng** tập (val/test) với `shuffle=False`; notebook có
  `assert` để chắc thứ tự mẫu khớp giữa hai chuyên gia.
- Tên lớp CIFAR-100 nhúng sẵn trong `src/experts/clip_expert.py` theo đúng thứ tự label
  (folder `class_{id:03d}`), nên CLIP gán đúng tên cho từng lớp.
