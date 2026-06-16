# 00 — Mục tiêu đầu ra & thông điệp của project

File này trả lời 2 câu hỏi cho người đọc/người chấm: **(1) project muốn nói điều gì?** và
**(2) mỗi giai đoạn cần ra cái gì để chứng minh điều đó?** Đọc file này trước, rồi sang
`01_how_to_run.md` để biết cách chạy.

---

## 1. Bài toán & vì sao khó

- **Dữ liệu:** CIFAR-100-LT, hệ số mất cân bằng **IF=100** — lớp nhiều nhất ~500 ảnh, lớp ít
  nhất chỉ **5 ảnh**. Test set **cân bằng** (100 ảnh/lớp).
- **Cái bẫy:** `accuracy` thường bị **lớp head lấn át** → nhìn cao nhưng đuôi gần như chết.
  Vì test cân bằng nên `accuracy == balanced_accuracy`; ta **luôn báo cáo theo
  `balanced_accuracy` và `few_shot_accuracy`** (độ chính xác trên nhóm lớp hiếm).

## 2. Thông điệp cốt lõi (3 ý, đây là "linh hồn" buổi thuyết trình)

1. **Đo cho đúng trước đã.** Trên long-tail, chọn sai thước đo là sai từ gốc. Ta dùng
   balanced-accuracy + few-shot, và **chọn model trên tập validation** (không đụng test) để
   không tự lừa mình.
2. **Nút thắt nằm ở *bộ phân loại*, không phải đặc trưng.** Cùng một encoder, chỉ cần *cân
   bằng lại đầu phân loại* (balanced-softmax, cRT, τ-norm) là đuôi khá lên rõ — chứng tỏ
   feature đã đủ tốt, cái lệch là **ranh giới quyết định** thiên về head.
3. **5 ảnh là không đủ để *học từ đầu* — hãy *mượn* tri thức.** Khi có một foundation model
   (CLIP) hiểu *nghĩa của tên lớp*, việc **thích nghi nhẹ** nó vào đuôi (vài % hoặc <1% tham
   số) đóng được khoảng cách mà train-from-scratch không với tới.

> Một câu tóm tắt: *"Đo đúng → sửa bộ phân loại → mượn tri thức ngoài. Mỗi bước rẻ hơn nhưng
> đuôi tốt lên nhiều hơn bước trước."*

## 3. Cấu trúc kể chuyện (mạch 3 hồi)

| Hồi | Câu hỏi | Phương pháp | Đầu ra chính |
|---|---|---|---|
| **A. Train từ đầu** | Can thiệp ở tầng nào thì giúp đuôi? | baseline → balanced_softmax → decoupling → supcon → meta → cmo | `comparison.csv` (leaderboard from-scratch) |
| **B. Tái dùng (free lunch)** | Vắt thêm điểm mà **không train lại**? | ensemble(+TTA), tier_fusion, τ-norm | thêm dòng vào `comparison.csv` |
| **C. Tri thức ngoài (fancy)** | Mượn CLIP thì đuôi tới đâu? | CLIP fusion → Tip-Adapter → LIFT | `comparison_vlm.csv` (bảng VLM riêng) |

---

## 4. Mong muốn đầu ra theo từng PHASE

> Số dưới đây là **kỳ vọng/ballpark** (CIFAR-100-LT, IF=100, train từ đầu trừ khi nói khác).
> **Đã chạy xong → số THẬT (cả CIFAR-100-LT lẫn CUB-200-LT) nằm trong [`../REPORT.md`](../REPORT.md).**
> File này giữ làm tài liệu *mục tiêu/kế hoạch*; xu hướng/thứ hạng đúng như kỳ vọng.

### PHASE 1 — Train tất cả phương pháp (`run_all_methods.ipynb`)
**Mục tiêu:** một leaderboard from-scratch trung thực, cho thấy can thiệp tail-aware giúp đuôi.

| METHOD | tầng can thiệp | balanced_acc (≈) | few_shot (≈) |
|---|---|---|---|
| baseline | — | 0.40 | 0.08 |
| balanced_softmax | loss | 0.44 | 0.18 |
| decoupling (cRT) | classifier | 0.41 | 0.09 |
| supcon | representation | 0.37 | 0.02 |
| meta (ProtoNet) | bonus (few-shot axis) | 0.35 | — |
| **cmo** | data | **0.47** | **0.30** |

**Đầu ra cần có:** `outputs/<method>/{metrics.json, metrics.csv, best_model.pt, *.png}`,
`outputs/comparison.csv`, `comparison_metrics.png`, các `overlay_*.png`.
**Câu chốt:** *"Sửa loss/đầu phân loại rẻ mà đuôi tăng gấp 2–3 lần; cmo (data + balanced-softmax)
tốt nhất trong nhóm from-scratch."*

### PHASE 0 — Tái dùng checkpoint (`phase0_reuse.ipynb`)
**Mục tiêu:** tăng điểm **không train lại**, chỉ inference trên checkpoint Phase 1.

| Kỹ thuật | ý tưởng | balanced_acc (≈) |
|---|---|---|
| ensemble (+TTA) | trung bình xác suất nhiều model | cao hơn model đơn ~1–3đ |
| tier_fusion | head dùng linear head, tail dùng prototype | đuôi nhỉnh hơn |
| τ-norm | chuẩn hóa chuẩn trọng số theo tần suất | gỡ thiên lệch head |

**Đầu ra cần có:** thêm dòng `ensemble / ensemble_tta / tier_fusion / tau_norm` vào
`comparison.csv`; cập nhật `comparison_metrics.png`.
**Câu chốt:** *"Có những điểm số 'cho không' nếu biết phối hợp checkpoint sẵn có."*

### PHASE 2 — Thích nghi CLIP đóng băng (`phase2_clip_adapt.ipynb`) + CLIP fusion (Phase 0, mục 8)
**Mục tiêu:** track **tri thức ngoài (VLM)** — phần "fancy", điểm cao nhất. Để **bảng riêng**.

| Phương pháp | học gì | balanced_acc (≈) | few_shot (≈) |
|---|---|---|---|
| clip_only (zero-shot) | không | 0.63–0.66 | ~0.60 |
| vlm_fusion (CLIP + vision) | trộn, α chọn trên val | 0.66–0.70 | ~0.60 |
| tip_adapter | không (truy hồi) | 0.68–0.72 | — |
| tip_adapter_f | ~vài triệu tham số cache | 0.71–0.75 | — |
| **lift** | **<1% tham số** (adapter + cosine head) | **0.76–0.83** | **cao nhất** |

**Đầu ra cần có:** `outputs/{clip_only, vlm_fusion, tip_adapter, tip_adapter_f, lift}/metrics.json`,
bảng VLM riêng `outputs/comparison_vlm.csv`.
**Câu chốt:** *"5 ảnh không đủ để học từ đầu; ta đóng băng một foundation model rồi học <1% tham
số với một loss nhận biết đuôi — đủ để đóng khoảng cách ở các lớp hiếm."*

### PHASE 3 — Nghiên cứu: loại tri thức ngoài nào cứu đuôi? (`phase3_knowledge_sources.ipynb`)
**Đây là đóng góp chính.** Câu hỏi: *với đuôi 5 ảnh, tri thức **ngôn ngữ** (LLM), **thị giác thứ
hai** (DINOv2), hay **sinh dữ liệu** (diffusion) giúp nhiều nhất, và chúng có **bù trừ** không?*

| Expert | Nguồn tri thức |
|---|---|
| `cmo` | nội tại (đối chứng from-scratch) |
| `clip_llm` | ngôn ngữ (LLM-enriched prototypes) |
| `dino_lift` | thị giác thứ hai (DINOv2) |
| `lift_clip_diff` | sinh dữ liệu (diffusion feature aug) |
| `fusion_tailaware` | gộp nhận biết đuôi (kiểm bù trừ) |

Cộng **GLA** (debias foundation model, tổng quát hóa `balanced_softmax`).
**Đầu ra cần có:** `outputs/knowledge_sources.csv` (bảng theo nhóm-shot — **bảng chính của bài**),
`outputs/<expert>/metrics.json`. Chi tiết: `04_knowledge_sources.md`.
**Câu chốt:** *"Mỗi nguồn tri thức mạnh ở một vùng của đuôi; phối hợp đúng cách (nhận biết đuôi +
debias công bằng) mới khai thác hết — và chứng minh chúng bù trừ nhau."*

---

## 5. "Đầu ra" mong muốn cuối cùng (deliverables)

1. **Hai bảng tách bạch** (rất quan trọng để chính danh khi chấm):
   - `comparison.csv` — leaderboard **from-scratch** (Phase 1 + Phase 0).
   - `comparison_vlm.csv` — track **external-knowledge (VLM)** (Phase 2). *Không trộn chung* vì
     dùng tri thức ngoài, khác setup.
2. **Biểu đồ** so sánh theo metric + overlay đường học theo epoch.
3. **Một slide "demo đuôi":** lấy 1 ảnh lớp hiếm → in dự đoán của `clip_only`/`vision` (có thể sai)
   vs `lift` (đúng); và biểu đồ `few_shot_accuracy` tăng dần qua 4 mức CLIP.
4. **Kết luận khớp 3 thông điệp ở mục 2** — không chỉ khoe điểm, mà rút ra *vì sao* mỗi bước giúp đuôi.

## 6. Tiêu chí "thành công" của project

- Thứ hạng đúng kỳ vọng: `cmo` dẫn đầu nhóm from-scratch; track VLM cao hơn rõ; `lift` cao nhất.
- `few_shot_accuracy` (đuôi) **tăng đơn điệu** theo mạch A → B → C.
- Mọi số đều chọn-model-trên-val, **không** đụng test khi tinh chỉnh → kết quả đáng tin.
- Trình bày tách 2 bảng, nêu được *thông điệp* chứ không chỉ con số.

> Chi tiết cách chạy: `01_how_to_run.md`. Giải thích CLIP fusion: `02_vlm_fusion.md`.
> Giải thích Tip-Adapter + LIFT: `03_clip_adaptation.md`. Giải thích từng method: `docs/01–07_*.md`
> (06 = cmo, 07 = CLIP fusion/Tip-Adapter/LIFT).
