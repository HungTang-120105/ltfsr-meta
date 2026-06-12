# 04 — Phase 3: Loại tri thức ngoài nào cứu được lớp đuôi? (nghiên cứu)

Đây là **đóng góp chính** của project — biến "thử vài method" thành một **câu hỏi nghiên cứu**
và trả lời nó bằng thực nghiệm, tái dùng mọi phần trước.

> File số **04** trong `guides/`. Chạy: `notebooks/phase3_knowledge_sources.ipynb`.

---

## 1. Câu hỏi nghiên cứu

> **Với lớp đuôi chỉ 5 ảnh, loại tri thức *ngoài* nào giúp nhiều nhất — *ngôn ngữ* (LLM),
> *thị giác thứ hai* (DINOv2), hay *sinh dữ liệu* (diffusion) — và chúng có *bù trừ* nhau không?**

Câu hỏi này hay vì nó cần đủ 3 thứ, và cả ba đều tái dùng phần cũ:
- **một mốc đối chứng** = mô hình from-scratch `cmo` (Phase 1): "không mượn ngoài thì tới đâu?"
- **một thước đo công bằng** = GLA (tổng quát hóa `balanced_softmax`, Method 2).
- **một phép kiểm bù trừ** = fusion (nâng cấp `vlm_fusion`/`tier_fusion`, Phase 0).

## 2. Bốn nguồn tri thức (mỗi cái là một "expert")

| Expert | Nguồn tri thức | Cách làm (bài tham khảo) |
|---|---|---|
| `cmo` | **nội tại** (dữ liệu của ta) | mô hình from-scratch tốt nhất — **đối chứng** |
| `clip_llm` | **ngôn ngữ** | LLM sinh mô tả lớp → prototype CLIP giàu hơn (CuPL, ICCV'23) |
| `dino_lift` | **thị giác thứ 2** | DINOv2 (tự giám sát, không ngôn ngữ) + LIFT, init bằng class-mean |
| `lift_clip_diff` | **sinh dữ liệu** | diffusion sinh feature đuôi → train LIFT trên real+synthetic (LDMLR, '24) |
| `lift_clip_mixup` | **augment (cmo-style)** | tail-aware feature mixup → train LIFT (ý tưởng `cmo` chuyển vào feature space) |
| `lift_clip` | vision-language | LIFT trên CLIP (mốc "external" mạnh) |

Cộng **GLA** (gỡ bias pretraining của CLIP, NeurIPS'23) và **fusion nhận biết đuôi** (trọng số
riêng cho many/medium/few, chọn trên val).

> **Tái dùng phương pháp cũ trong track hiện đại:** `balanced_softmax` (Method 2) **đã là loss
> huấn luyện** của *mọi* expert LIFT/Tip-Adapter-F (`BalancedSoftmaxLoss`), và GLA tổng quát hóa
> nó. `cmo` (Method 6) được chuyển vào feature space thành `lift_clip_mixup`. So `lift_clip_mixup`
> (trộn feature thật) với `lift_clip_diff` (sinh feature) trả lời: *augment kiểu nào giúp đuôi hơn?*
> Bật/tắt bằng `USE_MIXUP` / `MIXUP_ALPHA`.

## 3. Cài đặt & cách chạy

```bash
pip install open_clip_torch transformers   # notebook tự cài; DINOv2 qua torch.hub
```
Trên Kaggle: **bật Internet** (tải CLIP, DINOv2, và LLM lần đầu) + **bật GPU**.

1. Cần sẵn `outputs/cmo/best_model.pt` (chạy `run_all_methods.ipynb` trước; hoặc nạp checkpoint
   theo `guides/01` mục 6.1). Đặt `CMO_DIR` cho đúng.
2. Mở `phase3_knowledge_sources.ipynb`. Smoke test: `MAX_TRAIN_SAMPLES=2000`, `LIFT_EPOCHS=5`,
   `DIFFUSION_EPOCHS=20`. Rồi chạy full.
3. Có thể **bật/tắt từng nguồn** bằng `USE_LLM / USE_DINO / USE_DIFFUSION / USE_GLA / USE_CMO`.
4. Kết quả: `outputs/<expert>/metrics.json`, **`outputs/knowledge_sources.csv`** (bảng theo
   nhóm-shot — bảng chính của bài), cập nhật `comparison.csv`.

> LLM sinh mô tả **một lần** rồi cache vào `outputs/class_descriptions.json`; lần sau đọc lại,
> không chạy LLM nữa (commit file này để tái dùng giữa các session Kaggle).

## 4. Đọc kết quả = trả lời câu hỏi (slide kết luận)

Mở `knowledge_sources.csv` và đọc theo 4 ý:
1. **Nguồn nào cứu đuôi nhất?** so cột `few_shot_accuracy` giữa `clip_llm` (ngôn ngữ),
   `dino_lift` (thị giác 2), `lift_clip_diff` (sinh dữ liệu).
2. **Có bù trừ không?** `fusion_tailaware` có vượt expert đơn tốt nhất không (nhất là ở `few`)?
   Nếu có ⇒ các nguồn **bổ sung** cho nhau, không trùng lặp.
3. **Có cần mượn ngoài không?** mọi expert ngoài so với `cmo` (đối chứng from-scratch).
4. **Debias có cần không?** xem bảng before→after của GLA ở mục 10.

## 5. Cách trình bày
- Một câu chuyện 1 mạch: *"Đuôi 5 ảnh học từ đầu chạm trần (`cmo`). Ta thử mượn 3 loại tri thức
  khác nhau, đo công bằng (GLA), và thấy chúng **bù trừ**: gộp lại tốt hơn từng cái."*
- Hình đắt: biểu đồ `few_shot_accuracy` theo từng expert + cột `fusion` cao nhất; và bảng trọng
  số fusion theo nhóm-shot (cho thấy đuôi "tin" nguồn nào).
- Trung thực về tính mới: từng mảnh từ paper; **đóng góp = nghiên cứu so sánh có hệ thống** ba
  nguồn tri thức trên cùng một đuôi, có debias công bằng và kiểm bù trừ, đối chiếu mốc from-scratch.

## 6. Lưu ý
- Val long-tail **thiếu lớp đuôi** → trọng số fusion nhóm `few` được **buộc theo nhóm `medium`**
  (ghi rõ, không đoán mò). GLA strength có `0` trong lưới nên không bao giờ làm xấu kết quả chọn.
- Mọi expert đọc **cùng** val/test (cùng thứ tự mẫu); notebook có `assert` cho expert `cmo`.
- Nếu hết thời gian/GPU: **tắt `USE_DIFFUSION`** (phần nặng/rủi ro nhất) — câu chuyện vẫn trọn với
  ngôn ngữ vs thị giác-2 + bù trừ.
