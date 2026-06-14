# Nhận diện ảnh đuôi-dài trên CIFAR-100-LT: từ huấn luyện-từ-đầu đến thích nghi foundation model

**Báo cáo nghiên cứu** — Long-Tailed Image Recognition, CIFAR-100-LT (imbalance factor = 100).

---

## Tóm tắt (Abstract)

Trên dữ liệu **đuôi-dài** (long-tail), lớp hiếm chỉ có vài ảnh khiến mô hình thiên nặng về lớp
phổ biến. Chúng tôi nghiên cứu bài toán này trên **CIFAR-100-LT (IF=100)** theo ba hồi tăng dần:
(1) huấn luyện **từ đầu** với các kỹ thuật nhận-biết-đuôi, (2) **tái dùng** checkpoint không huấn
luyện lại, (3) **thích nghi foundation model đóng băng** (CLIP, DINOv2) cùng tri thức ngoài.

Câu hỏi nghiên cứu trung tâm: *với lớp đuôi chỉ 5 ảnh, loại tri thức ngoài nào giúp nhiều nhất —
ngôn ngữ (LLM), thị giác thứ hai (DINOv2), hay sinh dữ liệu (diffusion) — và chúng có bù trừ nhau
không?* Kết quả chính: huấn luyện từ đầu chạm trần ~0.47 balanced-accuracy (đuôi ~0.30); thích nghi
một foundation model đóng băng nâng lên **0.81–0.82** (đuôi **~0.75**). Trong các nguồn tri thức,
**một backbone thị giác thứ hai (DINOv2) giúp nhiều nhất**, vượt xa ngôn ngữ và sinh dữ liệu; phép
gộp (fusion) nhận-biết-đuôi cho thêm một chút nhờ bù trừ. Một phân tích ablation cho thấy đóng góp
của chúng tôi (loss nhận-biết-đuôi + adapter nhẹ) cứu độ chính xác đuôi từ **0.02 → 0.74**, và rằng
**fine-tune nặng làm hại đuôi** — biện minh cho thiết kế đóng băng.

---

## 1. Bài toán & vì sao khó

**Long-tail recognition.** Trong thực tế, tần suất các lớp thường tuân theo phân phối đuôi-dài: một
ít lớp "head" rất nhiều mẫu, đại đa số lớp "tail" rất ít mẫu. Mô hình huấn luyện theo cách thông
thường sẽ tối ưu tổng thể → **thiên về lớp head**, gần như bỏ rơi lớp tail. Đây là vấn đề cốt lõi của
nhận dạng thực tế (y khoa, loài hiếm, lỗi sản xuất…).

**Cái bẫy đo lường.** `accuracy` thường bị lớp head chi phối → nhìn cao nhưng đuôi chết. Vì vậy cần
các thước đo cân bằng theo lớp (xem §3).

---

## 2. Dữ liệu: CIFAR-100-LT (IF=100)

### 2.1. CIFAR-100 gốc
- **100 lớp** ảnh tự nhiên, gom thành **20 siêu lớp** (mỗi siêu lớp 5 lớp con; ví dụ siêu lớp *trees*
  = maple/oak/palm/pine/willow_tree; *people* = baby/boy/girl/man/woman; *aquatic mammals* =
  beaver/dolphin/otter/seal/whale). Cấu trúc này khiến nhầm lẫn hay xảy ra **trong cùng siêu lớp**.
- **Ảnh 32×32 RGB** — độ phân giải **rất thấp**. Đây là điểm khó đặc thù: chi tiết phân biệt (vân lá,
  khuôn mặt) gần như mất ở 32px, nhiều lớp **trông giống nhau** (5 loài cây, 5 nhóm người). Bài toán
  vì thế là **fine-grained + low-res**, khó hơn nhiều so với phân loại vật thể thô.
- Mỗi lớp gốc có 600 ảnh: **500 train + 100 test**.

### 2.2. Tạo phiên bản đuôi-dài
Tập **train** được lấy mẫu lại theo hàm **mũ giảm** (exponential profile), giữ nguyên tập test:

```
số_ảnh(lớp c) = 500 × IF^(− c/99),   c = 0..99,   IF = 100
```

→ lớp đông nhất **500 ảnh**, lớp hiếm nhất **5 ảnh** (tỉ lệ mất cân bằng **IF = 500/5 = 100**). Số
giảm trơn từ 500 (lớp 0) qua 49 (trung vị) xuống 5 (lớp 99).

| Đại lượng | Giá trị |
|---|---|
| Tổng ảnh train | **10.847** |
| Lớp đông nhất / hiếm nhất | 500 / 5 (IF = 100) |
| Trung bình / trung vị mỗi lớp | 108,5 / 49 |
| **Test** | **10.000 ảnh, CÂN BẰNG 100/lớp** |

### 2.3. Nhóm theo số shot (many / medium / few)
Theo quy ước long-tail, chia 100 lớp thành 3 nhóm để báo cáo riêng:

| Nhóm | Định nghĩa | Số lớp |
|---|---|---|
| **many** (head) | > 100 ảnh train | 35 |
| **medium** | 20–100 ảnh train | 35 |
| **few** (tail) | < 20 ảnh train | 30 |

Nhóm **few** là tâm điểm: ~30 lớp với 5–19 ảnh — quá ít để học một khái niệm thị giác từ đầu.

### 2.4. Chia train / val / test (chống tự lừa mình)
- Tách một phần **validation** theo tỉ lệ `VAL_FRACTION = 0.1` **phân tầng theo lớp** từ train, chỉ
  để **chọn mô hình** (chọn epoch/siêu tham số). **Test KHÔNG bao giờ** dùng để chọn — chỉ báo cáo cuối.
- Mỗi lớp giữ **≥ 1 ảnh train**; lớp tail (5 ảnh) giữ cả 5 → **val có thể không chứa lớp tail**. Đây là
  một thực tế quan trọng: việc tinh chỉnh/chọn trên val **thiếu thông tin về đuôi** (xem các hệ quả ở §6).
- **Chất lượng nhãn:** CIFAR-100 là benchmark chuẩn, nhãn sạch; cái khó đến từ **độ phân giải thấp +
  tính fine-grained + mất cân bằng**, không phải nhiễu nhãn.

---

## 3. Cách đánh giá

Vì test **cân bằng** (100/lớp) nên `accuracy == balanced_accuracy`. Ta báo cáo:

- **`balanced_accuracy`** — trung bình recall theo từng lớp (mỗi lớp trọng số bằng nhau) → không bị
  head chi phối. **Thước đo chính.**
- **`few_shot_accuracy`** — độ chính xác trung bình trên nhóm lớp hiếm. **Thước đo "linh hồn"** của
  long-tail: phương pháp tốt phải kéo được đuôi.
- Phụ trợ: macro-F1, G-mean (trung bình nhân recall — bằng ~0 nếu một lớp recall=0), và phân tách
  many/medium/few.
- **Quy tắc chọn mô hình:** giữ checkpoint tốt nhất theo **balanced-accuracy trên val** (không phải
  accuracy thô, vốn thiên head).

---

## 4. Phương pháp

Chúng tôi tổ chức theo **ba hồi**, mỗi hồi rẻ hơn nhưng cải thiện đuôi nhiều hơn hồi trước. Mọi mô
hình dùng chung encoder ResNet-18 (track từ đầu) hoặc foundation model đóng băng (track ngoài) để so
sánh công bằng.

### Hồi 1 — Huấn luyện từ đầu (track "nội tại")
Câu hỏi: *can thiệp ở tầng nào thì giúp đuôi?*

| Method | Can thiệp ở | Ý tưởng (giải thích dễ hiểu) | Vì sao giúp đuôi |
|---|---|---|---|
| **baseline** | — | ResNet-18 + cross-entropy thường | Mốc tham chiếu; thiên head nặng |
| **balanced_softmax** | loss | Cộng `log(tần suất lớp)` vào logits khi train | Bù lại giả định "lớp đều nhau" của softmax → bớt thiên head |
| **decoupling (cRT)** | bộ phân loại | Train cả mạng → **đóng băng encoder, train lại head** trên sampler cân bằng | Feature đã tốt; chỉ *ranh giới quyết định* bị lệch → sửa riêng nó |
| **supcon** | biểu diễn | Học đặc trưng bằng Supervised Contrastive rồi cRT | Đặc trưng tách lớp tốt hơn trước khi phân loại |
| **meta** | bonus | ProtoNet học theo "episode" few-shot | Học để học từ ít mẫu (mạnh ở trục few-way) |
| **cmo** | dữ liệu | **CutMix thiên-đuôi**: dán vật thể lớp hiếm lên nền lớp phổ biến + balanced-softmax | Tạo *ngữ cảnh mới* cho đuôi từ dữ liệu dồi dào → đuôi đa dạng hơn |

*Thông điệp Hồi 1:* sửa loss / đầu phân loại rẻ mà đuôi tăng 2–3 lần; **`cmo` mạnh nhất** nhóm này.

### Hồi 2 — Tái dùng checkpoint, không huấn luyện lại ("free lunch")
- **Ensemble (+TTA):** trung bình xác suất nhiều mô hình (và ảnh lật) → lỗi triệt tiêu, ổn định hơn.
- **Tier-aware fusion:** lớp head dùng *head tuyến tính*, lớp tail dùng *prototype* (NCM); trộn theo
  tầng — vì prototype data-hiệu-quả hơn cho đuôi.
- **τ-normalization:** chuẩn trọng số `‖w_c‖` phình theo tần suất lớp → chia bớt để **gỡ thiên head**,
  không cần train.

### Hồi 3 — Thích nghi foundation model đóng băng (track "tri thức ngoài")
Cốt lõi: 5 ảnh **không đủ học từ đầu** → **mượn** một mô hình đã pretrain quy mô lớn, **đóng băng**
backbone, chỉ học vài tham số nhỏ trên **feature đã cache**.

- **CLIP zero-shot:** nhận diện qua *tên lớp* ("a photo of a {class}"); không thấy dữ liệu lệch của ta.
- **Tip-Adapter / Tip-Adapter-F:** phân loại bằng **truy hồi** (so khớp với bộ nhớ feature train),
  trộn với zero-shot; bản "-F" tinh chỉnh nhẹ khóa cache.
- **LIFT** (ý tưởng "fine-tune nặng hại đuôi", ICML 2024): adapter residual nhỏ (**<1% tham số**,
  cổng khởi tạo 0 → bắt đầu *đúng bằng* zero-shot) + cosine head **khởi tạo từ feature chữ** (semantic
  init), huấn luyện bằng **balanced-softmax**. Đây là track mạnh nhất.

**Ablation độ-sâu fine-tuning** (`linear_probe` → `last_block` → `full_ft`): train *trong* backbone
ViT trên ảnh, để kiểm chứng "fine-tune nặng hại đuôi" (xem §5.2).

### Phase 3 — Nghiên cứu chính: nguồn tri thức nào cứu đuôi?
Mỗi "expert" là một mô hình tự cho ra dự đoán, mang một **nguồn tri thức** khác nhau:

| Expert | Nguồn tri thức | Cách làm | Bài tham khảo |
|---|---|---|---|
| `cmo` | **nội tại** (đối chứng) | mô hình từ-đầu tốt nhất | — |
| `clip_llm` | **ngôn ngữ** | LLM (Qwen2.5) sinh mô tả lớp → prototype CLIP giàu hơn | CuPL (ICCV'23) |
| `dino_lift` | **thị giác thứ 2** | DINOv2 (tự giám sát, không ngôn ngữ) + LIFT, init class-mean | DINOv2 (2023) |
| `lift_clip_diff` | **sinh dữ liệu** | diffusion sinh feature đuôi → train trên real+synthetic | LDMLR (2024) |
| `lift_clip_mixup` | **augment (cmo-style)** | tail-aware feature mixup → train LIFT | (cmo ở feature space) |
| **GLA** | (debias) | gỡ bias pretraining của *chính* CLIP — tổng quát hóa balanced-softmax | GLA (NeurIPS'23) |
| `fusion_*` | (gộp) | trộn nhận-biết-đuôi theo nhóm-shot; `fusion_greedy` chọn tham lam an toàn | Caruana ICML'04 |

*Vì sao tái dùng phương pháp cũ:* `balanced_softmax` đã là **loss huấn luyện** của mọi expert LIFT;
GLA tổng quát hóa nó; `cmo` chuyển vào feature space thành `lift_clip_mixup`; `fusion_*` tổng quát hóa
ensemble/tier/vlm-fusion của Hồi 2. Không phương pháp nào bị bỏ phí — chúng thành **đối chứng và thành
phần** của nghiên cứu chính.

---

## 5. Kết quả

> Số dưới đây là kết quả thực nghiệm trên CIFAR-100-LT (IF=100). **Thước đo chính: balanced-accuracy**;
> cột `few` là độ chính xác nhóm đuôi.

### 5.1. Bảng tổng hợp ba track

**Track 1 — huấn luyện từ đầu**

| Method | bal-acc | many | medium | few |
|---|---|---|---|---|
| **cmo** | **0.468** | 0.641 | 0.441 | **0.299** |
| balanced_softmax | 0.441 | 0.681 | 0.424 | 0.181 |
| decoupling | 0.407 | 0.706 | 0.375 | 0.094 |
| baseline | 0.401 | 0.707 | 0.371 | 0.079 |
| supcon | 0.367 | 0.668 | 0.361 | 0.021 |
| meta | 0.351 | 0.558 | 0.390 | 0.064 |

**Track 3 — thích nghi foundation model (VLM, CLIP ViT-B/32)**

| Method | bal-acc | many | medium | few |
|---|---|---|---|---|
| **LIFT (CLIP)** | **0.719** | 0.782 | 0.727 | 0.635 |
| tip_adapter | 0.668 | 0.679 | 0.648 | 0.678 |
| tip_adapter_f | 0.643 | 0.821 | 0.672 | 0.400 |
| clip zero-shot | 0.627 | 0.638 | 0.605 | 0.638 |
| *ft_full_ft* (fine-tune toàn bộ ViT) | 0.558 | 0.874 | 0.624 | **0.111** |
| *ft_last_block* | 0.536 | 0.850 | 0.570 | 0.128 |
| *ft_linear_probe* | 0.461 | 0.785 | 0.459 | 0.086 |

**Phase 3 — nghiên cứu nguồn tri thức (DINOv2 ViT-S/14 + CLIP)**

| Expert | bal-acc | many | medium | few |
|---|---|---|---|---|
| **fusion_greedy** | **0.823** | 0.890 | 0.813 | **0.755** |
| fusion_tailaware | 0.820 | 0.893 | 0.803 | 0.753 |
| **dino_lift** | **0.811** | 0.872 | 0.813 | 0.737 |
| lift_clip | 0.717 | 0.771 | 0.708 | 0.664 |
| lift_clip_mixup | 0.698 | 0.754 | 0.709 | 0.619 |
| lift_clip_diff | 0.675 | 0.807 | 0.689 | 0.505 |
| clip_llm | 0.626 | 0.618 | 0.606 | 0.658 |
| clip zero-shot | 0.627 | 0.638 | 0.605 | 0.638 |
| `cmo` (đối chứng) | 0.462 | 0.614 | 0.443 | 0.307 |

### 5.2. Ablation A — đóng góp của chúng tôi vs backbone (DINOv2)

Để tách "feature DINOv2 mạnh sẵn" khỏi "phần chúng tôi thêm", dùng cùng feature DINOv2 đóng băng:

| Mốc | Train | bal-acc | many | medium | **few** |
|---|---|---|---|---|---|
| `dino_linear_probe` | Linear + **CE thường** | 0.463 | 0.884 | 0.421 | **0.020** |
| `dino_linear_probe_bs` | Linear + **Balanced Softmax** | 0.657 | 0.882 | 0.715 | **0.325** |
| `dino_lift` | + adapter + cosine + class-mean init | **0.811** | 0.872 | 0.813 | **0.737** |

**Đọc ra:** feature DINOv2 rất mạnh ở head (many=0.88 chỉ với linear+CE), nhưng huấn luyện ngây thơ
**giết đuôi** (few=0.02, g-mean≈0). Đóng góp của chúng tôi tách thành hai bước **đều quan trọng**:
- **+ Balanced Softmax**: bal +0.19, đuôi **0.02 → 0.33** (loss nhận-biết-đuôi cứu khỏi sụp).
- **+ adapter + cosine/class-mean init**: bal +0.15, đuôi **0.33 → 0.74**.

Tức **toàn bộ pipeline của chúng tôi thêm +0.35 balanced-acc** và nâng đuôi **0.02 → 0.74** (×37) so
với backbone trần. Đáng chú ý: chúng tôi *hy sinh chút head* (0.884 → 0.872) để cứu cả đuôi — đúng đánh
đổi head↔tail mà phương pháp tail-aware nhắm tới.

### 5.3. Ablation B — fine-tune nặng có hại đuôi không?

| Cách thích nghi | bal-acc | few |
|---|---|---|
| ft_linear_probe (chỉ head) | 0.461 | 0.086 |
| ft_last_block | 0.536 | 0.128 |
| ft_full_ft (toàn bộ ViT) | 0.558 | 0.111 |
| **LIFT (đóng băng + adapter)** | **0.719** | **0.635** |

Mọi mức fine-tune *trong* backbone đều **kém xa** đóng băng + adapter, và **đuôi sụp** (~0.1 so với
0.64). → Khẳng định bằng số: với foundation model mạnh, **fine-tune nặng là sai công cụ cho đuôi**;
đóng băng + thích nghi nhẹ thắng. Đây là biện minh trực tiếp cho thiết kế đóng băng của chúng tôi.

### 5.4. Phép kiểm bù trừ (complementarity)
`fusion_greedy` (0.823) > expert đơn tốt nhất `dino_lift` (0.811) → các nguồn **bù trừ nhẹ**. Greedy
ensemble selection (Caruana ICML'04) đảm bảo **không tụt dưới expert đơn tốt nhất trên val ở mỗi nhóm**
— so với bản trộn theo lưới (`fusion_tailaware`) bị tụt ở medium (0.803 < 0.813), greedy giữ medium
= 0.813. Tuy nhiên mức bù trừ nhỏ vì **DINOv2 gánh gần hết**.

---

## 6. Phân tích & phát hiện chính

1. **Tri thức ngoài là bước nhảy thật, không phải kỹ thuật cân bằng:** từ-đầu chạm trần ~0.47 (đuôi
   0.30); foundation model đóng băng đạt **0.81–0.82** (đuôi ~0.75). Khoảng cách **+0.35**.
2. **Nguồn tri thức cứu đuôi nhất = một backbone thị giác thứ hai (DINOv2)**, vượt xa ngôn ngữ và sinh
   dữ liệu. So `few_shot`: **DINOv2 0.737** ≫ clip_llm 0.658 ≈ zero-shot 0.638 > diffusion 0.505. Lý
   do: DINOv2 tự giám sát cho feature tách lớp rất tốt, lại dùng patch 14 (mịn hơn CLIP B/32 patch 32)
   → hợp ảnh nhỏ. Phù hợp các báo cáo gần đây (DINOv2 vượt CLIP-Adapter trên CIFAR-100-LT).
3. **Loss nhận-biết-đuôi là yếu tố quyết định** (Ablation A): chỉ đổi CE → Balanced Softmax đã nâng
   đuôi 0.02 → 0.33; backbone mạnh đến đâu cũng vô dụng ở đuôi nếu huấn luyện ngây thơ.
4. **Fine-tune nặng hại đuôi** (Ablation B): đóng băng + adapter (đuôi 0.64) ≫ full fine-tune (đuôi 0.11).
5. **Bù trừ có nhưng yếu**, và greedy fusion đảm bảo không hồi quy ở từng nhóm — quan trọng khi một
   expert áp đảo.
6. **Kết quả âm (trung thực):**
   - *Ngôn ngữ (LLM) gần như không giúp*: `clip_llm` 0.626 ≈ zero-shot 0.627 (chỉ nhích đuôi 0.658
     vs 0.638). Mô tả giàu hơn không bù được hạn chế của CLIP ViT-B/32 ở ảnh 32px.
   - *Sinh dữ liệu & mixup không giúp, còn hơi hại*: `lift_clip_diff` 0.675 và `lift_clip_mixup` 0.698
     đều **dưới** `lift_clip` 0.717 — khi feature đã mạnh, augment thêm chỉ thêm nhiễu.
   - *Tinh chỉnh cache của Tip-Adapter hại đuôi*: `tip_adapter` (đuôi 0.678) > `tip_adapter_f` (0.400).

---

## 7. Kết luận

Trên CIFAR-100-LT (IF=100), **chìa khóa cho đuôi 5-ảnh không phải huấn luyện một mạng lớn hơn từ đầu,
mà là mượn và thích nghi nhẹ một foundation model đóng băng với một loss nhận-biết-đuôi.** Trong các
nguồn tri thức ngoài, **một backbone thị giác thứ hai (DINOv2) hữu ích nhất**; ngôn ngữ và sinh dữ liệu
đóng góp ít hoặc không. Phép gộp nhận-biết-đuôi cho thêm chút nhờ bù trừ. Đóng góp phương pháp của
chúng tôi — adapter nhẹ + balanced-softmax + chọn-trên-val — nâng độ chính xác đuôi từ **0.02 lên
0.74** so với một linear probe ngây thơ, và bằng số chứng minh **fine-tune nặng làm hại đuôi**.

## 8. Hạn chế & hướng tương lai
- **CLIP dùng ViT-B/32** (bản nhẹ nhất) → track ngôn ngữ có thể bị đánh giá thấp; ViT-L/14 có thể đổi
  kết luận về "ngôn ngữ ít giúp".
- **Val thiếu lớp đuôi** → mọi việc chọn/tinh chỉnh trên val (α của fusion, strength của GLA) đều
  thiếu tín hiệu về đuôi; nhóm `few` của fusion buộc theo `medium`.
- **Đối chứng `cmo` là từ-đầu thuần** (không ImageNet) — so sánh công bằng giữa "không tri thức ngoài"
  và "có tri thức ngoài"; nếu dùng cmo pretrained-ImageNet thì phải gọi tên khác.
- **Đóng góp gộp:** Ablation A tách được loss vs kiến trúc, nhưng chưa tách riêng adapter và cosine
  head — có thể ablation sâu hơn.
- Hướng tiếp: CLIP lớn hơn; ước lượng phân phối đuôi để chọn tham số tốt hơn; thử test-time adaptation.

## 9. Tài liệu tham khảo (chọn lọc)
- Ren et al., *Balanced Meta-Softmax* (NeurIPS 2020).
- Kang et al., *Decoupling Representation and Classifier* (ICLR 2020) — τ-norm, cRT.
- Khosla et al., *Supervised Contrastive Learning* (2020).
- Park et al., *CMO: Context-rich Minority Oversampling* (CVPR 2022).
- Zhang et al., *Tip-Adapter* (ECCV 2022).
- Shi et al., *LIFT: Long-tail Learning with Foundation Models — Heavy Fine-tuning Hurts* (ICML 2024).
- Pratt et al., *CuPL: Customized Prompts via Language Models* (ICCV 2023).
- Oquab et al., *DINOv2* (2023).
- Han et al., *LDMLR: Latent-based Diffusion for Long-tailed Recognition* (2024).
- Zhu et al., *Generalized Logit Adjustment* (NeurIPS 2023).
- Caruana et al., *Ensemble Selection from Libraries of Models* (ICML 2004); Wortsman et al., *Model
  Soups* (ICML 2022).

---

*Chi tiết tái lập: xem `guides/00–04`, `docs/01–07`, và các notebook `run_all_methods`, `phase0_reuse`,
`phase2_clip_adapt`, `phase3_knowledge_sources`. Kết quả thô: `outputs/comparison_*.csv`.*
