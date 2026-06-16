# Long-Tailed Image Recognition: từ huấn luyện-từ-đầu đến thích nghi Foundation Model

**Báo cáo nghiên cứu.** Hai bộ dữ liệu: **CIFAR-100-LT (IF=100)** và **CUB-200-LT (fine-grained, IF=10)**.

---

## 1. Bài toán

Trên dữ liệu **đuôi-dài**, lớp hiếm có rất ít ảnh → mô hình thiên về lớp phổ biến (head), bỏ rơi
lớp hiếm (tail). Câu hỏi nghiên cứu chính (Phase 3): *với lớp đuôi rất ít ảnh, loại tri thức ngoài
nào giúp nhiều nhất — ngôn ngữ (LLM), foundation model thị giác thứ hai (DINOv2), hay sinh dữ liệu
(diffusion) — và chúng có bù trừ nhau không?* Ta trả lời trên **hai** dataset để kiểm tính tổng quát.

## 2. Dữ liệu

### 2.1. CIFAR-100-LT (IF=100)
- 100 lớp ảnh tự nhiên 32×32 RGB (20 siêu lớp × 5 lớp con), gốc 500 train + 100 test/lớp.
- Train lấy mẫu mũ: `count(c) = 500·100^(−c/99)` → head **500**, tail **5** (IF=100). Tổng **10.847** ảnh train.
- Test **cân bằng 100/lớp** (10.000 ảnh) → `accuracy == balanced_accuracy`.
- Nhóm shot (>100 / 20–100 / <20): **many 35 / medium 35 / few 30** lớp.

### 2.2. CUB-200-LT (fine-grained, IF=10)
- 200 loài chim, ảnh kích thước biến thiên (~60 ảnh/lớp tổng). **Fine-grained**: phân biệt loài tinh vi.
- Dùng toàn bộ ảnh: tách **test cân bằng 10/lớp** (2.000 ảnh), phần còn lại lấy mẫu mũ → head **50**,
  tail **5** (IF=10). Train **3.868** ảnh. Test cân bằng → `accuracy == balanced_accuracy`.
- Nhóm shot (>20 / 10–20 / <10): **many 78 / medium 66 / few 56** lớp.
- CUB chỉ ~50 train/lớp nên IF tối đa ~50 (không đạt 100 như CIFAR) → trình bày như miền *fine-grained*.

### 2.3. Giao thức chung
- Tách `VAL_FRACTION=0.1` (phân tầng) để **chọn checkpoint theo balanced-accuracy**; test chỉ báo cáo cuối.
- Mỗi lớp giữ ≥1 ảnh train → lớp tail thường **không có** ảnh trong val (val không che được đuôi).
- **Thước đo chính: `balanced_accuracy` (trung bình recall mỗi lớp) và `few_shot_accuracy`.**

## 3. Phương pháp (3 track)

**Track 1 — huấn luyện từ đầu** (ResNet-18): `baseline` (CE), `balanced_softmax` (cộng log prior),
`decoupling` (cRT: đóng băng encoder, train lại head cân bằng), `supcon` (contrastive + cRT),
`meta` (ProtoNet), `cmo` (CutMix thiên-đuôi + Balanced-Softmax).

**Track 2 — tái dùng checkpoint, không train lại** (Phase 0): `ensemble`(+TTA), `tier_fusion`
(head tham số + prototype theo tầng), `tau_norm` (chuẩn hóa chuẩn trọng số), `vlm_fusion`
(CLIP zero-shot + vision, α chọn trên val).

**Track 3 — thích nghi foundation model đóng băng** (Phase 2/3, trên feature cache):
- `clip_zeroshot` / `clip_llm` (prototype CLIP, bản thường / bản giàu bằng mô tả LLM — CuPL).
- `tip_adapter` / `tip_adapter_f` (truy hồi cache, training-free / fine-tune nhẹ).
- `lift` (adapter <1% tham số + cosine head init từ text, Balanced-Softmax).
- `dino_lift` (DINOv2 + LIFT, init class-mean — không ngôn ngữ).
- `lift_clip_diff` / `lift_clip_mixup` (augment feature đuôi: diffusion / mixup).
- `fusion_tailaware` / `fusion_greedy` (gộp N expert nhận-biết-đuôi; greedy = Caruana ICML'04).
- Ablation: `ft_linear_probe` / `ft_last_block` / `ft_full_ft` (độ-sâu fine-tune CLIP);
  `dino_linear_probe` / `dino_linear_probe_bs` (tách đóng góp loss vs kiến trúc trên DINOv2).
- `cmo` = đối chứng "không tri thức ngoài".

---

## 4. Kết quả — CIFAR-100-LT

### 4.1. Track 1 — huấn luyện từ đầu (`comparison_train_from_scratch.csv`)
| Method | bal-acc | many | medium | few |
|---|---|---|---|---|
| **cmo** | **0.468** | 0.641 | 0.441 | **0.299** |
| balanced_softmax | 0.441 | 0.681 | 0.424 | 0.181 |
| decoupling | 0.407 | 0.706 | 0.375 | 0.094 |
| baseline | 0.401 | 0.707 | 0.371 | 0.079 |
| supcon | 0.367 | 0.668 | 0.361 | 0.021 |
| meta | 0.351 | 0.558 | 0.390 | 0.064 |

Trần from-scratch ~0.47 (tail ~0.30). Các kỹ thuật tail-aware (balanced-softmax, cmo) nâng tail rõ.

### 4.2. Track 2 — tái dùng (`comparison_phase0.csv`)
| Technique | bal-acc | many | few |
|---|---|---|---|
| **vlm_fusion** (CLIP + vision) | **0.681** | 0.810 | 0.571 |
| clip_only | 0.627 | 0.638 | 0.638 |
| tau_norm | 0.543 | 0.720 | 0.364 |
| ensemble (+TTA) | 0.494 | 0.772 | 0.172 |
| tier_fusion | 0.489 | 0.775 | 0.126 |

> Lưu ý: bộ checkpoint dùng cho Phase 0 cho số model-đơn cao hơn leaderboard từ-đầu (xem §7) → coi
> bảng này là minh họa các kỹ thuật reuse, không trộn trực tiếp với 4.1.

### 4.3. Track 3 — foundation model (Phase 2/3) — *kết quả chính*
| Expert | nguồn | bal-acc | many | medium | few |
|---|---|---|---|---|---|
| **fusion_greedy** | gộp | **0.823** | 0.890 | 0.813 | **0.755** |
| fusion_tailaware | gộp | 0.820 | 0.893 | 0.803 | 0.753 |
| **dino_lift** | thị giác-2 | **0.811** | 0.872 | 0.813 | 0.737 |
| lift (CLIP) | vision-language | 0.718 | 0.782 | 0.727 | 0.635 |
| tip_adapter | truy hồi | 0.668 | 0.679 | 0.648 | 0.678 |
| clip_llm | ngôn ngữ | 0.626 | 0.618 | 0.606 | 0.658 |
| clip_zeroshot | — | 0.627 | 0.638 | 0.605 | 0.638 |
| `cmo` (đối chứng) | nội tại | 0.462 | 0.614 | 0.443 | 0.307 |

**Ablation A — đóng góp của ta vs backbone (DINOv2):**
| Mốc | bal-acc | few |
|---|---|---|
| dino_linear_probe (Linear+CE) | 0.463 | **0.020** |
| dino_linear_probe_bs (Linear+Balanced-Softmax) | 0.657 | 0.325 |
| **dino_lift** (+adapter+cosine init) | **0.811** | **0.737** |

**Ablation B — độ sâu fine-tune CLIP:** ft_linear_probe 0.461 → ft_last_block 0.536 → ft_full_ft
0.558 (few **0.111**); tất cả **kém xa** LIFT đóng băng (0.718, few **0.635**) → fine-tune nặng hại đuôi.

**Augment feature:** lift_clip_diff 0.675, lift_clip_mixup 0.698 đều **< lift_clip 0.717** → không giúp.

---

## 5. Kết quả — CUB-200-LT (fine-grained)

### 5.1. Track 3 — foundation model (`comparison.csv` / `knowledge_sources.csv`) — *kết quả chính*
| Expert | nguồn | bal-acc | many | medium | few |
|---|---|---|---|---|---|
| **fusion_greedy** | gộp | **0.849** | 0.858 | 0.824 | **0.864** |
| fusion_tailaware | gộp | 0.846 | 0.859 | 0.824 | 0.854 |
| **dino_lift** | thị giác-2 | **0.841** | 0.851 | 0.812 | 0.859 |
| lift (CLIP) | vision-language | 0.670 | 0.741 | 0.650 | 0.593 |
| tip_adapter_f | truy hồi (FT) | 0.649 | 0.758 | 0.623 | 0.529 |
| tip_adapter | truy hồi | 0.565 | 0.637 | 0.521 | 0.516 |
| clip_llm | ngôn ngữ | 0.484 | 0.518 | 0.458 | 0.466 |
| clip_zeroshot | — | 0.460 | 0.494 | 0.433 | 0.445 |
| `cmo` (đối chứng) | nội tại | 0.195 | — | — | 0.120 |

**Ablation A (DINOv2):** dino_linear_probe (CE) **0.490** (few 0.077) → +Balanced-Softmax **0.744**
(few 0.675) → dino_lift **0.841** (few 0.859). Đóng góp của ta lại rất lớn, đặc biệt ở đuôi.

**Augment feature:** lift_clip_diff 0.655, lift_clip_mixup 0.654 đều **< lift_clip 0.670** → không giúp.

### 5.2. Track 1 — huấn luyện từ đầu trên CUB
`balanced_accuracy` chỉ **~0.20–0.27** (baseline 0.258, cmo 0.195) so với foundation **~0.84** → from-scratch
trên fine-grained ít ảnh gần như vô dụng. (Breakdown many/medium/few của nhóm này **không tin được**,
xem §7.) Đây là bằng chứng mạnh: **tri thức ngoài là thiết yếu**.

---

## 6. Phát hiện chính (nhất quán trên CẢ HAI dataset)

1. **Tri thức ngoài là bước nhảy quyết định.** CIFAR: from-scratch 0.47 → foundation 0.82.
   CUB: 0.20–0.27 → 0.85. Chênh càng lớn ở miền khó (fine-grained CUB).
2. **Nguồn cứu đuôi nhất = foundation model thị giác thứ hai (DINOv2).** dino_lift vượt LIFT-CLIP
   ở cả hai (0.81 vs 0.72; 0.84 vs 0.67) và là trụ cột của fusion. DINOv2 patch-14 + tự giám sát
   bắt chi tiết fine-grained tốt — trên CUB còn mạnh hơn cả CIFAR.
3. **Ngôn ngữ (LLM) giúp ít.** clip_llm ≈ clip_zeroshot (CIFAR 0.626≈0.627; CUB 0.484 vs 0.460 —
   nhích nhẹ). Mô tả giàu không bù được hạn chế của CLIP ViT-B/32 ở ảnh nhỏ / fine-grained.
4. **Sinh dữ liệu & mixup KHÔNG giúp** (cả hai dataset đều thấp hơn LIFT-CLIP) → khi feature đã
   mạnh, augment thêm chỉ thêm nhiễu. (Kết quả âm trung thực.)
5. **Đóng góp của pipeline ta đo được rõ** (Ablation A): chỉ riêng Balanced-Softmax cứu đuôi từ
   ~0.02→0.33 (CIFAR) / 0.08→0.68 (CUB); thêm adapter+cosine init đẩy lên 0.74/0.86.
6. **Fine-tune nặng hại đuôi** (Ablation B, CIFAR): full-FT few 0.111 vs LIFT đóng băng 0.635 →
   biện minh thiết kế đóng băng.
7. **Bù trừ có (nhẹ) và an toàn.** fusion_greedy > expert đơn tốt nhất ở cả hai (0.823>0.811;
   0.849>0.841), không tụt nhóm nào (greedy đảm bảo ≥ best-single trên val mỗi nhóm).

## 7. Ghi chú thực nghiệm (tính nhất quán)

- **CIFAR Phase 0** chạy trên một bộ checkpoint cho số model-đơn cao hơn leaderboard từ-đầu §4.1
  (vd cmo 0.51 vs 0.47), có thể do dùng `pretrained=True`. Bảng §4.2 nên đọc như *minh họa kỹ thuật
  reuse*, không so trực tiếp con số với §4.1/§4.3. Đối chứng from-scratch chuẩn dùng ở §4.1 và §4.3.
- **CUB from-scratch** (§5.2) được chấm khi ngưỡng nhóm-shot còn ở mặc định (100/20) → nhóm `many`
  rỗng, cột many/medium/few **không hợp lệ** cho nhóm này. `balanced_accuracy` vẫn đúng (không phụ
  thuộc cách chia nhóm). Track foundation (§5.1) dùng ngưỡng đúng (20/10).
- Để có bảng "sạch" tuyệt đối: chạy lại Phase 0 trên đúng checkpoint từ-đầu (pretrained=False), và
  chạy lại nhóm from-scratch CUB với ngưỡng 20/10. Không ảnh hưởng kết luận (đều dựa trên balanced-acc).

## 8. Kết luận & hạn chế
**Kết luận:** Với đuôi ít ảnh, chìa khóa không phải mạng lớn hơn từ đầu mà là **mượn + thích nghi nhẹ
foundation model đóng băng với loss nhận-biết-đuôi**. Trong các nguồn tri thức, **DINOv2 (thị giác
thứ hai) giúp nhất**; ngôn ngữ ít, sinh dữ liệu không. Kết luận **nhất quán trên 2 dataset, 2 quy mô
lớp (100/200), 2 miền (vật thể / fine-grained)**.

**Hạn chế:** CLIP dùng ViT-B/32 (nhẹ nhất) → track ngôn ngữ có thể bị đánh giá thấp; val không che
lớp đuôi → tinh chỉnh trên val thiếu tín hiệu đuôi; CUB IF tối đa ~50 (giới hạn dữ liệu); các điểm
ở §7. Hướng tiếp: CLIP lớn hơn, đối chứng from-scratch nhất quán, ước lượng phân phối đuôi để chọn
tham số.

## 9. Tài liệu tham khảo
Ren et al. *Balanced Softmax* (NeurIPS'20) · Kang et al. *Decoupling/cRT, τ-norm* (ICLR'20) ·
Khosla et al. *SupCon* (2020) · Park et al. *CMO* (CVPR'22) · Zhang et al. *Tip-Adapter* (ECCV'22) ·
Shi et al. *LIFT* (ICML'24) · Pratt et al. *CuPL* (ICCV'23) · Oquab et al. *DINOv2* (2023) ·
Han et al. *LDMLR* (2024) · Zhu et al. *Generalized Logit Adjustment* (NeurIPS'23) ·
Caruana et al. *Ensemble Selection* (ICML'04); Wortsman et al. *Model Soups* (ICML'22).

---
*Tái lập: `guides/00–05`, `docs/01–07`, notebook `run_pipeline` (4 phase, toggle) hoặc các notebook
phase riêng. Kết quả thô: `outputs/cifar/*.csv`, `outputs/cub_200_2011/*.csv`.*
