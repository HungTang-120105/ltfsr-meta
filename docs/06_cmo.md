# Method 6 — Mixing augmentation & CMO (data-level fix, best from-scratch)

**Code:** `src/datasets/augment.py`, `src/trainers/augment_trainer.py`

## The idea

Methods 2–4 change the loss, the classifier, or the features. Method 6 changes the
**data** the network sees, by *mixing* pairs of images during training. Three variants:

```
mixup  : mixed = λ·imgA + (1-λ)·imgB          ; label = mix(yA, yB) by λ
cutmix : paste a random box of imgB onto imgA ; label = mix(yA, yB) by box area
cmo    : cutmix, but the pasted box comes from a TAIL-RICH stream
```

`mixup`/`cutmix` are generic regularisers. **CMO** (Context-rich Minority Oversampling,
Park et al., CVPR 2022) is the long-tail-specific one and is the method we report:

- A second, **class-balanced sampler** supplies the "paste" images, so rare (tail) objects
  get pasted onto the **diverse backgrounds of frequent (head)** images.
- This synthesises *new tail contexts* from abundant head data — the right way to "augment
  the tail" — instead of heavily distorting the handful of real tail images.
- In this project `cmo` is **CMO + Balanced Softmax** combined (`use_balanced_softmax=True`):
  data-level oversampling *and* a loss-level prior correction, working together.

```
head image (rich background)  +  tail object box  →  new tail-context sample
```

## Why it helps long-tail data

The tail fails because it has too few, too similar examples — the classifier never sees the
object in varied scenes. CMO manufactures exactly that variety: each epoch the tail object
appears on many different head backgrounds, so its decision region widens. Pairing it with
Balanced Softmax stops the head from dominating the gradient. Empirically this is the
**strongest from-scratch method** here (≈0.47 balanced acc, tail ≈0.30 — roughly double the
baseline tail).

## How we report it

Trained and selected like every other method: best checkpoint by **balanced accuracy on the
validation split**, evaluated once on the balanced test set. `mixup` / `cutmix` can be added
to `METHODS` as ablations to show that the *tail-rich* sampling in CMO — not mixing alone — is
what moves the tail.
