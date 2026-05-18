# HACA — Hazard-Anchored Cross-Attention
## Design document — 2026-05-17

---

## 1. Motivation (one paragraph)

MCAT's co-attention layer learns `A[pathway_i, patch_j] = softmax(Q·K/√d)`
purely from gradient flow back through 4+ transformer/FC layers from the
final NLL-Survival loss. Attention has **no direct supervision** that
patch j is or isn't survival-informative. HACA introduces a small
per-patch hazard predictor `r_j ∈ (0,1)` whose log is added as a bias
into the co-attention pre-softmax logits, and (optionally) trains it
jointly with an auxiliary NLL-Surv loss applied to a mean-pooled patch
representation. The hypothesis: explicit risk anchoring at the attention
level should drive attention mass toward survival-informative patches
*earlier* in training, improving the final c-Index.

---

## 2. Architecture — exact equations

Let `N` = number of WSI patches (varies per slide, e.g. 38 - 36,966).
Let `E = 256` = MCAT's embedding dimension.
Let `K = 6` = number of gene-signature pathway groups.

### 2.1 Inputs (identical to MCAT)

```
x_path  : (N, 1024)       ResNet-50 truncated features
x_omic  : 6 × (P_k,)      per-pathway gene vectors
```

### 2.2 Feature projection (identical to MCAT)

```
h_path_bag = wsi_net(x_path).unsqueeze(1)              shape (N, 1, E)
h_omic_k   = sig_networks[k](x_omic[k])                shape (E,)
h_omic_bag = stack(h_omic).unsqueeze(1)                shape (K=6, 1, E)
```

### 2.3 Per-patch hazard head — NEW

```
patch_logit_j = patch_hazard_mlp(h_path_bag[j])        shape (N, 1)
r_j           = sigmoid(patch_logit_j)                 in (0,1), per patch
log_r         = log(r_j + eps)                         eps=1e-3 for stability
```

`patch_hazard_mlp` is a small 2-layer MLP:
```
Linear(256, 64) → ReLU → Dropout(0.25) → Linear(64, 1)
```
~16K parameters. Negligible vs MCAT's 4.5M.

### 2.4 Hazard-anchored co-attention — MODIFIED

Build a `(K, N)` attention bias broadcast across queries:
```
attn_bias[k, j] = λ · log_r[j]               for all k       shape (K, N)
```

Pass to nn.MultiheadAttention via its `attn_mask` parameter (which is
added to pre-softmax attention logits — exactly the integration point
we need, native API, no MultiheadAttention rewrite required):

```
h_path_coattn, A_coattn = coattn(
    query  = h_omic_bag,                              (K, 1, E)
    key    = h_path_bag,                              (N, 1, E)
    value  = h_path_bag,                              (N, 1, E)
    attn_mask = attn_bias                             (K, N) — NEW
)
```

After softmax this is mathematically equivalent to:
```
A_HACA[k,j] = softmax_j( Q[k]·K[j]/√d  +  λ·log(r_j) )
            = softmax_j( Q[k]·K[j]/√d ) · r_j^λ  (re-normalised)
```
i.e. patches with high `r_j` get attention multiplied by `r_j^λ` before
re-normalization. λ controls strength: λ=0 → vanilla MCAT, λ→∞ →
attention collapses onto highest-r_j patch.

### 2.5 Downstream — identical to MCAT

```
h_path_trans = path_transformer(h_path_coattn)
... path attention head, omic transformer, omic attention head, fusion ...
logits = classifier(h)                                  shape (1, 4)
hazards = sigmoid(logits)                               main NLL-Surv head
S = cumprod(1 - hazards)
```

### 2.6 Auxiliary risk head — NEW (optional, gated by α)

To give `r_j` an explicit supervision signal (rather than relying purely
on gradient flow through the attention bias), pool `h_path_bag` and
predict slide-level hazards:

```
h_aux = mean_pool(h_path_bag, weights = r_j_softmax_over_N)   shape (E,)
aux_logits  = aux_classifier(h_aux)                            shape (1, 4)
aux_hazards = sigmoid(aux_logits)
aux_S       = cumprod(1 - aux_hazards)
loss_aux    = NLL_Surv(aux_hazards, aux_S, Y, c)
```

`aux_classifier = nn.Linear(256, 4)` — another ~1K params.

### 2.7 Total training loss

```
loss = NLL_Surv(hazards, S, Y, c)  +  α · loss_aux
                                       └── α=0 disables, α=0.1 our default
```

Hyperparameters introduced:
- `haca_lambda` (λ) — attention bias strength.  Default 1.0.  Sweep {0.5, 1.0, 2.0}.
- `haca_aux_weight` (α) — auxiliary loss weight.  Default 0.1.  Sweep {0.0, 0.1, 0.5}.
- `haca_eps` — log stabiliser.  Fixed at 1e-3.

---

## 3. Files to create / modify

### NEW FILES

| Path | Purpose | Approx LOC |
|---|---|---|
| `MCAT/models/model_haca.py` | `HACA_Surv(nn.Module)` — copies MCAT_Surv, adds §2.3-2.6 | ~150 |
| `MCAT/utils/haca_train_utils.py` | `train_loop_survival_haca` / `validate_survival_haca` / `summary_survival_haca` — copies coattn_train_utils with the multi-output unpacking and the aux loss term | ~250 |
| `launch_haca_blca.sh` | 4-GPU sharded launcher mirroring `launch_mcat_blca.sh` | ~30 |

### MODIFIED FILES

| Path | Change | LOC delta |
|---|---|---|
| `MCAT/main.py` | Add `'haca'` to `--model_type` choices; add `--haca_lambda`, `--haca_aux_weight` argparse | +6 |
| `MCAT/utils/core_utils.py` | In `train()`, dispatch to HACA loops + model when `args.model_type == 'haca'`; instantiate `HACA_Surv` | +12 |

### UNCHANGED (intentionally)

- `datasets/*` — no data-side changes; HACA uses the same `coattn` collate
- `splits/*` — same 5-fold splits as MCAT
- Loss function `NLLSurvLoss` — same
- Optimizer, LR, gradient accumulation, epochs — same

This is **strictly additive** to MCAT. Setting `--model_type mcat` runs
vanilla MCAT identically; `--model_type haca` runs HACA. No vanilla code
path is altered.

---

## 4. Phased experimental plan

### Phase 1 — Implementation + smoke test (~4 h)
- Write `model_haca.py` and `haca_train_utils.py`
- Wire into `main.py` and `core_utils.py`
- 1-epoch run on fold 0 of BLCA, sanity-check: model trains, no NaNs,
  per-patch r_j distribution is non-degenerate (not all ~0.5).

### Phase 2 — BLCA pilot (~15 min training)
- Full 5-fold BLCA with λ=1.0, α=0.1
- **Decision point**: compare to locked baseline 0.632 ± 0.042
  - Δ ≥ +0.005  → proceed to Phase 3
  - Δ ∈ [-0.005, +0.005]  → run ablations to understand neutrality, then decide
  - Δ < -0.005  → kill, document null result

### Phase 3 — Full evaluation (~5 h training)
- BLCA + GBMLGG, 3 seeds each (seeds 0, 1, 2), λ=1.0, α=0.1
- Paired t-test vs MCAT baseline at same seed
- Report mean Δ + 95% CI

### Phase 4 — Ablations (~10 h training, BLCA only)
- (λ, α) = (0, 0)     → pure architecture only, no anchoring (control)
- (λ, α) = (1, 0)     → attention bias only, no aux loss
- (λ, α) = (0, 0.1)   → aux loss only, no attention bias
- (λ, α) = (1, 0.1)   → full HACA (main)
- (λ, α) = (2, 0.1)   → strong anchoring
- (λ, α) = (1, 0.5)   → heavy aux
- α schedule: warmup α from 0 → 0.1 over first 5 epochs

### Phase 5 — Report (~1 h)
- Paper-style `.txt` document mirroring `mcat_blca_replication_results.txt`
- Update CROSS_COHORT_SUMMARY with HACA columns

---

## 5. Honest risk assessment

| Risk | Probability | Mitigation |
|---|---|---|
| `r_j` collapses to constant (everywhere ~0.5) — no attention modulation | medium | Detect via diagnostic histogram in Phase 1; if degenerate, raise α |
| `r_j` collapses to 0 or 1 — softmax-after-bias becomes degenerate | low | log(r+ε) bounded, λ=1.0 is mild; ε prevents log(0) |
| Gain on BLCA only, null on GBMLGG | medium | Saturation hypothesis predicts this; report as ablation if so |
| Gradient instability from log term | low | Detected by NaN-check in Phase 1; clip if needed |
| Aux loss conflicts with main loss | medium | Default α=0.1 small; sweep down to 0.05 if oscillation |
| Even when r_j is learned correctly, no gain (InterSHAP says interaction ~4%) | high | Pre-registered null hypothesis; null result still contributes to saturation paper |

**My P(meaningful gain ≥ +0.012 averaged over both cohorts): ~25%**
**My P(any direction matters for the paper): 100%** — null or positive
both publishable in current framing.

---

## 6. What HACA is NOT

To avoid scope creep, HACA explicitly does NOT:
- Modify the WSI feature extractor (no UNI / no foundation-model swap)
- Change gene signatures or genomics SNN
- Change the splits, loss, optimizer, or epoch count
- Add per-pathway-specific attention biases (single shared `attn_bias`
  across all 6 pathway queries — could revisit in a v2)
- Use any external survival data or pre-training

Single-axis intervention by design, to keep paired comparisons clean.

---
END
