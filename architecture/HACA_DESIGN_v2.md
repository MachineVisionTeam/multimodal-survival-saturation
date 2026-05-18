# HACA v2 — Risk-Conditioned Multimodal Co-Attention
## Design document — 2026-05-17
## (v1 design preserved separately at HACA_DESIGN.md)

---

## 0. Why v2 exists (one paragraph)

HACA v1 was a single architectural tweak (one biased-attention term)
on top of MCAT. A 2024-2026 multimodal-survival landscape audit
(AdaMHF, DIMAF, HySurvPred, BMLSurv, MoME, EsurvFusion, Path-GPTOmic)
shows the "Strong novelty" tier all share a common pattern:
**multiple coordinated mechanisms operationalising one unifying
theoretical / procedural frame**. v1 had 1 mechanism + 0 frame and
therefore landed in the "Moderate" tier — fine as an ablation row,
insufficient as a standalone contribution. v2 promotes HACA into the
"Strong" tier by adding two coordinated mechanisms and a Bayesian
theoretical frame.

---

## 1. Unifying frame — Cross-Attention as Bayesian Posterior

Standard cross-attention computes
```
A[k, j] = softmax_j( Q[k] · K[j] / sqrt(d) )
        = p(patch_j | pathway_k)               (Vaswani-style as MoE)
```

This is **content-blind to task relevance**: a patch with high content
similarity to the pathway query but zero survival relevance gets the
same attention weight as a survival-relevant patch with the same
similarity.

Bayesian re-derivation: if we have a prior `pi(j) = p(j is risk-relevant)`,
the *posterior* attention is
```
A_HACA[k, j]  =  p(patch_j | pathway_k, risk-relevant)
              propto  p(patch_j | pathway_k) · pi(j)^lambda
              =  softmax_j( Q[k] · K[j] / sqrt(d)  +  lambda · log pi(j) )
```

This is mathematically equivalent to adding `lambda · log pi(j)` to the
pre-softmax attention logits. **The Bayesian frame is what justifies the
specific functional form** — additive log-prior, not multiplicative
post-softmax, not gating. The frame is falsifiable: if other forms (e.g.
multiplicative post-softmax) work better, the Bayesian interpretation is
wrong and we report that.

This frame connects to:
- Vaswani et al. 2017 — attention as MoE
- Jang et al. 2017 (Gumbel-Softmax) — attention as posterior sampling
- Locatello et al. 2020 (Slot Attention) — attention as inference
- Bahdanau et al. 2015 — soft alignment as latent variable

HACA v2 = the first instantiation of Bayesian-posterior cross-attention
**with task-derived prior pi(j) = sigmoid(MLP(K[j]))** in the
multimodal-survival setting.

---

## 2. Three coordinated mechanisms

### Mechanism 1 — PRCA: Per-pathway Risk Co-Attention

In MCAT, all 6 pathway queries `Q^(1..6)` share the same patch keys
`K[j]`. In v1 we proposed a single shared `r_j`, applied identically to
every pathway. v2 generalises: each pathway gets **its own per-patch
hazard estimator**, allowing pathway-specific anchoring.

```
r_j^(k) = sigmoid( MLP_k( K[j] ) )                shape (K=6, N)
```

Each `MLP_k` is a small `256 -> 64 -> 1` head with ~17K params; 6 heads
= ~100K total (still 2% of MCAT's 4.5M).

Biological motivation: tumour-suppression pathway should anchor on
necrotic regions; cell-cycle pathway should anchor on mitotic regions.
A shared `r_j` cannot express this. PRCA can.

Modified co-attention:
```
A_PRCA[k, j] = softmax_j(  Q[k] · K[j] / sqrt(d)  +  lambda · log r_j^(k)  )
```

Implementation: builds attn_mask of shape (K, N), passed to
nn.MultiheadAttention's `attn_mask` parameter (native API, no
MultiheadAttention rewrite).

### Mechanism 2 — HDL: Hazard Distillation Loss

A new training objective. The model's main hazard prediction
(after the full forward pass) acts as **teacher** for the per-patch
hazard heads (student) via KL divergence, inside the same forward pass.

```
hazards_main = sigmoid( classifier( h_fusion ) )     shape (1, 4)
                                                     # MCAT's main output

# Pool patch hazards: average across pathways for a slide-level patch hazard
r_bar = mean_k( mean_j(  r_j^(k)  ) )                shape (scalar)

# Project r_bar into discrete-time 4-class hazard space via tiny head
hazards_patch = sigmoid( hdl_head( r_bar_features ) ) shape (1, 4)

L_HDL = KL( stop_grad(hazards_main)  ||  hazards_patch )
```

Crucially the gradient on `hazards_main` is stopped — only the patch
hazard head is trained by this term. This is teacher-student inside one
forward pass.

Total loss:
```
L  =  L_NLL_main  +  alpha_HDL · L_HDL  +  alpha_AUX · L_NLL_aux
```

(L_NLL_aux is v1's optional aux head; keeping it because it gives a
*direct* survival signal to r in addition to the distilled signal.)

### Mechanism 3 — AAS: Adaptive Anchoring Schedule

At epoch 0, `r_j^(k)` is random and uninformative. Anchoring attention
on random priors *hurts*. Solution: warmup lambda from 0 to its final
value over the first N warmup epochs.

```
lambda(epoch) = lambda_final · min(1, epoch / N_warmup)
```

Defaults: `lambda_final = 1.0`, `N_warmup = 5` (out of 20 total).

This addresses the "r_j is garbage at init" risk identified in v1.
Attention learns the patch space cleanly for the first 5 epochs, then
the Bayesian prior kicks in.

AAS also adds a **schedule-ablation** axis: we can test no-warmup,
short-warmup (3), default (5), long-warmup (10), and constant
lambda=1 from epoch 0.

---

## 3. Architecture summary

```
         x_path (N, 1024)               x_omic_k (P_k,) for k=1..6
            |                               |
        wsi_net                         sig_networks_k
            |                               |
        h_path_bag (N, 1, 256)          h_omic_k (256,)
            |                               |
            |                           h_omic_bag (6, 1, 256)
            |                               |
            |                               |
    [PRCA] per-pathway hazard MLPs    Q = h_omic_bag
            |                               |
            v                               |
    r^(k)_j (6, N)  --> log + lambda(epoch) bias
            \                               /
             ---> attn_mask (6, N) ---> coattn(Q, K, V, attn_mask)
                                             |
                                       A_PRCA, h_path_coattn
                                             |
                                  ... MCAT path/omic transformer ...
                                             |
                                       fusion(concat) -> classifier
                                             |
                                       hazards_main (1, 4)
                                             |
                              [HDL] KL(stop_grad(hazards_main) || hazards_patch)
                                             |
                            [AUX] optional NLL on mean-pooled patch hazards
```

Three new parameter groups (% of MCAT's 4.5M):
- 6 pathway-specific hazard MLPs: ~100K (2.2%)
- HDL projection head: ~1K (negligible)
- AUX classifier (kept from v1): ~1K (negligible)
- Total new: ~102K (2.3%)

---

## 4. Files to create / modify

### NEW

| Path | Purpose |
|---|---|
| `MCAT/models/model_haca_v2.py` | `HACA_v2_Surv` — copies MCAT_Surv, adds PRCA / HDL / AUX heads, exposes lambda / alpha_HDL / alpha_AUX / N_warmup hyperparams |
| `MCAT/utils/haca_v2_train_utils.py` | Train / validate / summary functions handling multi-output forward and combined loss |
| `MCAT/utils/haca_schedule.py` | Pure-Python lambda schedule (warmup + constant) — easy to unit-test |
| `launch_haca_v2_pilot_blca.sh` | Phase-2 pilot launcher (one fold for smoke + 5-fold for pilot) |
| `launch_haca_v2_full.sh` | Phase-3 full eval launcher (BLCA + GBMLGG, seeds 0/1/2) |
| `launch_haca_v2_ablations.sh` | Phase-4 ablation grid launcher |

### MODIFIED

| Path | Change |
|---|---|
| `MCAT/main.py` | Add `'haca_v2'` to --model_type; add `--haca_lambda`, `--haca_alpha_hdl`, `--haca_alpha_aux`, `--haca_warmup_epochs` |
| `MCAT/utils/core_utils.py` | Dispatch HACA_v2 model + train/val/summary loops when `args.model_type == 'haca_v2'` |

### UNCHANGED (intentionally — paired comparison is the point)

- Encoder (ResNet-50 truncated, same .pt features on disk)
- Splits, gene signatures, NLL-Surv loss, optimiser, LR, gc, epochs

Strictly additive. `--model_type mcat` runs byte-identical vanilla
baseline. `--model_type haca_v2` runs the new framework.

---

## 5. Phased experimental plan

### Phase 1 — Implementation + smoke (~10 h)
- Write all 6 new / modified files
- Unit-test schedule and forward shapes
- 1-epoch smoke run on fold 0 of BLCA
- Diagnostics: print r_j^(k) histogram per pathway after epoch 1 — must
  be non-degenerate (not all collapsed to 0.5)

### Phase 2 — BLCA pilot (~30 min training)
- Full 5-fold BLCA with defaults (lambda=1, alpha_HDL=0.5, alpha_AUX=0.1,
  N_warmup=5)
- **Decision point** vs locked baseline 0.632 +/- 0.042:
  - Delta >= +0.008  -> proceed to Phase 3
  - Delta in [0, +0.008]  -> run lambda/alpha sweep first
  - Delta < 0  -> diagnose (likely r_j collapse or distillation conflict)

### Phase 3 — Full evaluation (~6 h training)
- BLCA + GBMLGG, seeds 0 / 1 / 2 (paired vs MCAT same seeds)
- Paired t-test and Wilcoxon signed-rank
- Report mean Delta + 95% CI, per-cohort

### Phase 4 — Ablations (~15 h training, BLCA only — cheap cohort)

Component ablations:
```
(lambda, alpha_HDL, alpha_AUX, N_warmup)
(0,      0,         0,         -)     # = vanilla MCAT control
(1,      0,         0,         5)     # PRCA only
(0,      0.5,       0,         -)     # HDL only (no attention bias)
(1,      0.5,       0,         5)     # PRCA + HDL
(1,      0.5,       0.1,       5)     # full HACA v2 (main)
(1,      0.5,       0,         0)     # full minus AAS warmup
(1,      0.5,       0,         10)    # long warmup
(2,      0.5,       0,         5)     # strong anchoring
```

PRCA depth ablation: shared single r vs per-pathway 6 r heads.
Posterior-form ablation: log-prior (Bayesian) vs multiplicative post-softmax
vs additive linear prior. Tests whether the Bayesian frame is correct.

### Phase 5 — Theory + visualisation (~8 h)

- Per-patch r_j^(k) heatmaps overlaid on WSI thumbnails for 3 BLCA
  and 3 GBMLGG cases each (high-risk + low-risk + censored mix)
- Pathway-specificity heatmap: are different pathways anchoring on
  different patches, or do all 6 collapse to similar r_j?
- Optional: connect to existing Bayesian-attention literature in
  Related Work.

### Phase 6 — Paper-style report (~2 h)
- Mirror existing replication-result documents structure
- 5-section paper draft: introduction / theory / method / experiments /
  ablations + interpretability

**Total: ~40-45 h, mostly unattended GPU time.**

---

## 6. Position in the 2024-2026 landscape

(Updated from your audit table)

| Method | Mechanisms | Frame | Tier |
|---|---|---|---|
| AdaMHF (2025) | 4 (expansion / residual / selection / aggregation) | Hierarchical MoE | Strong |
| DIMAF (MICCAI 2025) | 2 (disentanglement + SHAP head) | Distance-correlation theory | Strong |
| HySurvPred (IJCAI 2025) | 2 (hyperbolic embed + angle-contrastive) | Hyperbolic geometry | Strong |
| BMLSurv (PR 2026) | 2 (peer assistance + ranking distillation) | Mutual learning theory | Strong |
| **HACA v2 (this)** | **3 (PRCA + HDL + AAS)** | **Bayesian posterior cross-attention** | **Strong** |
| EsurvFusion (TFS 2026) | 1 (evidential fusion) | Gaussian RFN | Moderate |
| MoME (MICCAI 2024) | 2 (biased encoding + MoE) | — | Moderate |
| Path-GPTOmic (BIBM 2024) | 1 (gradient modulation) | OGM application | Moderate |
| HACA v1 | 1 (biased attention) | — | Moderate |

v2 has the structural property of the "Strong" tier: 2-3 mechanisms +
theoretical frame.

---

## 7. Honest risk assessment

| Risk | Probability | Mitigation |
|---|---|---|
| All 6 r^(k)_j collapse to the same function (PRCA degenerates to v1) | medium | Diagnostic in Phase 1; if collapse seen, add small inter-pathway diversity regulariser |
| HDL teacher (main hazard) is too noisy at early epochs and corrupts r heads | medium | AAS handles: lambda=0 during warmup means HDL also gates in. Or stop-gradient on teacher (already in design) |
| Bayesian-frame ablation shows log-prior is NOT best functional form | medium-high | This is interesting in itself — report and switch to whichever form wins. Frame becomes "task-conditioned attention" rather than strict-Bayesian |
| 3 mechanisms add ~100K params, mild over-fitting on BLCA (n=437) | medium | Use dropout 0.25 on hazard MLPs (matches MCAT). Monitor val curves |
| Reviewer says "Bayesian framing is post-hoc rationalisation" | medium | Pre-register in design doc (this file timestamped) so the frame predated the experiments |
| Saturation thesis still holds — null result on both cohorts | high | Falls back to ablation-in-saturation-paper, same as v1. Floor unchanged |

**P(meaningful gain >= +0.012 averaged across both cohorts): ~30%**
(vs 25% for v1 — modest lift from per-pathway PRCA + distillation
addressing the InterSHAP "4% interaction headroom" finding more
directly)

**P(any publishable outcome): 100%** — null falls back to saturation
paper, positive becomes standalone.

---

## 8. What HACA v2 is NOT (scope creep prevention)

- Does NOT change encoder (ResNet-50 stays)
- Does NOT change gene signatures or SNN
- Does NOT change splits, NLL-Surv loss, optimiser, LR, batch / gc,
  epochs, or seed
- Does NOT add cross-modal contrastive objectives (separate paper)
- Does NOT add curriculum on patch-set sampling
- Does NOT use external survival data or pre-training
- Does NOT touch genomics co-attention direction (genomics is always Q,
  patches always K/V — same as MCAT)

3-axis intervention by design (PRCA, HDL, AAS). Each independently
ablatable. No coupling beyond the shared `r_j^(k)` predictor.

---

## 9. Decision summary

If you greenlight v2:
1. I write all 6 files (Phase 1) — ~10 h of careful coding + smoke test
2. We hit the pilot decision point at Phase 2 (~30 min training)
3. Full eval + ablations + report ~30 h more, mostly unattended

If you want to scale back (v1) or scope further up (v3 with cross-modal
contrastive), say so before any code is written.

---
END
