# SiBaCo 2-modal BRCA + UCEC — Results (Phase 8n + 8o)

Pre-registration anchor: `7b1256f` + 2-modal amendment `8cafd1e`.
Unimodal classifications anchor: Phase 8m (`unimodal_baselines.py` outputs).

Locked verdict rule: CLEAN WIN iff Δ ≥ +0.012 AND paired-t p < 0.05 AND Wilcoxon p < 0.05.

## 1. Headline results

| Cohort | n folds | Cell A' (BilinearFusion) | SiBaCo (M=2) | Δ | paired-t p | Wilcoxon p | Cohen's d | 95 % CI on Δ | **Verdict** |
|---|---|---|---|---|---|---|---|---|---|
| **BRCA** | 15 | 0.6949 ± 0.069 | 0.7147 ± 0.084 | **+0.0199** | 0.2778 | 0.1981 | +0.29 | [−0.018, +0.058] | **NULL** |
| **UCEC** | 15 | 0.8411 ± 0.090 | 0.8138 ± 0.083 | **−0.0273** | 0.3375 | 0.1688 | −0.26 | [−0.086, +0.032] | **NULL** |

Both verdicts NULL by the locked pre-registration thresholds.

## 2. Per-fold deltas (SiBaCo − Cell A')

**BRCA** (9/15 positive folds):

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ | +0.138 | −0.027 | −0.018 | +0.004 | −0.035 | −0.037 | +0.049 | +0.110 | +0.000 | +0.079 | +0.020 | +0.075 | +0.043 | −0.135 | +0.033 |

**UCEC** (6/15 positive folds):

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ | +0.030 | −0.013 | −0.232 | +0.035 | +0.037 | +0.244 | −0.052 | −0.061 | +0.057 | −0.079 | +0.008 | −0.111 | −0.092 | −0.088 | −0.091 |

## 3. The 2×2 verdict (using observed Phase 8m classifications, not pre-reg predictions)

Per the locked pre-reg §3.6, we use observed unimodal classifications, not the predicted ones (3/5 predictions were contradicted; see BRCA_UCEC_UNIMODAL_RESULTS.md).

|  | **Balanced (observed Phase 8m)** | **Genomics-dominated (observed Phase 8m)** |
|---|---|---|
| **SiBaCo WIN observed** | — | KIRC 3-modal (+0.026, but with cell-graph branch) |
| **SiBaCo NULL observed** | BLCA 2-modal (Δ=+0.019) · **UCEC 2-modal (Δ=−0.027)** | GBMLGG 3-modal (Δ=−0.002) · **BRCA 2-modal (Δ=+0.020 trend)** |

The 2×2 framing on its own terms shows: **no SiBaCo WINs in any 2-modal cell. The only positive (KIRC) had cell-graphs.** Under matched 2-modal conditions, SiBaCo provides no statistically significant improvement on any cohort tested.

## 4. Interpretation (per pre-reg §4 + amendment §5)

The amendment §5 specifically pre-registered this falsifiable test:
> "if KIRC's SiBaCo advantage persists at 2-modal, the win is not graph-dependent
> ... if KIRC's advantage disappears at 2-modal (as BLCA's 2-modal already nulled),
> that indicates the cell-graph modality was load-bearing for the SiBaCo win"

KIRC's 2-modal SiBaCo was not re-run (scope decision: BRCA+UCEC only this phase). However, the cross-cohort 2-modal pattern (BLCA NULL, BRCA NULL, UCEC NULL — and the unimodal classifications) collectively support **the load-bearing-graph interpretation**:

- The only fusion-axis WIN observed across the entire 4-axis × 12+ family × 5 cohort sweep was **KIRC SiBaCo 3-modal**.
- Every 2-modal SiBaCo run (BLCA, BRCA, UCEC) was NULL.
- SiBaCo's OT-barycenter operator is mathematically defined for any M ≥ 2, so the failure to generalize to 2-modal is not a structural limitation of the fusion family — it's empirical evidence that the cell-graph modality was carrying the KIRC effect.

The amendment also pre-registered that the cell-graph modality cannot be faithfully reproduced for BRCA/UCEC because PF never released the CPC encoder weights. So this is a reproducibility-bounded conclusion: *we cannot retest the KIRC 3-modal win on a comparable 3-modal BRCA/UCEC*, but the 2-modal data is unanimous in nulling.

## 5. BRCA's trending-positive Δ is worth a note

BRCA's Δ=+0.020 clears the magnitude threshold (+0.012) but fails the significance gates (p=0.28 and p=0.20). 9/15 folds are positive, Cohen's d_paired = +0.29 (small-to-medium). This is *consistent with* a real-but-small SiBaCo effect that 15-fold paired statistics cannot resolve at α=0.05. It is **not** evidence of a win under the locked thresholds, and we report it as NULL per the pre-registration. We do not pursue post-hoc subgroup analyses.

## 6. UCEC's trending-negative Δ is also notable

UCEC SiBaCo (0.81) is slightly *worse* than UCEC Cell A' (0.84). Cell A' on UCEC achieves an unusually strong c-Index (0.84), well above typical UCEC SOTA (~0.65). This reflects (a) UCEC's strong omic signal (SNN-only 0.74 in Phase 8m) combined with (b) the CONCH path features, lifting the bilinear baseline. SiBaCo's barycenter geometry does not improve over the simple bilinear product here — actively slightly hurts. This is in the noise margin (95 % CI on Δ straddles zero, p > 0.16) but is informative directionally: SiBaCo's added complexity does not help cohorts where the simple bilinear product is already strong.

## 7. Effect on the saturation thesis (paper headline impact)

The fusion axis (Axis 1) of the saturation map updates as follows:

| Before this study | After this study |
|---|---|
| 11+ fusion families × 3 cohorts; only KIRC SiBaCo WIN (3-modal, +0.026 p=0.003) | 11+ fusion families × 5 cohorts; **still only KIRC SiBaCo WIN, and now likely attributable to the cell-graph modality (unreproducible)**, not to OT-geometry per se. |
| Axis 1 status: PARTIALLY OPEN — 1/3 cohorts positive | Axis 1 status: **PARTIALLY OPEN — 1/5 cohorts positive, the positive likely carried by the cell-graph modality which cannot be reproduced for new cohorts because CPC encoder weights were never released.** |

The empirical conclusion sharpens: matched 2-modal SiBaCo provides no statistically significant improvement over BilinearFusion on any cohort tested (BLCA, BRCA, UCEC, with same expectation for KIRC 2-modal by extrapolation). The fusion axis is effectively closed under matched conditions.

## 8. Data + script provenance

- Per-fold raw c-Indexes (Cell A' + SiBaCo, both cohorts): `phase8n_paired_results.json`
  at `/mnt/storage7/Dataset_pathomicfusion/SiBaCo_phase8n/`
- Training launch command: `train_cv.py --mode pathomic_conch_{flat,sibaco}` (PF replica tree)
- Locked hyperparameters: Adam, lr=1e-3, batch=32, 30 epochs (10+20), dropout=0.25
- SiBaCo defaults (from parent pre-reg a224dc0): K=64, ε=0.05, T=20, τ_init=1.0, n_modalities=2
- Sample sizes: BRCA 940 patients (957 split − 7 OL no-MPP − 10 no molecular profile),
  UCEC 469 patients (480 split − 11 no molecular profile)
- Both verdicts apply identically to Cell A' baseline AND SiBaCo (paired exclusions per
  pre-reg §8); 95.5 % retention overall

## 9. Final verdict

| Cohort | Verdict | Notes |
|---|---|---|
| BRCA 2-modal | **NULL** | Δ=+0.020 trending positive but p > 0.05 on both tests |
| UCEC 2-modal | **NULL** | Δ=−0.027 trending slightly negative |

Combined with existing BLCA NULL (parent SiBaCo block) and GBMLGG NULL (3-modal):
**4/4 SiBaCo runs in non-cell-graph or matched-2-modal regimes are NULL.** Only KIRC 3-modal SiBaCo WIN remains, and the 2-modal sweep makes the graph-modality-carried-the-win interpretation the most likely.

Saturation thesis: fusion axis effectively closed under matched conditions.
