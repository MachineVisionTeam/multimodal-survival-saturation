# Pre-Registration — SiBaCo Fusion: Wasserstein-Barycenter Fusion Operator for Multimodal Cancer Survival

**Pre-registered (UTC):** `2026-05-25T16:38:46Z`
**Author:** Shemonti Barua · Machine Vision Team, Kennesaw State University
**Author email:** sbarua@students.kennesaw.edu
**Status:** committed BEFORE any SiBaCo training run. No `sibaco_fusion.py` exists in the local repo, no SiBaCo network module has been implemented, no SiBaCo training has been started for any cohort. The git commit hash of this file is the temporal anchor.

**Parent commits (for context):**
- `22cf6d1` — PCAF-Pathway pre-registration (KIRC + GBMLGG)
- `38d78f5` — PCAF-Pathway results (KIRC CLEAN WIN +0.0301, GBMLGG NULL)
- `63ae06a` — BLCA CONCH-Bimodal pre-registration
- `62d8b95` — BLCA CONCH-Bimodal results (spirit-WIN +0.0580)

---

## 1. Motivation

The PCAF-Pathway + BLCA CONCH-Bimodal experiments established that the **omic-representation axis** is the only positive axis in our 4-axis saturation map (KIRC +0.030, BLCA +0.058, GBMLGG NULL). The **fusion axis** (Axis 1) remained null across 10+ tested fusion families: GenoFiLM × 3, FiLM_residual, NoGateTrilinear, DAF, HACA × 3, cross-attention with prototype queries.

All 10 prior null fusion families operate in **feature-coordinate space** (R^32): multiplicative gating, additive shift, Kronecker products, attention-weighted sums. The InterSHAP audit (Swift et al., arXiv 2603.29977v2) measured ~4% additive cross-modal interaction variance on glioma — consistent with our nulls.

SiBaCo (Sinkhorn-Barycenter Codebook) Fusion is the **first fusion design we test that operates in a fundamentally different geometry**: each modality is mapped to a soft distribution over a shared learnable codebook, and the three distributions are merged via **entropic Wasserstein barycenter** (Fréchet mean in W₂ optimal-transport space). This is mathematically distinct from every prior coordinate-space, evidential, attention, or tensor-decomposition fusion we tested.

Per the SiBaCo design document (Sinkhorn-Barycenter Codebook Fusion: A Novel Drop-In Replacement for TrilinearFusion_A on TCGA-GBMLGG, 2026-05-25):
- No published precedent for entropic Wasserstein barycenter on a shared learnable codebook as the fusion operator for cancer survival prediction (literature audit through May 2026 covering MICCAI 2024-2025, ICCV/CVPR/ICLR 2023-2024, IEEE TMI 2022-2025, Nature Medicine 2024-2025, arXiv).
- The closest prior work (MOTCat ICCV 2023, ME-Mamba 2025) uses OT for *token alignment regularization*, not as the *fusion operator*.

This pre-registration locks the experimental design before any SiBaCo training.

---

## 2. Hypotheses

**H1 (primary):** On each cohort, replacing PF's TrilinearFusion_A with SiBaCo (with all other components held byte-identical to Cell A) produces c-Index improvement Δ ≥ +0.012 over the matched Cell A baseline, with paired-t p < 0.05 and Wilcoxon p < 0.05.

**H2 (cross-cohort generalization):** If H1 is positive on ≥ 1 cohort, this falsifies the strong-form fusion-axis saturation claim (that ALL fusion mechanisms are null in this regime). The number and identity of cohorts where H1 is positive informs the geometric-prior interpretation.

**Pre-registered priors (calibration audit) per cohort:**

| Outcome | P(KIRC) | P(GBMLGG) | P(BLCA) |
|---|---|---|---|
| CLEAN WIN (Δ ≥ +0.012, both p < 0.05) | 15-20 % | 5-10 % | 15-25 % |
| MARGINAL positive (+0.005 ≤ Δ < +0.012, ≥1 p < 0.05) | 20 % | 15 % | 20 % |
| NULL (both p ≥ 0.05) | 50-55 % | 65-70 % | 45-55 % |
| LOSS (Δ < 0, ≥1 p < 0.05) | 10 % | 10 % | 10 % |

These priors are slightly higher than for vanilla cross-attention (which we gave ~10% on KIRC) because the OT-geometric distinctness is real. Still firmly "more likely null than positive" — consistent with the InterSHAP additive-decomposition finding.

---

## 3. Experimental design (LOCKED)

### 3.1 Architecture spec (no tuning allowed)

| Hyperparameter | Value | Notes |
|---|---|---|
| Codebook size K | 64 | shared across all 3 modalities |
| Codebook dim d | 32 | matches PF's `opt.path_dim = opt.grph_dim = opt.omic_dim` |
| Sinkhorn regularization ε | 0.05 | fallback to 0.1 if NaN occurs in first epoch |
| Sinkhorn iterations T | 20 | iterative Bregman projection |
| Soft-assignment temperature τ_init | 1.0 | learnable, clamped ≥ 0.1 |
| Codebook initialization | randn / √d + LayerNorm at forward time | numerically stable |
| Barycenter weights λ_init | uniform: (1/3, 1/3, 1/3) for 3-modal; (1/2, 1/2) for 2-modal | learnable softmax logits |
| Per-modality bias `b_m` | learnable, init 0 | prevents modality collapse to same simplex location |
| Skip residual | concat with mean residual `(h_P + h_G + h_C)/3` → 64-d → Linear(64, 1) Cox head | matches PF's skip convention |
| Loss | Cox partial likelihood (unchanged) | no auxiliary loss |
| Optimizer | Adam (unchanged from PF) | |
| Learning rate | 1e-3 (unchanged) | |
| Schedule | 30 epochs total (`--niter 10 --niter_decay 20`) | identical to PF baseline |
| Batch size | 32 (unchanged) | |
| Dropout | 0.25 (unchanged) | |
| Total added parameters | ≈ 2.3k (K·d + per-modality biases + temperature + λ logits) | negligible vs PF baseline |

### 3.2 Per-cohort runs

| Cohort | Folds | Modalities | SiBaCo K modalities | Comparator |
|---|---|---|---|---|
| TCGA-KIRC | 15 (PF splits) | path + graph + omic | 3 | Cell A (CONCH-Pathomic with TrilinearFusion_A) |
| TCGA-GBMLGG | 15 (PF splits) | path + graph + omic | 3 | Cell A (CONCH-Pathomic with TrilinearFusion_A) |
| TCGA-BLCA | 5 (SurvPath splits) | path + omic (no cell-graph) | 2 | Cell A' (CONCH-Bimodal with BilinearFusion) |

**Total runs:** 35 fold-level c-Indexes (15+15+5). All 35 will be reported regardless of outcome.

### 3.3 What is being swapped vs held constant

For each cohort, SiBaCo replaces ONLY the fusion module. All other components are byte-identical to Cell A / Cell A':

| Component | Cell A / A' (baseline) | SiBaCo cell |
|---|---|---|
| Image encoder | CONCHv1.5 (frozen) | same |
| MMP K=16 prototype aggregation | per-fold | **same per-fold GMM, same 12288-d signature** |
| Path adapter | LayerNorm + Linear(12288, 32) + ReLU | same |
| Cell-graph branch (3-modal only) | PF GraphNet (frozen) | same |
| Omic branch | PF MaxNet (frozen, 322/320-d flat) | same |
| **Fusion module** | **TrilinearFusion_A (3-modal) / BilinearFusion (2-modal)** | **SiBaCo** ← only difference |
| Classifier | Linear(64, 1) + Sigmoid·6−3 | same |
| Loss | Cox PL | same |
| Splits, optimizer, LR, schedule, seeds | unchanged | same |

This is a pure fusion-axis ablation by design.

### 3.4 Pre-registered comparator (locked)

**Cell A / Cell A' c-Indexes used for paired comparison are the ones already reported in:**
- KIRC: `_publish/.../conch_pathomic/PCAF_PATHWAY_RESULTS.md` (mean 0.7188 ± 0.0471, n=15)
- GBMLGG: `_publish/.../conch_pathomic/PCAF_PATHWAY_RESULTS.md` (mean 0.8075 ± 0.0774, n=15)
- BLCA: `_publish/.../conch_pathomic/BLCA_CONCH_BIMODAL_RESULTS.md` Cell A' (mean 0.6157 ± 0.0155, n=5)

These per-fold numbers are committed and locked. SiBaCo is compared **paired** against these exact folds — no re-running of Cell A is performed (and would not be valid for a paired test).

---

## 4. Success criteria (LOCKED, no post-hoc adjustment)

Primary outcome: paired-t test of SiBaCo vs Cell A / Cell A' c-Indexes, per cohort.

| Outcome | Definition |
|---|---|
| **CLEAN WIN** | Δ ≥ +0.012 AND paired-t p < 0.05 AND Wilcoxon p < 0.05 |
| **MARGINAL positive** | +0.005 ≤ Δ < +0.012 with at least one test p < 0.05 |
| **NULL** | both p ≥ 0.05 |
| **LOSS** | Δ < 0 with at least one test p < 0.05 |
| **KILL CRITERION** | Δ < +0.005 AND worse IBS than baseline → fusion axis confirmed null in OT geometry too; do not pursue further on that cohort |

**Note on BLCA n=5:** Wilcoxon's minimum 2-sided p-value with n=5 and unanimous sign is 2/2⁵ = 0.0625. As pre-registered for BLCA in `BLCA_CONCH_BIMODAL_PREREGISTRATION.md`, we will also report Cohen's d_paired as the power-independent effect-size measure.

---

## 5. Cross-cohort verdict matrix (LOCKED interpretation)

| KIRC | GBMLGG | BLCA | Conclusion |
|---|---|---|---|
| NULL | NULL | NULL | Fusion axis CLOSED across OT geometry too. SiBaCo adds the 11th null family to the saturation map. The strong-form fusion-axis-closed claim is firmly established. |
| WIN | NULL | NULL | Fusion axis PARTIALLY OPEN on KIRC only via OT geometry. Investigate whether this is cohort-specific. |
| WIN | WIN | WIN | Fusion axis is NOT closed under OT geometry. Strongest result. Re-evaluate saturation thesis: the prior 10 nulls were coordinate-space-specific, not all-geometry. |
| Any other combination | | | Report transparently with mechanistic discussion. |

---

## 6. Computational plan

| Phase | Step | Time |
|---|---|---|
| 0 | This pre-registration (committed locally — no push) | done |
| 1 | Implement `sibaco_fusion.py` (~120 LOC) + synthetic smoke tests | 1 day |
| 2 | Integrate into PF (`network_sibaco.py` wrapper + 3 small patches) | half day |
| 3 | Real-data smoke test (1 fold × 2 epochs on KIRC) | 30 min |
| 4 | KIRC 15-fold sweep + paired-t analysis + commit RESULTS_kirc_sibaco.txt | 1 hour |
| 5 | GBMLGG 15-fold sweep + paired-t analysis + commit RESULTS_gbmlgg_sibaco.txt | 1.5 hours |
| 6 | BLCA 5-fold sweep + paired-t analysis + commit RESULTS_blca_sibaco.txt | 30 min |
| 7 | Combined cross-cohort report (`SIBACO_COMBINED_RESULTS.md`) + README update | 1 day |
| **End** | **Single push of all SiBaCo commits to GitHub** | atomic |
| **Total wall clock** | | **~3 days** |

---

## 7. What we will NOT do (analytic-flexibility constraints)

- We will NOT tune K, ε, T, τ_init, or any other SiBaCo hyperparameter to chase a positive result. Defaults are locked.
- We will NOT skip cohorts. All 3 cohorts × 5+ folds will be trained and reported.
- We will NOT change the +0.012 threshold post-hoc.
- We will NOT add SiBaCo variants (e.g., learned-per-sample λ, different codebook init, hard-Sinkhorn, log-domain Sinkhorn, K-sweep ablations) if SiBaCo nulls. The first run is the test.
- We will NOT compare SiBaCo against an unmatched baseline. The paired comparator is committed at: KIRC `38d78f5`, GBMLGG `38d78f5`, BLCA `62d8b95`.
- We will NOT push commits to GitHub during the SiBaCo work. Only one push at the end, after all 3 cohort results are committed locally.

---

## 8. Diagnostic protocols (run regardless of outcome)

These are PRE-SPECIFIED reports — not contingent on the result:

| Diagnostic | When |
|---|---|
| Codebook visualization (UMAP/PCA projection of C ∈ R^{64×32}) | after KIRC training |
| Per-modality `argmax(p_m)` distribution | after KIRC training |
| Learned λ (barycenter weights) per cohort | after each cohort |
| τ (temperature) end-of-training value | after each cohort |
| Per-fold paired delta histogram | per cohort |
| IBS (Integrated Brier Score) vs baseline | per cohort, applies kill criterion |

---

## 9. Honest limitations acknowledged

1. **No public push of pre-reg before training.** Per project decision (to avoid GitHub collaborator-attribution issues), this pre-reg is committed locally only and will be pushed atomically with results at the end. Validity rests on (a) cryptographic git commit chain (parent → child immutable), (b) commit timestamps, (c) the local commit being made before any `sibaco_fusion.py` file exists in the repo. The pre-reg commit hash + timestamp is the anchor.

2. **The SiBaCo design proposal has not been independently peer-reviewed.** It is the recommendation of a calibrated expert-review document (the same one that correctly predicted PCAF-vanilla would null), but no published precedent exists for this exact design. We accept the theoretical-novelty risk.

3. **InterSHAP audit may bound our maximum gain.** If interaction variance is genuinely ~4% additive on TCGA-GBMLGG, the theoretical ceiling for ANY fusion mechanism is limited. SiBaCo's OT-geometric argument is that "distribution-space" interaction may not be captured by InterSHAP's Shapley decomposition — this is a defensible but not proven claim.

4. **BLCA n=5 underpowered for Wilcoxon.** Acknowledged. Cohen's d will be the secondary measure.

---

## 10. Commit log convention

Pre-registration commit (this file):
```
SiBaCo pre-registration: 1-cell fusion-axis test on 3 cohorts before any training
```
NO `Co-Authored-By` trailer. Sole author: Shemonti Barua.

Per-cohort results commits will be:
```
SiBaCo KIRC results: <verdict>
SiBaCo GBMLGG results: <verdict>
SiBaCo BLCA results: <verdict>
```

Final commit:
```
SiBaCo combined results across 3 cohorts: <X> WIN / <Y> NULL / <Z> ... — 4-axis saturation update
```

All commits locally only. Single `git push origin main` at the end of Phase 7.

---

*End of pre-registration. No SiBaCo code, no SiBaCo training, no SiBaCo results exist at the time this file is committed.*
