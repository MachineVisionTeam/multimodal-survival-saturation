# Pre-Registration — BLCA CONCH-Bimodal: third-cohort representation-axis test

**Pre-registered (UTC):** `2026-05-22T18:17:55Z`
**Author:** Shemonti Barua · Machine Vision Team, Kennesaw State University
**Author email:** sbarua@students.kennesaw.edu
**Status:** committed BEFORE any BLCA training run. No CONCHv1.5 feature
extraction, GMM fitting, pkl building, or model training has occurred for
BLCA at the time of this commit. The hash of this commit serves as the
temporal anchor.
**Parent commits (for context):**
- `4f5c6a6` — PCAF-Pathway pre-registration (KIRC + GBMLGG)
- `be3654b` — PCAF-Pathway results (KIRC CLEAN WIN, GBMLGG NULL)

---

## 1. Motivation

The PCAF-Pathway 2×2 block (committed at `be3654b`) produced:
- **KIRC: CLEAN WIN** — Δ = +0.0301, paired-t p = 0.029, Wilcoxon p = 0.035
- **GBMLGG: NULL** — Δ = −0.0128, paired-t p = 0.412
- **Mechanism:** the entire gain comes from the omic representation axis
  (50 Hallmark pathway tokens). Cross-attention fusion is null.

Cross-cohort agreement is **partial** (1 win, 1 null). With only two
cohorts we cannot distinguish between:
- (a) **Cohort-specific KIRC artifact** — the pathway-token effect is specific to KIRC and does not generalize.
- (b) **Driver-curation-dependent effect** — the pathway-token effect generalizes to cohorts whose curated omic vector under-represents their biology, but not to cohorts (like GBMLGG) whose curated panel already captures driver signal.

A third cohort distinguishes these hypotheses. TCGA-BLCA (urothelial bladder cancer) is a natural choice:
- We have 437 BLCA WSIs already on disk
- BLCA has different biology from KIRC and GBMLGG (urothelial, no single dominant driver gene panel)
- SurvPath, MCAT, MOTCat, DIMAF have all reported on BLCA — direct literature comparability
- cBioPortal full RNA-seq is downloadable via the same LFS path we used for GBMLGG

---

## 2. Hypotheses

**H1 (primary):** On TCGA-BLCA (paired n ≈ 437 patients, 5 folds), replacing flat omic with 50 Hallmark pathway tokens produces c-Index improvement Δ ≥ +0.012 over the matched flat-omic baseline, with paired-t p < 0.05 and Wilcoxon p < 0.05 (same threshold as the KIRC/GBMLGG pre-registration).

**H2 (mechanistic, informational):** If H1 is positive, this supports hypothesis (b) above — pathway tokens help on cohorts whose curated omic vectors do not already include dominant driver genes. If H1 is null, this supports (a) — the KIRC effect was cohort-specific.

**Pre-registered priors (calibration audit):**
| Outcome | P(BLCA) |
|---|---|
| CLEAN WIN (Δ ≥ +0.012, both p < 0.05) | 25 % |
| MARGINAL WIN (0 < Δ < +0.012, at least one p < 0.05) | 25 % |
| NULL (both p ≥ 0.05) | 40 % |
| LOSS (Δ < 0, at least one p < 0.05) | 10 % |

The slightly higher P(WIN) on BLCA than on GBMLGG (25 % vs 15 %) reflects:
- BLCA has no published driver-gene panel as dominant as IDH1/TP53/ATRX for glioma
- BLCA baseline c-Index in published work is ~0.67–0.75 (mid-range — neither floor like KIRC nor ceiling like GBMLGG), so there is statistical headroom
- BLCA was tested by SurvPath and is responsive to pathway-style representations in their reported numbers

---

## 3. Experimental design (locked 2-cell block)

### Why fewer cells than KIRC/GBMLGG (2 instead of 4)

The PCAF-Pathway result already established that **cross-attention fusion contributes essentially nothing** when paired with pathway-tokenized omic (Cell D − Cell C ≈ 0 on KIRC; both null on GBMLGG). Re-testing the fusion axis on BLCA would be a known-null experiment and is explicitly **not** in scope. BLCA tests only the representation axis.

### Two cells

| Cell | Omic representation | Fusion | Mode name |
|---|---|---|---|
| **A'** | flat (Linear projection from full BLCA RNA-seq z-scores to 32-d) | BilinearFusion (PF's 2-way, image × omic) | `pgomic_conch_bimodal_flat` |
| **C'** | 50 Hallmark pathway tokens (mean-pool over genes per pathway from cBioPortal RNA-seq) | BilinearFusion (same as A') | `pgomic_conch_bimodal_pathway` |

### Architectural notes

- **Path branch (identical to KIRC/GBMLGG Cells A–D):** CONCHv1.5 patch encoder → GMM K=16 MMP aggregation → 12288-d slide signature → LayerNorm + Linear(12288, 32) + ReLU → 32-d path_vec.
- **No cell-graph branch** for BLCA. PF-format cell-graphs do not exist for BLCA and constructing them is outside the scope of this experiment (would require HOVER-NET / CellViT and per-nucleus encoder training, ~2–3 weeks).
- **Omic branch (Cell A'):** Linear(N_genes, 64) + ReLU + Linear(64, 32) + ReLU on the full cBioPortal RNA-seq z-score vector (~20,000 genes). Trained from scratch (no PF MaxNet checkpoint exists for BLCA).
- **Omic branch (Cell C'):** LayerNorm(50) + Linear(50, 32) + ReLU on the 50 pathway tokens (same `pathway_pool` architecture used in Cell C of the PCAF-Pathway experiment). Trained from scratch.
- **Fusion (both A' and C'):** PF's BilinearFusion module (`define_bifusion(fusion_type='pofusion', ...)`), 2-way image × omic, gated → 64-d → Linear(64, 1) + sigmoid · 6 − 3 → Cox PL.
- **Training:** 30 epochs, Adam, batch 32, lr 1e-3, dropout 0.25 — identical to PF training schedule.
- **Splits:** 5 folds, using SurvPath's published BLCA splits (`SurvPath/splits/5foldcv/tcga_blca/splits_{0..4}.csv`) for direct comparability with SurvPath / MCAT / DIMAF published numbers on this cohort.

### Why SurvPath splits, not BLCA's local `splits_{0..4}.csv`

The two split sets are similar but not identical. We use SurvPath's splits because:
- Direct comparability with SurvPath, MMP, MOTCat, DIMAF (all of whom used the same split conventions for BLCA)
- The MCAT-format local splits were built for HACA, a different prior experiment, and contain only train/val splits rather than fully balanced 5-fold CV

### Pre-registered total runs

2 cells × 5 folds × 30 epochs = 10 fold-level c-Index values. All 10 will be reported regardless of outcome.

---

## 4. Success criteria (locked, same thresholds as PCAF-Pathway pre-reg)

Primary outcome: paired-t test of Cell C' vs Cell A' across 5 folds.

| Outcome | Definition |
|---|---|
| **CLEAN WIN** | Δ (C' − A') ≥ +0.012 AND paired-t p < 0.05 AND Wilcoxon p < 0.05 |
| **MARGINAL WIN** | 0 < Δ < +0.012 with at least one test p < 0.05 |
| **NULL** | both p ≥ 0.05 |
| **LOSS** | Δ < 0 with at least one test p < 0.05 |

Note: 5 folds vs the 15 folds used on KIRC/GBMLGG gives substantially less statistical power. A null result on BLCA may reflect insufficient power rather than absence of effect. This will be acknowledged in the results section. We will additionally report Cohen's d_paired as an effect-size measure that is power-independent.

---

## 5. Cross-cohort verdict matrix (locked interpretation)

After BLCA result is in:

| KIRC | GBMLGG | BLCA | Conclusion |
|---|---|---|---|
| WIN | NULL | **WIN** | Cohort-dependent on driver-curation (hypothesis (b) supported) |
| WIN | NULL | **NULL** | KIRC-specific artifact; effect does not generalize (hypothesis (a) supported) |
| WIN | NULL | LOSS | KIRC-specific artifact + BLCA shows representation actually hurts. Strongest falsification of generalization. |
| WIN | NULL | MARGINAL | Inconclusive — the directional pattern matches (b) but does not clear the +0.012 threshold |

The first row above is the only outcome that supports a strong generalization claim.
The third row is the only outcome that actively falsifies the saturation thesis on this axis (and would warrant follow-up investigation before publication).

---

## 6. Compute plan

| Phase | Time |
|---|---|
| Phase 1: CONCHv1.5 feature extraction (TRIDENT batch on 437 WSIs, 4 GPUs) | ~3 hr |
| Phase 2: cBioPortal BLCA RNA-seq download + pathway token construction | ~30 min |
| Phase 3: per-fold GMM K=16 aggregation (5 folds) | ~20 min |
| Phase 4: network implementation (`network_conch_bimodal.py`) + smoke tests | 1 day |
| Phase 5: pkl construction (flat + pathway variants) | ~1 hr |
| Phase 6: 2-cell training (2 cells × 5 folds × 30 epochs, parallel on GPUs) | ~1 hr |
| Phase 7: paired analysis + verdict + writeup | ~1 day |
| **Total wall clock** | **~3–4 days** |

---

## 7. What we commit to (analytic-flexibility constraints)

- Only the two cells specified will be trained. No additional fusion variants or architectures.
- Hyperparameters identical to KIRC/GBMLGG pre-registration (30 epochs, Adam, batch 32, lr 1e-3). No per-cell tuning.
- All 5 folds reported. No fold-level dropping.
- Pathway-token construction uses the same recipe as KIRC/GBMLGG: mean-pool over Hallmark genes per pathway, no learnable per-pathway encoder.
- The +0.012 / p < 0.05 threshold is locked. Not adjustable after seeing results.
- If H1 is NULL, write up as null. No fishing for sub-population effects.
- If H1 is WIN, write up as a third-cohort positive that supports the driver-curation-dependence hypothesis. Do not over-claim "PCAF works across cancers" without acknowledging this is a 2-modal architecture, not the full PF setup.

---

## 8. Honest scope acknowledgments

- BLCA is 2-modal (image + omic). KIRC and GBMLGG were 3-modal (image + cell-graph + omic). This architectural asymmetry is unavoidable because BLCA does not have PF-format cell-graphs and constructing them was outside scope. The paper will state this clearly.
- 5 folds vs 15 folds: lower statistical power. We accept this in exchange for using SurvPath's published splits.
- The "flat omic" baseline for BLCA differs from the "flat omic" baseline for KIRC/GBMLGG. On KIRC/GBMLGG, the flat omic was PF's curated 320-d vector (hand-picked driver genes + CNVs + mutations). For BLCA, no equivalent curated panel exists, so we use a Linear projection from the full cBioPortal RNA-seq z-score vector. This is a fair comparison because:
  - Both flat baselines feed the same downstream architecture
  - Both are trained from scratch with the same hyperparameters
  - The representation axis tested (flat vs pathway-tokenized) is the same
  - But the result is not directly comparable to KIRC/GBMLGG Cell A absolute c-Index; only the paired Δ within BLCA matters for the pre-registered test

---

## 9. Commit log convention

Pre-registration commit message:
```
BLCA CONCH-Bimodal pre-registration: 2-cell rep-axis test before any training
```

After all 10 folds complete, the results commit message will be:
```
BLCA CONCH-Bimodal results (post-registration): {VERDICT}
```

The hash of THIS commit is the anchor. The results commit's parent will be this one.

---

*End of pre-registration. No BLCA training has been run at the time this file is committed.*
