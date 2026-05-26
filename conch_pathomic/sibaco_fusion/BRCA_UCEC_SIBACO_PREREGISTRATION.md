# Pre-Registration — SiBaCo Fusion 2×2 Extension: BRCA + UCEC (modality-balance × fusion-axis test)

**Pre-registered (UTC):** `2026-05-26T00:00:00Z`
**Author:** Shemonti Barua · Machine Vision Team, Kennesaw State University
**Author email:** sbarua@students.kennesaw.edu
**Status:** committed BEFORE any BRCA or UCEC data work. As of this commit:
- No TCGA-BRCA or TCGA-UCEC WSIs are downloaded to this machine.
- No CONCHv1.5 path features exist for either cohort.
- No HoVerNet cell-graphs exist for either cohort.
- No hand-curated omic panels exist for either cohort.
- No `data/TCGA_BRCA/` or `data/TCGA_UCEC/` directories exist in the PathomicFusion replica tree.
- No training run for either cohort has been launched.

The git commit hash of this file is the temporal anchor.

**Parent commits (for context):**
- `a224dc0` — SiBaCo pre-registration (3-cohort)
- `f4a6892` — SiBaCo KIRC CLEAN WIN (Δ=+0.0256, paired-t p=0.0028, Wilcoxon p=0.0015)
- `6aec1f3` — SiBaCo GBMLGG NULL (Δ=-0.0017, paired-t p=0.7648)
- `bac54f0` — SiBaCo BLCA NULL (Δ=+0.0187 magnitude pass, paired-t p=0.4455)
- `c009875` — README integration of 3-cohort SiBaCo results

---

## 1. Motivation

After the 3-cohort SiBaCo block, the cross-cohort pattern is:

| Cohort | Modalities | SiBaCo verdict | Hypothesized modality balance (pre-data) |
|---|---|---|---|
| KIRC | 3-modal | **CLEAN WIN** (+0.026, p=0.003) | balanced — histology + genomics both prognostic |
| GBMLGG | 3-modal | NULL (Δ≈0, p=0.76) | genomics-dominated — IDH1 / 1p19q-codel drives signal |
| BLCA | 2-modal | NULL (+0.019, p=0.45, n=5 noisy) | balanced — histology + RNA-seq both contribute |

A single 3-modal positive (KIRC) is not enough to claim that "SiBaCo wins on cohorts where modalities are balanced and saturates where genomics dominates." That claim needs **at least one more cohort per cell** of the 2×2:

|  | **Balanced modality** | **Genomics-dominated** |
|---|---|---|
| **SiBaCo win expected** | KIRC ✅ (observed) · BRCA (predicted WIN) | — |
| **SiBaCo null expected** | BLCA ✅ (observed, 2-modal asterisk) | GBMLGG ✅ (observed) · UCEC (predicted NULL) |

The BRCA + UCEC extension tests whether SiBaCo's positive result on KIRC generalizes specifically to cohorts where histology and genomics carry comparable prognostic signal. UCEC adds a second genomics-dominated null to strengthen (or break) the saturation argument on that axis.

This is a **prediction-driven extension**, not a fishing expedition: each cohort's expected outcome is locked here before any feature extraction begins, and the verdict rules are stated in §6.

---

## 2. Hypotheses

**H1 (BRCA primary):** On TCGA-BRCA, replacing PF's TrilinearFusion_A with SiBaCo (with all other components held byte-identical to Cell A) produces c-Index improvement Δ ≥ +0.012 over the matched Cell A baseline, with paired-t p < 0.05 AND Wilcoxon p < 0.05.

**H2 (UCEC primary, falsificationist):** On TCGA-UCEC, the same SiBaCo swap produces NO statistically significant improvement (Δ < +0.012 OR paired-t p ≥ 0.05 OR Wilcoxon p ≥ 0.05).

**H3 (modality-balance mediator):** The cohort-specific SiBaCo verdict is mediated by pre-trained-unimodal modality balance, where balance is operationalized in §3.6. Specifically: cohorts classified "genomics-dominated" by the pre-locked rule null; cohorts classified "balanced" win.

**Pre-registered priors (calibration audit):**

| Outcome | P(BRCA) | P(UCEC) |
|---|---|---|
| CLEAN WIN (Δ ≥ +0.012, both p < 0.05) | 35-45 % | 5-10 % |
| MARGINAL positive (+0.005 ≤ Δ < +0.012, ≥1 p < 0.05) | 20 % | 10 % |
| NULL (both p ≥ 0.05) | 30-40 % | 70-80 % |
| LOSS (Δ < 0, ≥1 p < 0.05) | 5-10 % | 5-10 % |

Note: BRCA prior is meaningfully higher than KIRC's was (which was 15-20 %). This is calibrated by the fact that we now have an observed KIRC win and BRCA shares the balanced-modality property. If BRCA nulls anyway, our modality-balance mediation hypothesis (H3) is weakened.

---

## 3. Experimental design (LOCKED)

### 3.1 Architecture spec (no tuning allowed)

Identical to the SiBaCo parent pre-reg (a224dc0 §3.1):

| Hyperparameter | Value |
|---|---|
| Codebook size K | 64 |
| Codebook dim d | 32 |
| Sinkhorn regularization ε | 0.05 (fallback 0.1 if NaN in epoch 1) |
| Sinkhorn iterations T | 20 |
| Soft-assignment temperature τ_init | 1.0 (learnable, clamped ≥ 0.1) |
| Codebook init | randn / √d + LayerNorm at forward |
| Barycenter weights λ_init | uniform (1/3, 1/3, 1/3) |
| Per-modality bias `b_m` | learnable, init 0 |
| Skip residual | concat with mean residual → 64-d → Linear(64, 1) Cox head |
| Loss | Cox partial likelihood |
| Optimizer | Adam |
| LR | 1e-3 |
| Schedule | 30 epochs (`--niter 10 --niter_decay 20`) |
| Batch size | 32 |
| Dropout | 0.25 |

**Locked code:** `/home/sbarua/Region_based_segmentation/CONCH-Pathomic/sibaco_fusion/sibaco_fusion.py`, unchanged from parent pre-reg.

### 3.2 Per-cohort runs

| Cohort | Folds | Modalities | SiBaCo n_modalities | Comparator |
|---|---|---|---|---|
| TCGA-BRCA | 15 (custom-built, stratified by site, see §3.4) | path + graph + omic | 3 | Cell A (CONCH-Pathomic with TrilinearFusion_A) |
| TCGA-UCEC | 15 (custom-built, stratified by site, see §3.4) | path + graph + omic | 3 | Cell A (CONCH-Pathomic with TrilinearFusion_A) |

Plus secondary runs (for field comparability with MCAT/SurvPath/MOTCat leaderboards, *reported separately, not part of primary 2×2 verdict*):

| Cohort | Folds | Notes |
|---|---|---|
| TCGA-BRCA | 5 (MCAT splits, on disk) | identical pipeline, different splits |
| TCGA-UCEC | 5 (MCAT splits, on disk) | identical pipeline, different splits |

**Total primary runs:** 60 fold-level c-Indexes (2 cohorts × 15 folds × 2 architectures). All reported regardless of outcome.

### 3.3 What is swapped vs held constant

Identical to parent SiBaCo pre-reg §3.3. Only the fusion module changes between Cell A and the SiBaCo cell. Encoders, adapters, splits, optimizer, LR, schedule, seeds — all byte-identical.

### 3.4 Splits

15-fold splits will be built once per cohort and committed to disk before any training run. Construction rule (locked):
- One slide per case (drop duplicate slides per case_id, keep diagnostic FFPE only).
- Stratify by `tissue_source_site` (cBioPortal column) to match SurvPath's stratification convention.
- Fixed random seed = 42.
- Output: `data/TCGA_{BRCA,UCEC}/splits/{brca,ucec}15cv_st_conch.pkl` in PF's `cv_splits` dict format.

The split files are derived deterministically from cBioPortal case lists; no test-fold information will be used during omic-panel curation or any pre-training step.

### 3.5 Hand-curated omic panels (LOCKED methodology)

Mirrors KIRC's PF-style panel (which produced a 362-d feature vector). For each cohort:

**Source:** cBioPortal pan-cancer-atlas studies — `brca_tcga_pan_can_atlas_2018` and `ucec_tcga_pan_can_atlas_2018` (open-access, no auth).

**Curation procedure (locked, deterministic):**
1. Pull `Mutated_Genes.txt` from cBioPortal study page (top mutated genes with MutSig q-value, OncoKB cancer-gene annotation).
2. Pull `CNA_Genes.txt` (GISTIC peaks, amp + del at q < 0.05).
3. Pull `data_RNA_Seq_v2_mRNA_median_Zscores.txt` (RNA-seq z-scores per gene per sample).
4. Define the panel as the union of:
   - Genes with mutation frequency ≥ 5 % AND OncoKB-annotated
   - GISTIC peak genes at q < 0.05
5. Intersect with genes present in the RNA-seq table.
6. Target panel size: **100–130 genes per cohort**. If raw union exceeds 130, rank by max(mutation frequency, GISTIC q-significance rank, RNA-seq variance rank) and keep top 130. If union is below 100, lower mutation-frequency threshold to 3 % and re-derive (locked fallback rule).
7. Per gene, produce **3 features**: mutation binary {0,1} + CNA categorical mapped to ordinal {-2,-1,0,+1,+2} + RNA-seq z-score (continuous).
8. **Final dim:** 300–390 per cohort. Comparable to KIRC's 362.

The curated gene lists for both cohorts will be committed to disk under `data/TCGA_{BRCA,UCEC}/omic_panel/genes.txt` BEFORE any training begins. The git commit hash of those files is a second-stage anchor.

### 3.6 Modality-balance classification rule (LOCKED, pre-data)

Before SiBaCo training, run **two unimodal sanity-baselines on each cohort** at 15-fold:

- **SNN-only** = PF's MaxNet on the curated omic panel + Cox head. No path, no graph.
- **CNN-only** = PF's CONCH path adapter + Cox head. No graph, no omic.

Both use identical splits, optimizer, LR, and schedule to the multimodal baseline.

**Classification rule (locked threshold):**

A cohort is classified **"genomics-dominated"** iff BOTH:
1. `mean(SNN c-Index over 15 folds) − mean(CNN c-Index over 15 folds) ≥ +0.04`
2. paired-t p < 0.05 (SNN > CNN, 15 paired folds)

Otherwise the cohort is classified **"balanced"** (includes negative deltas and small positive deltas < +0.04).

**Why +0.04:** chosen pre-data to cleanly separate GBMLGG's known IDH-driven dominance (typical SNN−CNN ≈ +0.15-0.20 on TCGA-GBMLGG with 240-gene panel) from KIRC's modest genomics lead (typical Δ < +0.03). The threshold is a single number, locked here, and is not adjusted after seeing the SNN/CNN numbers.

**Pre-registered classification predictions:**
- BRCA: balanced (predict SNN−CNN < +0.04)
- UCEC: genomics-dominated (predict SNN−CNN ≥ +0.04). UCEC's molecular subtypes (POLE ultramutated, MSI, copy-number low/endometrioid, copy-number high/serous-like) per TCGA UCEC marker paper (Nature 2013) are genomics-defined; SNN is expected to recover survival-relevant subtype signal that CNN cannot.

If actual SNN/CNN numbers contradict the predicted classification, **we report the contradiction explicitly and re-interpret the SiBaCo verdict against the actual classification, not the predicted one.** No re-curation of the omic panel based on SNN/CNN performance.

### 3.7 Verdict thresholds (LOCKED, identical to parent SiBaCo pre-reg)

For each cohort, the SiBaCo verdict is:

| Verdict | Condition |
|---|---|
| **CLEAN WIN** | Δ ≥ +0.012 AND paired-t p < 0.05 AND Wilcoxon p < 0.05 |
| **MARGINAL** | Δ ≥ +0.005 AND at least one of (paired-t, Wilcoxon) p < 0.05 |
| **NULL** | not CLEAN WIN and not MARGINAL and not LOSS |
| **LOSS** | Δ < 0 AND paired-t p < 0.05 |

Plus reported alongside the verdict (for transparency): Cohen's d_paired, 95 % CI on Δ, per-fold deltas, fold sign-count.

---

## 4. Pre-registered 2×2 verdict interpretation rules

Define the four possible 2×2 cells based on observed SiBaCo verdicts:

| Outcome | BRCA | UCEC | Interpretation |
|---|---|---|---|
| **A. Pattern confirmed** | WIN or MARGINAL | NULL | Modality-balance mediation hypothesis supported. Saturation thesis updated to "fusion axis is open on balanced cohorts only; closed on genomics-dominated cohorts." Paper headline: *"OT-geometry fusion generalizes specifically on balanced-modality cohorts."* |
| **B. Universal saturation** | NULL | NULL | Modality-balance mediation hypothesis falsified. Saturation thesis strengthened to "fusion axis near-universally closed; KIRC is a cohort-specific exception." Paper headline: *"Saturation is near-universal; KIRC win is the rare cohort-specific exception."* |
| **C. Universal SiBaCo** | WIN or MARGINAL | WIN or MARGINAL | Modality-balance mediation falsified in the other direction. Saturation thesis weakened on fusion axis. Paper headline: *"SiBaCo's OT-barycenter geometry generalizes broadly; modality balance is not the mediator."* |
| **D. Inverted pattern** | NULL | WIN or MARGINAL | Most surprising outcome. Modality-balance hypothesis inverted. Paper headline: *"Genomics-dominated cohorts paradoxically benefit from OT-barycenter fusion."* |

Each outcome is independently publishable. Outcomes A and B are pre-registered as the higher-prior expectations (combined P ≈ 0.7 by §2 priors). Outcomes C and D are lower-prior but reportable as-is without re-interpretation.

If H3 (modality-balance mediation) is confirmed (outcome A) but the predicted modality classifications in §3.6 are inverted by the actual SNN/CNN numbers, this is reported as a **predicted-mechanism / wrong-direction** anomaly and flagged in the paper.

---

## 5. BLCA 2-modal asterisk (locked acknowledgement)

The 2×2 table includes BLCA in the "balanced / null" cell, but the BLCA SiBaCo run was 2-modal (no cell-graph; BLCA does not have PF-format cell graphs available, and we did not build them). All BRCA and UCEC runs in this extension are 3-modal.

The 2×2 will be presented in two forms in the paper:
- **Primary 2×2:** 3-modal cohorts only (KIRC, GBMLGG, BRCA, UCEC). BLCA listed separately.
- **Secondary 2×2:** including BLCA, with explicit asterisk that BLCA used 2-modal SiBaCo. Predicted to behave like a balanced cohort in 2-modal regime; observed result is NULL (Δ=+0.019, n=5 noisy).

If reviewers request BLCA 3-modal, we acknowledge that building BLCA cell graphs via the HoVerNet pipeline is an additional ~1-2 weeks of preprocessing and would be a follow-up pre-registration.

---

## 6. Pre-commit checklist (audit trail)

At commit time of this file, the following are true:

- [ ] No `data/TCGA_BRCA/` directory exists in the PF replica tree.
- [ ] No `data/TCGA_UCEC/` directory exists.
- [ ] No CONCHv1.5 features for BRCA or UCEC exist anywhere on this machine.
- [ ] No HoVerNet cell-graphs for BRCA or UCEC exist anywhere.
- [ ] No curated omic panel gene-list file exists for BRCA or UCEC.
- [ ] No SiBaCo / Cell A / SNN / CNN training run has been launched for either cohort.
- [ ] No 15-fold split file has been created for either cohort.
- [ ] MCAT-style 5-fold split files for both cohorts pre-existed on disk under `mcat_replication/MCAT/splits/5foldcv/` (used only for secondary leaderboard runs, not primary verdict).
- [ ] MCAT-style omic CSVs for both cohorts pre-existed on disk under `mcat_replication/MCAT/dataset_csv/` and `mcat_replication/MCAT/datasets_csv_sig/` — **explicitly NOT used** as the omic input for the primary 3-modal SiBaCo runs (different curation recipe; see §3.5).
- [ ] The SiBaCo module (`/home/sbarua/Region_based_segmentation/CONCH-Pathomic/sibaco_fusion/sibaco_fusion.py`) is unchanged from parent commit a224dc0.
- [ ] The PF wrappers (`network_sibaco.py`) are unchanged from parent commit a224dc0 (3-modal class already supports any 3-modal PF cohort with `input_size_omic` set per cohort opt).

All checklist items will be verified before commit.

---

## 7. Pipeline order (no shortcuts)

1. **Commit this pre-registration.** Anchor.
2. **Build 15-fold splits** for BRCA + UCEC. Commit.
3. **Curate omic panels** per §3.5 from cBioPortal. Commit gene lists + per-patient feature vectors.
4. **Download TCGA-BRCA + TCGA-UCEC WSIs** from GDC (open-access diagnostic FFPE).
5. **Extract CONCHv1.5 path features** via TRIDENT pipeline (segmentation → 20× patch → CONCHv1.5 → GMM K=16 MMP). Per-slide 12288-d signatures.
6. **Build cell graphs** via PF's HoVerNet + CPC + KNN pipeline. Per-slide PyG `Data` objects.
7. **Build PF data pkls** in the `cv_splits` dict format, one per cohort.
8. **Train unimodal SNN + CNN baselines** (per §3.6). Compute SNN−CNN delta. Lock modality-balance classification per §3.6 rule.
9. **Train multimodal Cell A baseline** (TrilinearFusion_A, 15-fold) per cohort.
10. **Train SiBaCo cell** (3-modal SiBaCo, 15-fold) per cohort. Identical wrapper code, only fusion module changes.
11. **Paired-t / Wilcoxon / Cohen's d / 95 % CI / per-fold table.** Apply §3.7 thresholds.
12. **Apply §4 2×2 interpretation rules** based on observed verdicts.
13. **Write `BRCA_UCEC_SIBACO_RESULTS.md`** with full per-fold tables, SNN/CNN modality-balance numbers, and verdict.
14. **Run secondary 5-fold MCAT-splits runs** (same pipeline, MCAT splits, identical features). Report as field-comparability appendix.
15. **Update README + commit + push.**

No step may be skipped. Step 8 (unimodal baselines) must complete and modality-balance be classified BEFORE Step 10 (SiBaCo training), so the predictions in §3.6 cannot be back-calibrated.

---

## 8. Failure modes (pre-registered)

| Scenario | Pre-registered response |
|---|---|
| SiBaCo training NaN in epoch 1 | Apply locked fallback ε = 0.1, re-run. Flag in results. |
| GDC WSI download incomplete for some cases | Drop cases with missing diagnostic FFPE from BOTH baseline and SiBaCo runs (paired). Report drop count. |
| Curated panel gene-count outside 100-130 range after §3.5 procedure | Apply locked fallback (lower mutation-frequency threshold to 3 %). If still outside, document and proceed; no further panel adjustment. |
| Cell-graph build fails on some slides (HoVerNet failure) | Drop affected slides from BOTH baseline and SiBaCo (paired). Report drop count. ≤10 % drop acceptable; >10 % triggers pre-registered re-discussion of cohort viability. |
| Cell A baseline c-Index radically below MCAT leaderboard (e.g., > 0.05 below) | Flag as preprocessing issue; do not start SiBaCo runs until baseline is in literature-consistent range. |
| Modality-balance classification contradicts §3.6 prediction | Report contradiction. Apply §4 verdict-interpretation rules using actual (not predicted) classification. |

---

## 9. Anchor

**This document committed at:** (to be filled by git commit hash after `git commit`)
**Author:** Shemonti Barua
**Lab:** Machine Vision Team, Kennesaw State University
**No SiBaCo training, no data download, no panel curation has been performed for BRCA or UCEC as of this commit.**
