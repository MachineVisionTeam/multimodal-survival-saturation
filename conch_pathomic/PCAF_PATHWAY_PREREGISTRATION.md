# Pre-Registration — PCAF-Pathway: Pathway-Tokenized Cross-Attention Fusion on CONCH-Pathomic

**Pre-registered (UTC):** `2026-05-22T04:33:51Z`
**Author:** Shemonti Barua · Machine Vision Team, Kennesaw State University
**Author email:** sbarua@students.kennesaw.edu
**Approved by:** project supervisor (verbal, prior to commit)
**Status:** committed BEFORE any 2×2 training run begins. No PCAF-Pathway training has occurred at the time of this commit. The hash of this commit serves as the temporal anchor.
**Related docs:**
- [`../README.md`](../README.md) — saturation paper repo overview
- [`../CONCH_PATHOMIC_OVERVIEW.md`](../CONCH_PATHOMIC_OVERVIEW.md) — CONCH-Pathomic Phase 0–6 results (to be added to this repo)
- expert review at `/home/sbarua/paper/sgementation paper/regon based segmentation/Fusion-Mechanism Innovation for Multimodal Cancer Survival_ A Calibrated Verdict and Pre-Registered Plan.pdf`

---

## 1. Background (state of evidence at registration time)

Two cohorts (TCGA-KIRC, TCGA-GBMLGG) have now produced **paired-t null results** on the encoder axis of the three-axis saturation map:

| Cohort | CONCH-Pathomic | PF baseline (paired) | Δ | paired-t p |
|---|---|---|---|---|
| KIRC | 0.7188 ± 0.0471 | 0.7184 ± 0.0513 | **+0.0004** | **0.946** |
| GBMLGG | 0.8075 ± 0.0774 | 0.8078 ± 0.0717 | **−0.0003** | **0.973** |

These null results are consistent with the broader 2024–2026 literature:
- **InterSHAP (Swift et al., arXiv 2603.29977, XAI 2026 late-breaking):** cross-modal interaction = 3–5 % of variance on glioma survival; better-discrimination models have *less* interaction, not more.
- **MMP fusion-only ablation (Song et al., ICML 2024):** Transformer cross-attention vs Optimal Transport cross-alignment averages c-Index 0.665 vs 0.665 across 6 cohorts.
- **DIMAF (Eijpe et al., MICCAI 2025):** best 2025 fusion-innovation paper, +1.85 % avg across 4 cohorts — but bundles encoder + training-protocol changes; pure fusion ablation isolates +0.005 on KIRC.
- **Student's own prior nulls:** 9 fusion families null on GBMLGG, 3 on BLCA; DSM-Pathomic null on IBS; GRFN-Pathomic mechanistically falsified.

An expert review (calibrated probability assessment, dated 2026-05-22) identified one untested axis: **omic representation upgrade** — replacing the flat 32-d SNN omic vector with **50 Hallmark pathway tokens** (the design choice universally adopted in SurvPath, MMP, MOTCat, APL, PIBD, DIMAF, but never independently ablated on a PF-class trimodal baseline). The review estimates P(≥ +0.012 win | fusion-only) ≈ 5–20 %, vs P(≥ +0.012 win | fusion + pathway-token representation) ≈ 30–40 %.

This pre-registration locks down a 2×2 experiment that **separately tests the representation axis and the fusion axis** before any results are seen.

---

## 2. Hypothesis

**H1 (primary):** On TCGA-KIRC (paired n = 417 patients, 15 folds), the joint intervention of (a) pathway-tokenized omic representation + (b) prototype × token-set cross-attention fusion produces c-Index improvement Δ ≥ +0.012 over the locked Pathomic Fusion baseline, with paired-t p < 0.05 and Wilcoxon p < 0.05.

**H2 (secondary):** Same as H1 but on TCGA-GBMLGG (paired n ≈ 489 patients, 15 folds, the 97 %-of-PF subset for which CONCH features are available).

**Axis-isolation sub-hypotheses (informational, not for win/loss determination):**
- H_rep: Pathway tokens alone (with locked TrilinearFusion) clear Δ ≥ +0.012 — would attribute any gain to the representation axis specifically.
- H_fus: Cross-attention alone (with locked flat omic) clears Δ ≥ +0.012 — would falsify the reviewer's claim that fusion alone is dominated by gated variants we already nulled.

---

## 3. Experimental design (locked 2×2 block)

| Cell | Omic representation | Fusion mechanism | Mode name |
|---|---|---|---|
| **A** | flat 32-d SNN (PF MaxNet output) | TrilinearFusion_A (PF) | `pathgraphomic_conch` (already run; serves as the within-block reference) |
| **B** | flat 32-d SNN | prototype × flat-vector cross-attention | `pathgraphomic_conch_xattn` |
| **C** | 50 Hallmark pathway tokens (mean-pooled to 32-d for Trilinear) | TrilinearFusion_A (PF) | `pathgraphomic_conch_pathway` |
| **D** | 50 Hallmark pathway tokens (as K/V token set) | prototype × pathway-token cross-attention | `pathgraphomic_conch_pcaf` |

Each cell trained on **both cohorts** (KIRC, GBMLGG) with **identical hyperparameters** to the locked PF baseline (15 folds, 30 epochs, Adam, batch 32, Cox PL, paired splits). Total runs: **4 cells × 2 cohorts = 8 sweeps**, **15 folds each = 120 fold-level c-Index values**.

All 8 runs **must complete** and **all results must be reported**. No variant-picking, no post-hoc dropping of cells. If a cell fails to train (e.g., NaN loss), it will be re-run with the same seed; if it still fails, the failure will be reported in the results, not excluded.

---

## 4. Architecture specifications

### Cell A (reference — already completed)
- Path: CONCH 768-d patches → GMM K=16 soft-aggregation → 12288-d → `LayerNorm + Linear(12288, 32) + ReLU` → 32-d path_vec
- Graph: GraphNet → 32-d graph_vec (frozen)
- Omic: PF MaxNet on flat omic → 32-d omic_vec (frozen)
- Fusion: TrilinearFusion_A → 64-d → Linear(64,1) + sigmoid

### Cell B (vanilla PCAF, the "fusion-only" arm)
- Path: CONCH 768-d patches → GMM K=16 → **16 prototypes of dim 768 (NOT flattened)** → `Linear(768, 32)` per-prototype → (B, 16, 32) prototype tokens
- Graph: GraphNet → 32-d → unsqueezed to (B, 1, 32)
- Omic: PF MaxNet on flat omic → 32-d → unsqueezed to (B, 1, 32)
- K/V set: `concat([graph_token; omic_token], dim=1)` → (B, 2, 32)
- Q: 16 prototype tokens (B, 16, 32)
- Fusion: `MultiHeadAttention(d=32, heads=4)`, Q × K/V → (B, 16, 32) → mean-pool over prototypes → (B, 32) → Linear(32, 1) + sigmoid · 6 − 3

### Cell C (pathway-rep + Trilinear)
- Path: same as Cell A
- Graph: same as Cell A
- Omic: pathway tokens (50, d_token) → **learnable attention pool** → 32-d omic_vec
- Fusion: TrilinearFusion_A (unchanged)

### Cell D (PCAF-Pathway — the primary hypothesis arm)
- Path: same as Cell B (16 prototypes of dim 32)
- Graph: same as Cell B (1 token of dim 32)
- Omic: pathway tokens (50, d_token) → `Linear(d_token, 32)` → (50, 32)
- K/V set: `concat([graph_token; pathway_tokens], dim=1)` → (B, 51, 32)
- Q: 16 prototype tokens (B, 16, 32)
- Fusion: `MultiHeadAttention(d=32, heads=4)`, Q × K/V → (B, 16, 32) → mean-pool over prototypes → (B, 32) → Linear(32, 1) + sigmoid · 6 − 3

**Frozen components in all 4 cells:** CONCHv1.5 ViT (per-patch encoder), HEST tissue segmenter, GraphNet, per-fold GMM means/vars/weights, PF MaxNet (when used in Cells A and B for the flat omic).

**Trainable parameter counts** (target — exact will be reported post-implementation):
- Cell A: ~4.0 M (includes 417k adapter + 3.5 M TrilinearFusion)
- Cell B: ~70 k (per-prototype Linear + MHA + classifier; no TrilinearFusion)
- Cell C: ~4.0 M (same as A + small pathway pool head)
- Cell D: ~80 k (per-prototype Linear + pathway Linear + MHA + classifier)

---

## 5. Data prep specification

### Hallmark pathway tokens (Steps 1–3 — same for all cohorts)

Following the user requirement: **reuse SurvPath's Hallmark gene-set mapping** (avoids gene-ID ambiguity issues).

- **Source:** clone `mahmoodlab/SurvPath` repo; load their `Hallmarks.csv` or equivalent gene-set definition file. Use their exact gene-symbol-to-pathway mapping with no modifications.
- **Pathway count:** K=50 (Hallmark H collection).
- **Per-patient token construction:** for each pathway p with member genes G_p:
  - From PF's pkl `x_omic` vector, identify the columns whose names match genes in G_p (after `_rnaseq` suffix stripping for the GBMLGG pkl convention).
  - Token_p = mean of the matched gene values (mean pool; matches SurvPath's published recipe).
  - Pathways with zero matched genes in the patient's available omic vector: token_p = zero vector of dim 1, padded to d_token = 1. (These will be flagged in the audit.)
- **Token dim:** d_token = 1 initially (per-pathway scalar from mean pool); will be linearly projected to 32 inside the network. *Alternative:* if SurvPath's recipe uses higher-dim tokens (e.g., one-hot + value), follow their exact recipe verbatim.
- **Output:** per-patient (50, d_token) tensor saved to `aggregated_pathway/{cohort}/patient_pathway_tokens.pt`.

### Pkl build (4 pkls per cohort, but mostly identical)
- Cells A, B share x_omic = flat (320 or 322 d)
- Cells C, D share x_omic_pathway = (50, d_token)
- Both share x_path = CONCH MMP signature (12288-d)
- All share x_grph, e, t, g, x_patname (unchanged from PF pkl)

To minimize disk: build two pkls per cohort — one with flat omic (used by A, B), one with pathway tokens (used by C, D). The cell-specific runs select the right one via the `--mode` flag.

---

## 6. Success criteria (locked, no post-hoc adjustment)

### Primary outcome
For each cohort, compute the **paired-t test** and **Wilcoxon signed-rank test** comparing **Cell D vs Cell A** (the natural pre-vs-post comparison: full PCAF-Pathway intervention vs current CONCH-Pathomic baseline).

### Verdict table (committed before any results are seen)

| Outcome | Definition | Interpretation |
|---|---|---|
| **CLEAN WIN** | Δ (D − A) ≥ +0.012 **AND** paired-t p < 0.05 **AND** Wilcoxon p < 0.05 | The combined intervention beat the saturation threshold. Run the secondary isolation analysis (Cell C vs A for "representation only"; Cell B vs A for "fusion only") to attribute the gain. |
| **MARGINAL WIN** | 0 < Δ (D − A) < +0.012 with at least one test p < 0.05 | Within-noise positive. Report and discuss; emphasize that gain is below the pre-registered threshold and does not falsify the saturation thesis. |
| **NULL** | both p ≥ 0.05 (regardless of Δ sign) | Even the strongest available intervention (pathway tokens + cross-attention) fails to beat the locked baseline. The saturation thesis is now closed across **four axes** (architecture, encoder, loss, representation). Strongest available falsification of the fusion-innovation hypothesis on TCGA-GBMLGG / TCGA-KIRC. |
| **LOSS** | Δ (D − A) < 0 with at least one test p < 0.05 | Significant decrease. Investigate as a methodological problem (likely overfitting on the smaller PF baseline since PCAF has ~50× fewer params). Report transparently. |

### Cross-cohort agreement
- If both cohorts agree on the verdict (WIN/WIN, NULL/NULL, etc.): the conclusion is firm.
- If they disagree (e.g., WIN on KIRC, NULL on GBMLGG): report both; the saturation thesis is **partially closed**; flag KIRC-vs-GBMLGG differential as a finding.

### Sub-hypothesis attribution (informational only)
- If WIN and Cell C ≥ +0.012 with p < 0.05: gain is attributable to **representation axis** (pathway tokens), not fusion. Adds a row to Axis 4 (Representation) of the saturation map. Publishable as "representation axis is the only one with a positive result."
- If WIN and Cell B ≥ +0.012 with p < 0.05: gain is attributable to **fusion axis** (cross-attention). Would falsify the saturation thesis on Axis 1 (Architecture/Fusion). Would require a follow-up confirmation paper.
- If WIN but both C and B individually null: gain is attributable to the **interaction** (representation × fusion). Strongest case for PCAF-Pathway as a method.

---

## 7. Statistical analysis plan

For each pair (Cell X vs Cell A), both cohorts:
1. **Per-fold deltas:** Δ_k = c-Index_k(X) − c-Index_k(A) for k = 1..15
2. **Paired t-test:** `scipy.stats.ttest_rel(X, A)` — two-sided, df = 14
3. **Wilcoxon signed-rank:** `scipy.stats.wilcoxon(X, A)` — two-sided
4. **95 % CI on Δ:** `Δ̄ ± 1.96 · std(Δ)/√15`
5. **Effect size:** Cohen's d_paired = Δ̄ / std(Δ)

**Multiple comparisons:** the primary outcome is **D vs A only**, single test per cohort. Cells B and C are reported for axis isolation but do NOT count as separate primary tests, so no Bonferroni adjustment is applied to the primary outcome. The B-vs-A and C-vs-A tests will be reported with their raw p-values for transparency.

**No interim analyses.** All 8 sweeps will run to completion before any p-values are computed.

---

## 8. Compute and timeline

| Phase | Step | Time |
|---|---|---|
| 0 | This pre-registration committed | done |
| 1 | Clone SurvPath, identify Hallmark mapping, build per-patient pathway tokens for both cohorts | 1 day |
| 2 | Implement Cells B, C, D architectures (`network_conch_xattn.py`, `network_conch_pathway.py`, `network_conch_pcaf.py`) | 1 day |
| 3 | Smoke tests: unit tests for each network + 1-fold smoke runs on KIRC for all 3 new cells | ½ day |
| 4 | Build pathway pkls for both cohorts (`build_pkl_pathway.py`) | ½ day |
| 5 | Launch 8 sweeps (4 cells × 2 cohorts × 15 folds × 30 epochs) | ~3 hr GPU on 4× L40 |
| 6 | Paired-t analysis, results table, verdict | ½ day |
| 7 | Update CONCH_PATHOMIC_OVERVIEW.md, push to repo | ½ day |
| **Total wall clock** | | **~4–5 days** |

---

## 9. What I will NOT do (commitments against analytic flexibility)

- I will NOT try additional fusion variants (MoE, MoME, APL, etc.) and pick the best one. Only the four cells specified above will be run.
- I will NOT re-tune hyperparameters per cell to "rescue" a null result. All cells use PF baseline hyperparameters.
- I will NOT drop folds where the result is anomalous (e.g., low test set size, censoring imbalance). All 15 folds are included.
- I will NOT change the success threshold post-hoc. +0.012 / p<0.05 is locked.
- I will NOT report only the cells that produced favorable results. All 4 cells × 2 cohorts will be reported regardless.
- If the result is NULL or LOSS, I will write that up as the headline finding and submit to TOMM. No fishing expedition.

---

## 10. Pre-registered probability priors (calibration check)

These are my honest estimates **before** seeing any results, for the purpose of calibration auditing later:

| Outcome | P(KIRC) | P(GBMLGG) |
|---|---|---|
| Clean WIN | 30 % | 15 % |
| Marginal WIN | 25 % | 25 % |
| NULL | 35 % | 50 % |
| LOSS | 10 % | 10 % |

These priors integrate (a) the reviewer's calibrated probabilities, (b) my prior calibration miss on CONCH-Pathomic encoder swap (predicted P(win) = 0.30, observed null × 2 cohorts), (c) the InterSHAP additive-decomposition evidence as a real ceiling, and (d) the cohort-level high variance on KIRC (σ ≈ 0.05).

---

## 11. Commit log convention

The pre-registration commit message will be:
```
PCAF-Pathway pre-registration: 2x2 locked block before any training run
```

After all 8 sweeps complete, the results commit message will be:
```
PCAF-Pathway results (post-registration): {VERDICT_KIRC} / {VERDICT_GBMLGG}
```

The hash of THIS commit (the pre-registration) is the anchor; the results commit's parent will be this one.

---

*End of pre-registration. No training has been run at the time this file is committed.*
