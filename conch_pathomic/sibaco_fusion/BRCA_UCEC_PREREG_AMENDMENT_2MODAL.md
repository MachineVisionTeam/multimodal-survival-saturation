# Pre-Registration Amendment — BRCA+UCEC SiBaCo: 3-modal → 2-modal

**Amendment date (UTC):** `2026-05-28T00:00:00Z`
**Author:** Shemonti Barua · Machine Vision Team, Kennesaw State University
**Amends:** `BRCA_UCEC_SIBACO_PREREGISTRATION.md` (parent pre-reg commit `7b1256f`)
**Status:** committed BEFORE any BRCA/UCEC/KIRC-2modal/GBMLGG-2modal SiBaCo training run.
No 2-modal training has been launched for any cohort as of this commit.

---

## 1. What changed and why

The parent pre-registration (§3.2) locked BRCA and UCEC as **3-modal** (CONCH path +
cell-graph + curated omic), to mirror the 3-modal TCGA-KIRC SiBaCo win exactly.

**During Phase 8j we established that faithful 3-modal cell graphs cannot be built for
BRCA or UCEC.** Evidence (recorded 2026-05-28):

- Pathomic Fusion released **only the output cell graphs** for KIRC + GBMLGG (the
  `pt_bi/*.pt` PyG files), **not the CPC encoder weights** used to produce them.
- PF's graph-construction notebook (`CellGraph/Graph Construction.ipynb`) hard-codes
  `ckpt_dir = './pretrained_models/cpc.pt'` — a Contrastive-Predictive-Coding encoder
  checkpoint that **does not exist anywhere on disk and was never publicly released**.
- The pipeline also depends on `pyflann` (a Python-2-era FLANN binding) and
  `torch_geometric`, neither installed, the former being effectively unmaintained on
  modern Python.

**Consequence:** any cell graphs we built for BRCA/UCEC would use a *reconstructed,
different* nuclei encoder. Their node-feature space would differ systematically from
KIRC/GBMLGG's PF-released graphs. Putting non-comparable graph features into the same
2×2 introduces a **cross-cohort confound that is scientifically worse than having no
graph modality at all** — a null (or win) could be attributed to the graph-encoder
mismatch rather than to SiBaCo's fusion behaviour.

## 2. The amendment

All cohorts in the SiBaCo modality-balance 2×2 are run as **2-modal** SiBaCo
(CONCH path + curated omic), so that every modality is built with an **identical
pipeline across all cohorts** — zero cross-cohort confound.

| Cohort | Parent pre-reg (locked) | Amended |
|---|---|---|
| BRCA | 3-modal | **2-modal** (CONCH path + curated omic) |
| UCEC | 3-modal | **2-modal** (CONCH path + curated omic) |
| KIRC | (3-modal, already run) | **also re-run 2-modal** for matched comparison |
| GBMLGG | (3-modal, already run) | **also re-run 2-modal** for matched comparison |
| BLCA | (2-modal, already run) | unchanged — already 2-modal |

The 2-modal SiBaCo operator is `PathomicCONCHBimodalSiBaCoNet` (n_modalities=2),
**byte-identical to the module already used for BLCA** (parent commit a224dc0). The
comparator is the matched 2-modal Cell A' (CONCH path + omic + BilinearFusion).

## 3. What is UNCHANGED (still locked from parent pre-reg)

- SiBaCo hyperparameters: K=64, ε=0.05, T=20, τ_init=1.0, skip=True.
- Verdict thresholds: CLEAN WIN = Δ ≥ +0.012 AND paired-t p<0.05 AND Wilcoxon p<0.05.
- Modality-balance classification rule (§3.6): genomics-dominated iff
  mean(SNN c-Index) − mean(CNN c-Index) ≥ +0.04 with paired-t p<0.05 over folds.
  (CNN-only = CONCH path adapter; SNN-only = omic MaxNet. Graph plays no role in
  either unimodal baseline, so this rule is unaffected by dropping the graph branch.)
- Pre-registered priors per cohort (§2) and the 2×2 interpretation rules (§4).
- Curated omic panels (BRCA 126 genes / 378-d, UCEC 130 genes / 390-d), committed `4f54912`.
- 15-fold splits, seed=42 (committed `b02de21`), minus the 7 unrecoverable BRCA OL
  patients (documented `f703791`).

## 4. The 2×2 framing under this amendment

**Primary 2×2 — fully matched, all 2-modal (CONCH path + omic):**

|  | Balanced modality | Genomics-dominated |
|---|---|---|
| SiBaCo win expected | KIRC (2-modal, re-run) · BRCA (predict) | — |
| SiBaCo null expected | BLCA (2-modal, done: NULL) | GBMLGG (2-modal, re-run) · UCEC (predict) |

Every cell is built with identical path + omic pipelines → internally consistent.

**Motivating context (reported separately, NOT part of the matched 2×2):**
The original 3-modal results (KIRC CLEAN WIN +0.026 p=0.003; GBMLGG NULL) using
PF-released graphs. Clearly labelled as 3-modal with PF-provided graphs. This is what
motivated the saturation/replication question; the matched 2-modal 2×2 is what tests
it without confound.

## 5. New hypothesis note (honest, pre-registered)

Because the original KIRC win was **3-modal**, re-running KIRC at 2-modal is itself a
test: if KIRC's SiBaCo advantage **persists** at 2-modal, the win is not graph-dependent
and the matched 2×2 is well-founded. If KIRC's advantage **disappears** at 2-modal
(as BLCA's 2-modal already nulled), that indicates the cell-graph modality was
load-bearing for the SiBaCo win — itself a reportable finding, and it would reframe the
2×2 around the 2-modal regime. Either outcome is reported as-is; no post-hoc
re-interpretation.

## 6. Pre-commit checklist (audit)

- [ ] No 2-modal SiBaCo / Cell A' / SNN / CNN training run has been launched for BRCA,
      UCEC, KIRC-2modal, or GBMLGG-2modal as of this commit.
- [ ] The 2-modal SiBaCo module is unchanged from parent commit a224dc0.
- [ ] No cell-graph construction was performed for BRCA or UCEC (3-modal abandoned).
- [ ] CONCH path features exist (BRCA 950 patients, UCEC 480 patients); MMP K=16
      aggregation to 12288-d signatures is the next step (not yet run).

## 7. Anchor

**This amendment committed at:** (filled by git hash after commit)
**Parent pre-reg:** `7b1256f`
**No 2-modal training has been performed as of this commit.**
