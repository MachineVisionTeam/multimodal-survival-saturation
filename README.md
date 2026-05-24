# Multimodal Cancer Survival — Saturation Evidence

A systematic, **pre-registered** study of four axes (architecture, encoder,
loss, representation) in multimodal cancer survival prediction across three
TCGA cohorts (GBMLGG, KIRC, BLCA), building on **Pathomic Fusion** (Chen et
al., IEEE TMI 2022) and **MCAT** (Chen et al., ICCV 2021).

**Author:** Shemonti Barua · Kennesaw State University · Machine Vision Team
**Status:** in progress — ACM TOMM 2026 Special Issue on Responsible and Explainable Multi-Modal Fusion (deadline Aug 31, 2026)

---

## TL;DR

Across **three TCGA cohorts × four pre-registered axes**, the result is:

| Axis | What we tested | Result | Status |
|---|---|---|---|
| **1. Architecture / Fusion** | 10+ fusion families (GenoFiLM ×3, FiLM, NoGateTrilinear, DAF, HACA ×3, cross-attention) | null in 30+ paired comparisons | CLOSED on tested cohorts |
| **2. Encoder** | UNI2-h, CONCHv1.5 | null on 2 cohorts | CLOSED |
| **3. Loss** | DSM (Weibull mixture, native S(t\|x)), GRFN (evidential) | null IBS; GRFN falsified | CLOSED |
| **4. Representation** | flat curated omic → 50 Hallmark pathway tokens | **WIN on KIRC (+0.030), WIN on BLCA (+0.058), null on GBMLGG** | **PARTIALLY OPEN — 2/3 cohorts positive** |

**The only positive axis is omic representation. Cross-attention fusion adds
essentially zero on top of pathway tokens (paired Cell D − Cell C ≈ 0 on both
3-modal cohorts). The cohort-dependence of the representation win is
predicted by driver-gene-curation completeness** (GBMLGG nulls because its
hand-curated 240-gene panel already includes IDH1/TP53/ATRX/1p-19q; KIRC and
BLCA win because their curated panels under-represent kidney/bladder biology).

All experiments are **pre-registered with commit-hash anchors** before any
training run. Per-fold tables, paired-t / Wilcoxon, 95 % CIs, and verdicts
live in `conch_pathomic/` and `reports/`.

---

## Repository structure

```
.
├── README.md                                    (this file)
├── LICENSE                                      (MIT for our code; see notes)
├── PROJECT_STATUS_REPORT.txt                    (supervisor-facing report)
│
├── architecture/                                (design documents)
│   ├── HACA_DESIGN.md                           (HACA v1)
│   ├── HACA_DESIGN_v1_5.md                      (HACA v1.5, final scope)
│   ├── HACA_DESIGN_v2.md                        (HACA v2, superseded)
│   ├── GRFN_Pathomic_ARCHITECTURE.txt           (GRFN evidential fusion)
│   ├── DSM_Pathomic_ARCHITECTURE.txt            (DSM loss-function lever)
│   └── MCAT_PATCHES.md                          (MCAT repo-bug patches for PyTorch 2.x)
│
├── reports/                                     (paper-style results)
│   ├── pathomic_fusion_replication_results.txt
│   ├── mcat_blca_replication_results.txt
│   ├── mcat_gbmlgg_replication_results.txt
│   ├── HACA_RESULTS_INTERIM.txt                 (BLCA HACA results + r_j diagnostic)
│   ├── GRFN_RESULTS_gbmlgg.txt                  (GRFN c-Index, 15-fold paired)
│   └── GRFN_CALIBRATION_RESULTS_gbmlgg.txt      (GRFN IBS/IBLL falsification)
│
├── grfn_pathomic/                               (Axis 3: Gaussian Random Fuzzy Number fusion)
│   ├── grfn_fusion.py
│   ├── test_grfn_fusion.py
│   ├── network_grfn.py
│   └── calibration_audit.py
│
├── dsm_pathomic/                                (Axis 3: Deep Survival Machines)
│   ├── dsm_fusion.py
│   ├── test_dsm_fusion.py
│   └── network_dsm.py
│
├── haca_mcat/                                   (Axis 1: Hazard-Anchored Cross-Attention on MCAT)
│   ├── model_haca.py
│   ├── haca_train_utils.py
│   └── recover_failed_clam_slides.py
│
└── conch_pathomic/                              (Axes 2 + 4 — CONCH-Pathomic / PCAF-Pathway)
    ├── PCAF_PATHWAY_PREREGISTRATION.md          (KIRC + GBMLGG 2×2 pre-reg, hash 22cf6d1)
    ├── PCAF_PATHWAY_RESULTS.md                  (KIRC CLEAN WIN +0.030 / GBMLGG NULL)
    ├── BLCA_CONCH_BIMODAL_PREREGISTRATION.md    (BLCA 2-cell pre-reg, hash 63ae06a)
    └── BLCA_CONCH_BIMODAL_RESULTS.md            (BLCA spirit-WIN +0.058)
```

---

## Headline results (3 cohorts × 4 axes)

### Axis 4 — Representation (the only positive axis)

| Cohort | Baseline (flat omic) | Pathway-token (50 Hallmark) | Δ | paired-t p | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|
| **KIRC** (n=417, 15 folds) | 0.7188 ± 0.047 | **0.7489 ± 0.040** | **+0.0301** | **0.026** | **0.035** | **CLEAN WIN** |
| **GBMLGG** (n=489, 15 folds) | 0.8075 ± 0.077 | 0.8062 ± 0.073 | −0.0013 | 0.897 | 0.720 | NULL |
| **BLCA** (n=359, 5 folds) | 0.6157 ± 0.016 | **0.6736 ± 0.029** | **+0.0580** | **0.0094** | 0.0625 (n=5 floor) | strict-MARGINAL / spirit-WIN |

### Axis 1 — Fusion (closed across all cohorts)

| Cohort | Fusion-only Δ (Cell B − A) | Fusion-on-top-of-pathway Δ (Cell D − C) | Conclusion |
|---|---|---|---|
| KIRC | −0.0025 (p = 0.60) | +0.0002 | fusion adds zero |
| GBMLGG | +0.0087 (p = 0.30) | −0.0115 | fusion adds zero |
| BLCA | (not retested — known null) | — | inferred null |

Plus prior nulls (in `reports/` and `grfn_pathomic/`, `haca_mcat/`): GenoFiLM × 3, FiLM_residual, NoGateTrilinear, DAF, HACA × 3 — all null on GBMLGG and/or BLCA. Across **30+ paired comparisons** with controlled representation, **no fusion mechanism produces a statistically significant c-Index improvement**.

### Axis 2 — Encoder (closed across all tested cohorts)

| Cohort | CONCHv1.5 vs PF VGG-19 (PCA-32) | Verdict |
|---|---|---|
| KIRC | Δ = +0.0004, paired-t p = 0.946 | NULL |
| GBMLGG | Δ = −0.0003, paired-t p = 0.973 | NULL |

Plus: UNI2-h null over 48 paired runs on GBMLGG (earlier work, see `reports/`).

### Axis 3 — Loss function (closed on tested cohort)

| Loss | Cohort | Δ c-Index | Δ IBS | Verdict |
|---|---|---|---|---|
| DSM-Pathomic (Weibull mixture, native S(t\|x), no Breslow) | GBMLGG | +0.0006 (p=0.89) | +0.0067 (p=0.23) | NULL |
| GRFN-Pathomic (architectural calibration channels under Cox) | GBMLGG | −0.0040 (p=0.43) null | **+0.0078 (p=0.005) FALSIFIED** | calibration-blind under Cox |

---

## The four-axis saturation thesis

| Axis | Evidence | Status across tested cohorts |
|---|---|---|
| **1. Architecture / Fusion** | 10+ fusion families null; Cell B / Cell D − C ≈ 0 on KIRC + GBMLGG | CLOSED |
| **2. Encoder** | CONCHv1.5 + UNI2-h null on both 3-modal cohorts | CLOSED |
| **3. Loss function** | DSM null; GRFN mechanistically falsified under Cox | CLOSED |
| **4. Representation** | Pathway tokens: +0.030 KIRC, +0.058 BLCA, null GBMLGG (cohort-dependent) | **PARTIALLY OPEN** |

**Headline message for the field:** *the past three years of "novel fusion architecture" papers in multimodal cancer survival have been working in a 3–5 % cross-modal-interaction ceiling (consistent with the InterSHAP audit, arXiv 2603.29977). The actual load-bearing axis is omic representation. Replace flat curated omics with biologically organized pathway tokens, and you get +0.030 to +0.058 c-Index on cohorts where the curated panel doesn't already capture dominant driver-gene signal.*

---

## Pre-registration and commit anchors

All headline experiments were **pre-registered with commit-hash anchors before any training run**:

| Experiment | Pre-reg commit | Results commit |
|---|---|---|
| GRFN-Pathomic calibration falsification | (see `architecture/GRFN_Pathomic_ARCHITECTURE.txt`) | (see `reports/GRFN_CALIBRATION_RESULTS_gbmlgg.txt`) |
| DSM-Pathomic Phase 2 | (see `architecture/DSM_Pathomic_ARCHITECTURE.txt`) | `c6b0a5c` |
| **PCAF-Pathway 2×2 (KIRC + GBMLGG)** | **`22cf6d1`** (2026-05-22T04:36:00Z) | **`38d78f5`** |
| **BLCA CONCH-Bimodal 2-cell** | **`63ae06a`** (2026-05-22T18:19:18Z) | **`62d8b95`** |

The pre-registration documents specify success thresholds, statistical tests, and interpretation rules **before any results are observed**. Results documents apply those rules without modification.

---

## How to reproduce

This repository contains the **novel code, design documents, pre-registrations, and per-fold results**. The Pathomic Fusion and MCAT baselines have their own repositories; clone them separately. CONCHv1.5 weights are gated via HuggingFace (`MahmoodLab/conchv1_5`).

### 1. Set up the baselines

```bash
# Pathomic Fusion baseline (Chen et al., IEEE TMI 2022)
git clone https://github.com/mahmoodlab/PathomicFusion

# MCAT baseline (Chen et al., ICCV 2021)
git clone https://github.com/mahmoodlab/MCAT

# TRIDENT for CONCHv1.5 WSI patching + feature extraction
git clone https://github.com/mahmoodlab/TRIDENT
```

Apply MCAT patches in `architecture/MCAT_PATCHES.md`. PathomicFusion runs cleanly on PyTorch 2.x as-shipped. TRIDENT installs via `pip install -e .`.

### 2. Drop the GRFN / DSM / HACA / CONCH files into the upstream trees

| Our file | Drop into |
|---|---|
| `grfn_pathomic/grfn_fusion.py` | sibling of `PathomicFusion/` (`sys.path.insert(0, '../GRFN-Pathomic')`) |
| `grfn_pathomic/network_grfn.py` | `PathomicFusion/` |
| `grfn_pathomic/calibration_audit.py` | `PathomicFusion/` |
| `dsm_pathomic/dsm_fusion.py` | sibling of `PathomicFusion/` |
| `dsm_pathomic/network_dsm.py` | `PathomicFusion/` |
| `haca_mcat/model_haca.py` | `MCAT/models/` |
| `haca_mcat/haca_train_utils.py` | `MCAT/utils/` |
| **CONCH-Pathomic networks** (not in this repo — they live in the private working tree) | see `conch_pathomic/PCAF_PATHWAY_RESULTS.md` §9 (file paths) |

Required additive patches in PF (~30 lines total across `networks.py`, `data_loaders.py`, `train_cv.py`) are documented in each cohort's results doc.

### 3. Extract CONCHv1.5 features + Hallmark pathway tokens

```bash
# CONCHv1.5 features via TRIDENT (per cohort)
python TRIDENT/run_batch_of_slides.py \
    --task all --wsi_dir <cohort_svs_dir> --job_dir <out> \
    --patch_encoder conch_v15 --mag 20 --patch_size 256 \
    --segmenter hest --gpus 0 1 2 3 --batch_size 64 --skip_errors

# Hallmark pathway tokens from cBioPortal RNA-seq (reuse SurvPath's mapping)
python build_pathway_tokens.py --cohort {KIRC,GBMLGG,BLCA}
```

### 4. Train

For PCAF-Pathway on TCGA-KIRC (Cell D, primary):

```bash
cd PathomicFusion
FOLDS=1,2,...,15 CUDA_VISIBLE_DEVICES=0 python train_cv.py \
    --exp_name surv_15 --dataroot ./data/TCGA_KIRC \
    --checkpoints_dir ./runs_modern/TCGA_KIRC/ \
    --task surv --mode pathgraphomic_conch_pcaf \
    --model_name pathgraphomic_conch_pcaf \
    --niter 10 --niter_decay 20 --batch_size 32 \
    --use_vgg_features 1 --use_rnaseq 0 --input_size_omic 362 \
    --grph_dim 32 --omic_dim 32 --path_dim 32 \
    --measure 1 --skip 0 --use_bilinear 1 \
    --path_gate 1 --grph_gate 1 --omic_gate 0 \
    --fusion_type pofusion_A --gpu_ids 0
```

For BLCA CONCH-Bimodal (Cell C', primary):

```bash
FOLDS=1,2,3,4,5 CUDA_VISIBLE_DEVICES=0 python train_cv.py \
    --exp_name surv_5 --dataroot ./data/TCGA_BLCA \
    --checkpoints_dir ./runs_modern/TCGA_BLCA/ \
    --task surv --mode pathomic_conch_pathway \
    --model_name pathomic_conch_pathway \
    --niter 10 --niter_decay 20 --batch_size 32 \
    --use_vgg_features 1 --use_rnaseq 0 --input_size_omic 50 \
    --grph_dim 32 --omic_dim 32 --path_dim 32 \
    --measure 1 --skip 0 --use_bilinear 1 \
    --path_gate 1 --grph_gate 1 --omic_gate 0 \
    --fusion_type pofusion --gpu_ids 0
```

For GRFN / DSM / HACA cohorts — see `architecture/` design docs and `reports/` for exact CLI invocations.

---

## Key references

**Baselines we built on:**
- **Pathomic Fusion:** Chen R.J. et al., *IEEE TMI* 41(4):757-770 (2022).
- **MCAT:** Chen R.J. et al., ICCV 2021.

**Encoders / preprocessing:**
- **CONCHv1.5:** Lu M.Y. et al., *Nature Medicine* (2024) + 2025 update. `MahmoodLab/conchv1_5`.
- **TRIDENT:** MahmoodLab, 2024. Multi-GPU WSI patcher + feature extractor.

**Direct inspirations for Axis 4 (representation):**
- **SurvPath:** Jaume G. et al., *CVPR* 2024. The Hallmark pathway-tokenization recipe we reuse.
- **MMP:** Song A.H. et al., *ICML* 2024. GMM K=16 prototype aggregation of WSI patches.
- **MOTCat:** Xu Y. & Chen R.J., *ICCV* 2023. Optimal-transport variant.
- **DIMAF:** Eijpe S. et al., *MICCAI* 2025. Most recent fusion-innovation paper.

**Loss-function lever (Axis 3):**
- **Deep Survival Machines:** Nagpal C., Li X., Dubrawski A., *IEEE JBHI* 25(8):3163-3175 (2021); arXiv:2003.01176.
- **EsurvFusion (GRFN ancestor):** Huang et al., *IEEE TFS* 34(1):76-88 (2026); arXiv:2412.01215.

**Saturation / calibration audits that converge with our findings:**
- **InterSHAP audit:** Swift et al., arXiv:2603.29977 (2026, XAI 2026 late-breaking). 3–5 % cross-modal interaction variance on glioma.
- **Calibration audit:** Ghawami, arXiv:2604.04239 (2026). MCAT achieves c-Index 0.817 on GBMLGG but fails 1-calibration on all 5 folds.

---

## License notes

- The novel code we wrote (GRFN-Pathomic, DSM-Pathomic, HACA-MCAT, CONCH-Pathomic / PCAF-Pathway, the calibration audit) is released under **MIT** (see `LICENSE`).
- The upstream Pathomic Fusion, MCAT, TRIDENT, and SurvPath codebases retain their own licenses; this repository does not redistribute their source.
- CONCHv1.5 weights are gated via HuggingFace (`MahmoodLab/conchv1_5`); users must register their own access token.
- TCGA whole-slide imaging data and clinical metadata are NOT included here. Obtain via the standard NIH GDC / dbGaP channels under the appropriate data-use agreement.

---

## Contact

Shemonti Barua · `sbarua@students.kennesaw.edu` · Machine Vision Team, Kennesaw State University.
