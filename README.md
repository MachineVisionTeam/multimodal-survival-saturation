# Multimodal Cancer Survival — Saturation Evidence

A systematic study of fusion-architecture, encoder, and loss-function levers
in multimodal cancer survival prediction (TCGA-GBMLGG, TCGA-KIRC, TCGA-BLCA),
building on the **Pathomic Fusion** (Chen et al., IEEE TMI 2022) and **MCAT**
(Chen et al., ICCV 2021) baselines.

**Author:** Shemonti Barua · Kennesaw State University · Machine Vision Team
**Status:** in progress — TOMM 2026 Special Issue on Responsible and Explainable Multi-Modal Fusion (deadline Aug 31, 2026)

---

## TL;DR

I tested **13 distinct fusion / loss / attention interventions** on top of two
locked baselines (PF, MCAT). **No intervention has produced a statistically
significant improvement in concordance index over its baseline.** A
pre-registered calibration-improvement hypothesis was cleanly *falsified*
(paired-t p = 0.005). The pattern is consistent with the published InterSHAP
audit (arXiv 2603.29977, 2026) showing only ~3–5 % cross-modal interaction
variance on glioma survival. The project frames this as a **three-axis
saturation map**: architecture (closed), encoder (closed), loss function
(in progress with DSM-Pathomic).

---

## Repository structure

```
.
├── README.md                                 (this file)
├── LICENSE                                   (MIT for our code; see notes)
├── PROJECT_STATUS_REPORT.txt                 (full project-status report,
│                                              supervisor-facing)
│
├── architecture/                             (design documents)
│   ├── HACA_DESIGN.md                        (v1 — narrow scope)
│   ├── HACA_DESIGN_v1_5.md                   (v1.5 — final HACA scope)
│   ├── HACA_DESIGN_v2.md                     (v2 — superseded by v1.5)
│   ├── GRFN_Pathomic_ARCHITECTURE.txt        (GRFN evidential fusion design)
│   ├── DSM_Pathomic_ARCHITECTURE.txt         (DSM loss-function lever design)
│   └── MCAT_PATCHES.md                       (7 deprecation/repo-bug patches
│                                              required to run MCAT on PyTorch 2.x)
│
├── reports/                                  (paper-style results)
│   ├── pathomic_fusion_replication_results.txt
│   ├── mcat_blca_replication_results.txt
│   ├── mcat_gbmlgg_replication_results.txt
│   ├── HACA_RESULTS_INTERIM.txt              (BLCA HACA results + r_j
│   │                                          collapse diagnostic)
│   ├── GRFN_RESULTS_gbmlgg.txt               (GRFN c-Index, 15-fold paired)
│   └── GRFN_CALIBRATION_RESULTS_gbmlgg.txt   (GRFN IBS/IBLL falsification +
│                                              mechanistic diagnosis)
│
├── grfn_pathomic/                            (Gaussian Random Fuzzy Number fusion)
│   ├── grfn_fusion.py                        (GRFNFusion module — pure PyTorch)
│   ├── test_grfn_fusion.py                   (T1-T8 smoke tests)
│   ├── network_grfn.py                       (PF integration wrapper)
│   └── calibration_audit.py                  (IBS / IBLL / Platt-scaled PF
│                                              evaluation script)
│
├── dsm_pathomic/                             (Deep Survival Machines mixture loss)
│   ├── dsm_fusion.py                         (DSMFusion module — pure PyTorch)
│   ├── test_dsm_fusion.py                    (T1-T8 smoke tests)
│   └── network_dsm.py                        (PF integration wrapper)
│
└── haca_mcat/                                (Hazard-Anchored Cross-Attention on MCAT)
    ├── model_haca.py                         (HACA_Surv extension of MCAT_Surv)
    ├── haca_train_utils.py                   (HACA training loop with AAS
    │                                          warmup schedule + r_j logging)
    └── recover_failed_clam_slides.py         (helper that recovered 20 brain WSIs
                                               whose default CLAM segmentation failed)
```

---

## Headline results

|  | TCGA-GBMLGG | TCGA-KIRC | TCGA-BLCA |
|---|---|---|---|
| **Pathomic Fusion** (locked) | 0.8174 ± 0.072 | 0.7184 ± 0.051 | — |
| **MCAT** (locked) | 0.820 ± 0.020 | — | 0.632 ± 0.042 |
| GenoFiLM (pre_ln / residual / plain) | null × 3 | — | — |
| FiLM_residual / NoGateTrilinear | null | — | — |
| UNI2-h encoder upgrade | −0.0072 (48 paired runs) | — | — |
| DAF (signed disagreement) | +0.0018 null | — | — |
| HACA (3 variants) | — | — | null (best Δ = −0.006) |
| **GRFN-Pathomic** (architectural calibration channels under Cox) | Δ c-Index = −0.0040 null; **Δ IBS = +0.0078 (p=0.005, falsified)** | (pending) | — |
| **DSM-Pathomic** (loss-function lever; Weibull mixture, native S(t\|x), no Breslow) | Δ c-Index = +0.0006 (p=0.89) null; **Δ IBS = +0.0067 (p=0.23) null** | (pending) | — |

Per-fold tables, paired t-tests, Wilcoxon signed-rank, 95% CIs, and
mechanistic diagnoses live in the `reports/` directory.

---

## The three-axis saturation thesis

| Axis | Evidence so far | Status |
|---|---|---|
| **Architecture** | 9 fusion families nulled on GBMLGG, 3 on BLCA | CLOSED on single cohort |
| **Encoder** | UNI2-h null over 48 paired runs on GBMLGG | CLOSED |
| **Loss function** | DSM-Pathomic Weibull mixture: native S(t\|x) IBS 0.1038 ± 0.026 vs PF 0.0971 ± 0.010 (p=0.23 null) | **CLOSED** — completed 2026-05-19 |

**All three pre-specified axes are now empirically closed on TCGA-GBMLGG.**
The KIRC cross-cohort confirmation (Phase 3) is the next experiment.

**Cox is calibration-blind.** GRFN-Pathomic computes calibrated channels
(σ²_f and h_f) but they are gradient-free when trained with Cox partial
likelihood. The Breslow estimator used for IBS evaluation consumes only
μ_f. The architectural calibration apparatus is therefore mathematically
guaranteed to be discarded, which is exactly what the Phase 2c audit
measured. The natural next experiment — DSM-Pathomic — keeps the PF
encoders frozen and replaces the loss with a calibrated parametric
mixture likelihood; this is the only remaining lever in the three-axis
map that has not been tested.

---

## How to reproduce

This repository contains **only the novel code I wrote** plus design
documents and results. The Pathomic Fusion and MCAT baselines have their
own repositories and licenses; you must clone them separately.

### 1. Set up the baselines

```bash
# Pathomic Fusion baseline (Chen et al., IEEE TMI 2022)
git clone https://github.com/mahmoodlab/PathomicFusion

# MCAT baseline (Chen et al., ICCV 2021)
git clone https://github.com/mahmoodlab/MCAT
```

Apply the deprecation/repo-bug patches in `architecture/MCAT_PATCHES.md`
to the MCAT clone. The PathomicFusion clone needs no such patches; it runs
cleanly on PyTorch 2.x as-shipped.

### 2. Drop the GRFN / DSM / HACA files into the upstream trees

| Our file | Drop into |
|---|---|
| `grfn_pathomic/grfn_fusion.py` | a sibling folder of `PathomicFusion/` (import path is `sys.path.insert(0, '../GRFN-Pathomic')`) |
| `grfn_pathomic/network_grfn.py` | `PathomicFusion/` (sits beside `networks.py`) |
| `grfn_pathomic/calibration_audit.py` | `PathomicFusion/` |
| `dsm_pathomic/dsm_fusion.py` | sibling folder (analogous to GRFN) |
| `dsm_pathomic/network_dsm.py` | `PathomicFusion/` |
| `haca_mcat/model_haca.py` | `MCAT/models/` |
| `haca_mcat/haca_train_utils.py` | `MCAT/utils/` |

Three small additive patches are required in the upstream codebases:

1. In `PathomicFusion/networks.py`: add `elif` branches in `define_net()`
   for `pathgraphomic_grfn` and `pathgraphomic_dsm` (~10 lines).
2. In `PathomicFusion/data_loaders.py`: alias the two new modes to
   `pathgraphomic` in both `__getitem__` methods (~4 lines).
3. In `PathomicFusion/train_test.py`: dispatch to
   `model.compute_dsm_loss(...)` when the model is DSM (~10 lines).
4. In `MCAT/main.py`, `MCAT/utils/core_utils.py`, `MCAT/utils/utils.py`:
   add HACA mode + argparse flags. See `architecture/MCAT_PATCHES.md` for
   the exact diffs.

Run the smoke tests:

```bash
python grfn_pathomic/test_grfn_fusion.py        # 8/8 checks
python dsm_pathomic/test_dsm_fusion.py          # 8/8 checks
```

### 3. Train

For GRFN-Pathomic on TCGA-GBMLGG:

```bash
cd PathomicFusion
FOLDS=1,2,3,4,5,6,7,8 CUDA_VISIBLE_DEVICES=0 python train_cv.py \
    --exp_name surv_15_rnaseq --dataroot ./data/TCGA_GBMLGG \
    --checkpoints_dir ./runs_modern/TCGA_GBMLGG/ \
    --task surv --mode pathgraphomic_grfn \
    --model_name pathgraphomic_grfn \
    --niter 10 --niter_decay 20 --batch_size 32 \
    --use_vgg_features 1 --use_rnaseq 1 --input_size_omic 320 \
    --grph_dim 32 --omic_dim 32 --path_dim 32 \
    --measure 1 --skip 0 --use_bilinear 1 \
    --path_gate 1 --grph_gate 1 --omic_gate 0 --gpu_ids 0
```

For DSM-Pathomic, change `--mode pathgraphomic_dsm` and `--model_name
pathgraphomic_dsm`.

For HACA on MCAT (BLCA):

```bash
cd MCAT
FOLDS=1,2,3,4,5 CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root_dir <path-to-BLCA> --task tcga_blca_survival \
    --mode coattn --model_type haca \
    --haca_lambda 1.0 --haca_warmup_epochs 5 --haca_aux_weight 0.0
```

### 4. Run the calibration audit

After GRFN-Pathomic and the locked PF baseline have both trained on all
15 folds, the audit script computes IBS / IBLL paired:

```bash
python grfn_pathomic/calibration_audit.py
```

---

## Key references

- **Pathomic Fusion (baseline):** Chen R.J. et al., *IEEE TMI*  41(4):757-770 (2022).
- **MCAT (baseline):** Chen R.J. et al., ICCV 2021.
- **Deep Survival Machines:** Nagpal C., Li X., Dubrawski A., *IEEE JBHI* 25(8):3163-3175 (2021); arXiv:2003.01176.
- **EsurvFusion (GRFN ancestor):** Huang et al., *IEEE TFS* 34(1):76-88 (2026); arXiv:2412.01215.
- **InterSHAP audit:** Swift et al., arXiv:2603.29977 (2026).
- **Calibration audit:** Ghawami, arXiv:2604.04239 (2026).

---

## License notes

- The novel code I wrote (GRFN-Pathomic, DSM-Pathomic, HACA-MCAT, the
  calibration audit) is released under **MIT** (see `LICENSE`).
- The upstream Pathomic Fusion and MCAT codebases retain their own
  licenses; this repository does not redistribute their source code.
- TCGA whole-slide imaging data and clinical metadata are NOT included
  here. Obtain them via the standard NIH GDC / dbGaP channels under
  the dbGaP data-use agreement appropriate to your institution.

---

## Contact

Shemonti Barua · `sbarua@students.kennesaw.edu` · Machine Vision Team, Kennesaw State University.
