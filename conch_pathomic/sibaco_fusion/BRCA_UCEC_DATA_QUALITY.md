# BRCA + UCEC Data-Quality Log (CONCH feature extraction)

Pre-registration anchor: `7b1256f` (BRCA+UCEC SiBaCo 2×2). Per §8, slides that fail
feature extraction are dropped from BOTH baseline and SiBaCo runs (paired), with the
drop count reported. ≤10% drop is acceptable.

## CONCH path-feature extraction outcome (Phase 8i)

| Cohort | Split cases | Patients with CONCH features | Dropped | Drop % |
|---|---|---|---|---|
| **BRCA** | 957 | **950** | 7 | 0.73% |
| **UCEC** | 480 | **480** | 0 | 0.00% |

Both within the pre-registered ≤10% tolerance.

## Dropped BRCA patients (7) — root cause

All 7 are from TCGA tissue-source-site **OL**, consecutive accessions:

```
TCGA-OL-A5RU, TCGA-OL-A5RV, TCGA-OL-A5RW, TCGA-OL-A5RX,
TCGA-OL-A5RY, TCGA-OL-A5RZ, TCGA-OL-A5S0
```

**Root cause:** these slides are missing scan-resolution metadata. Their Aperio
`ImageDescription` is truncated to `Aperio Image Library v12.1.3 / <dims> (256x256)
J2K/KDU Q=70` — the `AppMag` and `MPP` fields are absent, and there is no
`tiff.XResolution`. openslide reports `mpp-x = MISSING`, `objective-power = MISSING`.

Without microns-per-pixel, TRIDENT cannot compute the correct downsample factor to
reach the locked 20× / 256-px patch grid, so segmentation produced zero patch
coordinates ("Coords not found ... Skipping") and no CONCH features could be
extracted.

**Why not recover by assuming MPP = 0.25 (40×)?** All successfully-processed BRCA
slides are uniformly 0.25 µm/px @ 40× (4-level pyramid [1,4,16,32], ~100k px wide).
The OL slides have a different pyramid (3-level [1,4,8]) and are ~5× smaller in
pixels (~20k px wide), so they may not be 40×. Forcing a guessed MPP risks
extracting CONCH features at the wrong physical scale — a systematic confound in the
exact cohort under test. A 0.73% paired drop is the lower-risk, pre-registered
choice. These slides are excluded rather than recovered.

**Patient TCGA-D8-A3Z6 retained:** its DX1 slide failed for an unrelated transient
reason, but DX2 + DX3 slides extracted successfully, so the patient keeps CONCH
features and is NOT dropped.

## Final 2-modal patient accounting (after omic-modality join)

Building the 2-modal (path+omic) pkls dropped additional patients who have histology
slides + CONCH features but **no molecular profile** in the cBioPortal pan-cancer-atlas
study — they cannot enter a path+omic analysis (missing a required modality).

| Cohort | Split cases | − no CONCH features | − no omic profile | **Final (path+omic)** | Retained |
|---|---|---|---|---|---|
| BRCA | 957 | −7 (OL site, no MPP) | −10 (no cBioPortal molecular) | **940** | 98.2% |
| UCEC | 480 | −0 | −11 (no cBioPortal molecular) | **469** | 97.7% |

BRCA "no molecular" patients (10): TCGA-5L-AAT1, A8-A084, A8-A08F, A8-A08S, A8-A09E,
A8-A09K, AR-A2LL, AR-A2LR, BH-A0B6, D8-A146. Both cohorts remain far within the
pre-registered ≤10% drop tolerance. Drops are paired (apply identically to Cell A'
baseline and SiBaCo).

## Effect on the 15-fold splits

The 7 dropped BRCA patients are removed from every fold (paired across Cell A
baseline + SiBaCo). Final BRCA per-fold case counts will be recomputed when the PF
data pkl is built. UCEC splits are unchanged (no drops).

## Pipeline note (for reproducibility)

A missing Python dependency (`einops_exts`, required by the CONCHv1.5 model loader)
caused the initial TRIDENT launch to crash on both cohorts. After installing it and
clearing stale `.lock` files left by the crash, extraction completed cleanly. The
seg + patch stages do not need this package; only the feature stage does. This is a
runtime-environment note, not a data-quality issue.
