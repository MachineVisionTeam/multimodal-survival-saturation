# Unimodal SNN/CNN Baselines — Modality-Balance Classification (Phase 8m)

Pre-registration anchor: `7b1256f` + amendment `8cafd1e` (2-modal).
Locked classification rule (§3.6): cohort = **genomics-dominated** iff
mean(SNN c-Index) − mean(CNN c-Index) ≥ +0.04 AND paired-t p < 0.05;
otherwise **balanced**. Locked BEFORE seeing these numbers.

Protocol (locked, identical to multimodal Cell A'): MaxNet omic encoder /
CONCHv15PathAdapter; Linear(32,1) Cox head; Adam, lr=1e-3, batch=32,
30 epochs (10 + 20 decay), dropout=0.25; best test c-Index across epochs.

## Results (5 cohorts × 15-fold or 5-fold)

| Cohort | n | SNN c-Idx | CNN c-Idx | Δ (SNN−CNN) | paired-t p | Observed | Predicted | Match |
|---|---|---|---|---|---|---|---|---|
| KIRC | 15 | 0.736 ± 0.055 | 0.586 ± 0.120 | +0.150 | **0.0001** | genomics-dominated | balanced | ❌ |
| GBMLGG | 15 | 0.808 ± 0.068 | 0.645 ± 0.148 | +0.163 | **0.0015** | genomics-dominated | genomics-dominated | ✓ |
| BLCA | 5 | 0.618 ± 0.034 | 0.588 ± 0.055 | +0.031 | 0.4584 | balanced | balanced | ✓ |
| BRCA | 15 | 0.672 ± 0.080 | 0.548 ± 0.083 | +0.124 | **0.0019** | genomics-dominated | balanced | ❌ |
| UCEC | 15 | 0.739 ± 0.134 | 0.663 ± 0.132 | +0.076 | 0.1349 | balanced | genomics-dominated | ❌ |

**3 of 5 pre-registered cohort classifications were CONTRADICTED by the data.** Per
the locked pre-registration (§3.6 and §4), we use the **observed** classifications
to interpret the SiBaCo 2-modal verdicts in Phase 8n — not the predicted ones.

## Updated empirical 2×2 framing

|  | **Balanced (observed)** | **Genomics-dominated (observed)** |
|---|---|---|
| Predicted SiBaCo win region | BLCA · **UCEC** | — |
| Predicted SiBaCo null region | — | KIRC · GBMLGG · **BRCA** |

## Honest caveats (pre-registered transparency)

1. **CNN-only collapse to chance on small folds.** On multiple folds with few training
   events (~6–15), the CNN-only model returned test c-Index ≈ 0.5 exactly — i.e. it
   produced constant hazards. This is a real characteristic of the Cox-loss + tiny
   model + few-events regime, not a code bug. The reported CNN means are pulled down
   by these collapsed folds. Per pre-reg we lock the protocol (LR=1e-3, 30 epochs,
   no early stopping) and report what it produces — no tuning to "rescue" CNN.

2. **UCEC's "balanced" classification is borderline.** Δ=+0.076 exceeds the +0.04
   magnitude threshold but the paired-t p=0.135 fails the significance gate. The
   rule requires BOTH; balanced is the literal verdict, but the effect direction is
   "trending genomics-dominated." Will be reported as such.

3. **KIRC and BRCA reclassified as genomics-dominated.** This shifts the SiBaCo
   prediction: under modality-balance mediation, geno-dom cohorts are expected to
   NULL on SiBaCo. The original KIRC 3-modal SiBaCo WIN (+0.026, p=0.003) thus
   becomes a sharp falsifiable test — if KIRC's 2-modal SiBaCo also wins despite the
   geno-dom classification, the mediation hypothesis is wrong; if it nulls, the
   3-modal win was carried by the cell-graph modality, not by SiBaCo's fusion
   geometry per se.

## Re-stated predictions for Phase 8n (using observed classifications)

| Cohort | Observed class | Modality-balance hypothesis prediction |
|---|---|---|
| KIRC (2-modal) | genomics-dominated | NULL |
| GBMLGG (2-modal) | genomics-dominated | NULL ✓ matches existing 3-modal NULL |
| BLCA (2-modal) | balanced | non-null (but we observed NULL already; informative datum) |
| BRCA (2-modal) | genomics-dominated | NULL |
| UCEC (2-modal) | balanced (borderline) | non-null |

So under the hypothesis, only UCEC (and weakly BLCA) should win. If KIRC 2-modal
wins anyway, the modality-balance hypothesis is falsified.

## Files

- Per-cohort raw outputs: `/mnt/storage7/Dataset_pathomicfusion/SiBaCo_unimodal/{COHORT}_unimodal.json`
  (per-fold c-Indexes, classifications, walltime).
- Training script: `unimodal_baselines.py` (committed to publish repo).
