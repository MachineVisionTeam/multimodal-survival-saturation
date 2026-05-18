# HACA v1.5 — Hazard-Anchored Cross-Attention (final scope)
## Design document — 2026-05-17
## Supersedes: v1 (HACA_DESIGN.md), v2 (HACA_DESIGN_v2.md)
## (Both kept on disk as historical record; this is the design we will build)

---

## 0. How we got here (one paragraph)

v1 was scoped as one biased-attention term — clean but thin. v2 added
PRCA (per-pathway priors), HDL (distillation loss), AAS (warmup
schedule) under a Bayesian frame to chase a "Strong-tier" standalone
contribution. External critique correctly identified that PRCA / HDL /
AAS are a multiplicity choice + an aux loss + a schedule rather than
three coordinated mechanisms, and that the Bayesian frame only directly
justifies the log-additive bias form. v1.5 takes the one structural
addition from v2 that addresses a real failure mode (AAS warmup) and
the one rhetorical asset (Bayesian frame in the writeup), drops the
rest, and **positions HACA as a single deep intervention inside an
existing saturation paper** rather than a new standalone contribution.

---

## 1. Scope decision

**HACA v1.5 is one row in the "When Fusion Is Saturated" paper, not the
headline of a new paper.**

The saturation paper already has, locked and verified:
- ~270 paired-seed controlled runs across 7 architectural interventions
- InterSHAP-measured 4% interaction-variance ceiling on glioma
- Falsified GenoFiLM (3 mod_schemes, all negative)
- Falsified UNI2-h encoder upgrade (Delta = -0.0072, 48 paired runs)
- DAF with full signed-disagreement ablation (Delta = +0.0018, within noise)
- Pathomic Fusion replication within published std (0.8174 on GBMLGG)
- KIRC baseline anchor (0.7184)
- MCAT replication on two cohorts (this work — BLCA 0.632, GBMLGG 0.820)

HACA v1.5 is the **deepest intervention in this evidence ladder** — it
attempts to break saturation by reaching into the attention mechanism
itself rather than the fusion stack. Whether positive or null, that
result is publishable as the most informative entry in the paper's main
comparison table.

---

## 2. Architecture — single deep mechanism

### 2.1 Inputs (identical to MCAT)
```
x_path  : (N, 1024)       ResNet-50 truncated features (locked baseline)
x_omic  : 6 × (P_k,)      per-pathway gene vectors
```

### 2.2 Feature projection (identical to MCAT)
```
h_path_bag = wsi_net(x_path).unsqueeze(1)              shape (N, 1, 256)
h_omic_bag = stack([sig_networks[k](x_omic[k])])       shape (6, 1, 256)
                                  .unsqueeze(1)
```

### 2.3 Per-patch hazard head (NEW — single shared MLP)

```
patch_hazard_mlp:  Linear(256, 64) -> ReLU -> Dropout(0.25) -> Linear(64, 1)

r_j = sigmoid(patch_hazard_mlp(h_path_bag[j]))         in (0, 1)
log_r = log(r_j + eps)                                 eps = 1e-3
```

**Single shared head across all 6 pathway queries.** This is the
critique's "K=1, not K=6" choice. If v1.5 shows promise, K=6 (PRCA)
becomes the journal-extension contribution. If v1.5 is null, K=6 was
unlikely to rescue it.

~17K new parameters on MCAT's 4.5M backbone (0.4%).

### 2.4 Hazard-anchored co-attention (NEW)

Build a (6, N) attention bias, broadcast the shared `log_r` across all
6 pathway queries:

```
attn_bias[k, j] = lambda(epoch) · log_r[j]              for all k

h_path_coattn, A_coattn = coattn(
    query      = h_omic_bag,                            (6, 1, 256)
    key        = h_path_bag,                            (N, 1, 256)
    value      = h_path_bag,                            (N, 1, 256)
    attn_mask  = attn_bias                              (6, N) — NEW
)
```

After softmax (equivalent to the Bayesian re-derivation):
```
A_HACA[k, j] = softmax_j(  Q[k] · K[j] / sqrt(d)  +  lambda · log r_j  )
            propto  softmax_j(Q · K / sqrt(d))  ·  r_j^lambda
            =  p(patch_j | pathway_k)  ·  pi(j)^lambda    (Bayesian)
```

The Bayesian frame justifies *this functional form* (additive log-prior
rather than multiplicative post-softmax or additive linear). The frame
appears in the Methods writeup; per the critique, we don't oversell it
as "motivating multiple mechanisms" because it doesn't.

### 2.5 AAS — Adaptive Anchoring Schedule (NEW, kept from v2)

At epoch 0, `r_j` is random. Anchoring attention on random priors hurts.
Warmup lambda linearly from 0 to lambda_final over the first N epochs:

```
lambda(epoch) = lambda_final · min(1, epoch / N_warmup)
```

Defaults: `lambda_final = 1.0`, `N_warmup = 5` (out of 20 total).

This is the one v2 addition that addresses a *real* v1 failure mode
without adding conceptual baggage. Extra cost: 1 hyperparameter,
~5 LOC, one extra ablation row (no-warmup vs warmup).

### 2.6 Downstream (identical to MCAT)
```
h_path_trans = path_transformer(h_path_coattn)
... path attention head, omic transformer, omic attention head, fusion ...
logits   = classifier(h)                                shape (1, 4)
hazards  = sigmoid(logits)
S        = cumprod(1 - hazards)
```

### 2.7 Loss (identical to MCAT — no aux head, no distillation)
```
L = NLL_Surv(hazards, S, Y, c)
```

**Dropped from v2: HDL distillation loss.** Per critique, having
three losses on different versions of the same patch hazard is
confusing and the distillation isn't motivated by the Bayesian frame.

**Dropped from v1: optional aux NLL head.** Removing this too — keep
the loss path identical to MCAT so that any difference is attributable
purely to the attention-bias intervention. (We can re-add as an
ablation if needed; default off.)

---

## 3. Position in the saturation paper

HACA v1.5 becomes one row in the main comparison table:

```
Method                                BLCA c-Index    GBMLGG c-Index
------------------------------------  --------------  --------------
Pathomic Fusion (replication)         ~0.59           0.817
GenoFiLM (pre_ln)                     -0.0028 vs PF   -0.0044 vs PF
GenoFiLM (residual)                   -0.0051         -
FiLM_residual replacement             +0.0005         -
UNI2-h encoder upgrade                -              -0.0072
DAF (signed disagreement)             +0.0018         -
MCAT (replication)                    0.632           0.820
HACA v1.5 (this)                      <result>        <result>
```

Whatever HACA v1.5 produces, it is the **deepest intervention** in this
ladder — the only one that reaches inside the attention mechanism.
A positive Delta supports "attention supervision can break saturation."
A null Delta supports "saturation holds even at the attention level —
the 4% interaction-variance ceiling is structural, not an attention
problem." Both findings advance the saturation paper.

---

## 4. Files to create / modify

### NEW

| Path | Purpose | Approx LOC |
|---|---|---|
| `MCAT/models/model_haca.py` | `HACA_Surv` — copies MCAT_Surv, adds 2.3 + 2.4 + lambda(epoch) acceptance | ~140 |
| `MCAT/utils/haca_train_utils.py` | train / validate / summary loops; passes current epoch to model for AAS lambda | ~220 |
| `launch_haca_blca.sh` | 4-GPU sharded BLCA launcher mirroring `launch_mcat_blca.sh` | ~35 |
| `launch_haca_gbmlgg.sh` | 4-GPU sharded GBMLGG launcher mirroring `launch_mcat_gbmlgg.sh` | ~35 |
| `launch_haca_ablations.sh` | warmup / lambda ablation grid launcher (BLCA only, cheaper cohort) | ~50 |

### MODIFIED

| Path | Change | LOC |
|---|---|---|
| `MCAT/main.py` | Add `'haca'` to --model_type; add `--haca_lambda`, `--haca_warmup_epochs` | +4 |
| `MCAT/utils/core_utils.py` | Dispatch HACA model + train/val/summary loops when `args.model_type == 'haca'`; pass epoch number into model.forward | +10 |

### UNCHANGED (intentionally)

- Encoder (ResNet-50, locked .pt features on disk)
- Splits, gene signatures, NLL-Surv loss, optimiser, LR, gc, epochs, seed
- All MCAT downstream layers (transformers, attention heads, fusion, classifier)

**Strictly additive.** `--model_type mcat` runs byte-identical vanilla
baseline. `--model_type haca` runs HACA v1.5.

---

## 5. Experimental plan

### Phase 1 — Implementation + smoke test (~6 h)
- Write all 5 new files + 2 modifications
- Unit-test AAS schedule (lambda(0)=0, lambda(5)=1, lambda(20)=1)
- 1-epoch smoke run on fold 0 of BLCA
- Diagnostic: print r_j histogram after epoch 1 — must be non-degenerate
  (not all collapsed to 0.5 or 0/1)

### Phase 2 — BLCA pilot (~15 min training)
- Full 5-fold BLCA, seed=1, lambda_final=1.0, N_warmup=5
- **Decision point** vs locked baseline 0.632 +/- 0.042:
  - Delta >= +0.005  -> Phase 3
  - Delta in [-0.005, +0.005]  -> ablation tells us why (Phase 4 first)
  - Delta < -0.005  -> diagnose r collapse / training instability; if
    not fixable, this is the row's result and we report it as the
    deepest-intervention null in the saturation paper

### Phase 3 — Cross-cohort confirmation (~2 h training)
- GBMLGG, seed=1, same hyperparams
- Cross-cohort consistency is the headline ask — both directions same
  sign matters more than absolute magnitude

### Phase 4 — Ablations (BLCA only, ~6 h training)
- (lambda, N_warmup):
  - (0, -)         vanilla MCAT control (sanity)
  - (1, 0)         no warmup — tests v1 failure mode hypothesis
  - (1, 5)         default v1.5
  - (1, 10)        long warmup
  - (0.5, 5)       weak anchoring
  - (2.0, 5)       strong anchoring
  - (1, 5) + aux   v1 aux-loss on (test if explicit supervision helps)
- Posterior-form ablation (one comparison):
  - log additive (Bayesian, default)
  - multiplicative post-softmax: A *= r^lambda then renormalise
  - additive linear: A += lambda * r (no log)
  - Tests whether the Bayesian frame predicts the best form. Negative
    result on the Bayesian form is interesting in itself — report
    honestly.

### Phase 5 — Writeup integration (~3 h)
- Add HACA v1.5 row to saturation paper's main comparison table
- Add Bayesian-attention paragraph in Methods
- Add posterior-form ablation table in Ablations
- Optional: per-patch r_j heatmap on 2 BLCA cases (1 high-risk, 1 low-risk)
  — defer to journal extension if time-pressed

**Total: ~17 h. Implementation 6, runs 8, writeup 3.**

---

## 6. Honest risk assessment (tight)

| Risk | P | Mitigation |
|---|---|---|
| r_j collapses to ~constant — bias is uninformative | low-med | AAS gives r 5 epochs of training before being used; Phase 1 diagnostic catches it |
| Even with informative r, no c-Index lift (saturation hypothesis) | high | Pre-planned: this IS the publishable null in the saturation paper |
| AAS warmup masks a real negative result by starting at vanilla MCAT | low | Posterior-form ablation isolates from warmup; no-warmup ablation row checks both |
| Bayesian-form ablation falsifies the additive-log functional form | medium | Report honestly; reframe as "task-conditioned attention" without strict Bayesian claim |
| Gradient through log(r+eps) explodes when r drops near 0 | low | eps=1e-3 keeps log bounded above -7; no observed instability in v1 prototypes elsewhere |

**P(meaningful c-Index gain >= +0.012 averaged across both cohorts):
~25%** (same as v1 — AAS doesn't change the ceiling, just makes training
more stable)

**P(publishable outcome in saturation paper): 100%** — null is the most
informative null in the ladder, positive is the first positive in the
ladder.

---

## 7. What HACA v1.5 is NOT

- Not a standalone-paper contribution (intentional — saturation paper
  is the stronger bet given existing evidence)
- Not per-pathway anchoring (deferred to journal extension if v1.5 hints
  at promise)
- Not knowledge distillation (deferred — confusing loss structure)
- Not v1's optional aux head (deferred — keep loss path identical to
  MCAT for clean attribution)
- Not an encoder change (ResNet-50 stays; UNI2-h already falsified)
- Not a splits / signatures / NLL-Surv / optimiser change

Single intervention, deepest in the saturation ladder, paired comparison
clean.

---

## 8. Decision summary

If you greenlight v1.5:
1. ~6 h of careful coding + smoke test (Phase 1)
2. BLCA pilot at Phase 2 (~15 min training, decision point)
3. GBMLGG cross-cohort at Phase 3 (~2 h training)
4. Ablations at Phase 4 (~6 h training)
5. Writeup integration at Phase 5 (~3 h)
6. Total ~17 h, mostly unattended GPU

The next message can be "go" and I write `model_haca.py` first.

---
END
