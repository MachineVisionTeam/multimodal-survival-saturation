"""
Phase 2c — Calibration audit for GRFN-Pathomic vs locked PF on TCGA-GBMLGG.

Computes per-fold and aggregate:
  - Integrated Brier Score (IBS) over a clipped event-time grid
  - Integrated Binomial Log-Likelihood (IBLL)
  - Brier score curves at fixed eval times
  - Optionally: Platt-recalibrated PF as a "post-hoc fix" baseline
    (recommended by the audit paper 'Good Rankings, Wrong Probabilities',
    arXiv 2604.04239, April 2026)

Inputs:
  - Locked PF checkpoints   under runs_modern/TCGA_GBMLGG/surv_15_rnaseq/pathgraphomic_fusion/
  - GRFN-Pathomic checkpoints under runs_modern/TCGA_GBMLGG/surv_15_rnaseq/pathgraphomic_grfn/
  - Per-fold patch-level test/train prediction pickles
  - Per-fold checkpoint's `data` dict (has patient names + survival data)

Outputs:
  - GRFN-Pathomic/gbmlgg/CALIBRATION_RESULTS.pkl (per-fold metrics + aggregate stats)
  - Printed table for inspection

Author: 2026-05-18.
"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch

from sksurv.util import Surv
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import integrated_brier_score, brier_score
from scipy.stats import ttest_rel, wilcoxon, t as student
from sklearn.linear_model import LogisticRegression


ROOT = '/home/sbarua/Region_based_segmentation/pathomic_fusion_replica/PathomicFusion'
EXP_DIR = f'{ROOT}/runs_modern/TCGA_GBMLGG/surv_15_rnaseq'


def load_patient_level(model_name: str, fold: int, split: str):
    """
    Load patch-level predictions + checkpoint's data dict and aggregate
    to patient level by averaging risk over each patient's patches.

    Returns
    -------
    pd.DataFrame with columns: patient, risk (mean), t, c (event indicator 0/1).
    Sorted by survival time, with one row per unique patient.
    """
    pkl_path = f'{EXP_DIR}/{model_name}/{model_name}_{fold}_patch_pred_{split}.pkl'
    pkl = pickle.load(open(pkl_path, 'rb'))
    risks_patch, times, censors, _, _ = pkl

    ckpt = torch.load(
        f'{EXP_DIR}/{model_name}/{model_name}_{fold}.pt',
        map_location='cpu', weights_only=False)
    pat_names = ckpt['data'][split]['x_patname']

    df = pd.DataFrame({
        'patient': list(pat_names),
        'risk': risks_patch,
        't': times,
        'c': censors.astype(int),     # event indicator (1=event, 0=censored)
    })

    # Aggregate per patient: mean risk (patch predictions are nearly identical
    # for the same patient anyway, since the model is deterministic and the
    # patch differences are minor visual augmentations). Take first t / c
    # (they're identical per patient).
    pat = df.groupby('patient', as_index=False).agg({
        'risk': 'mean', 't': 'first', 'c': 'first'
    }).sort_values('t').reset_index(drop=True)
    return pat


def survival_grid(train_events_t: np.ndarray, test_events_t: np.ndarray,
                  n_points: int = 50, lo_pct: float = 10.0, hi_pct: float = 80.0):
    """
    Build the time grid used for IBS / Brier.

    Standard practice: take the inner [lo_pct%, hi_pct%] quantiles of event
    times across BOTH train and test, then take n_points evenly spaced in
    this window. We use 10-80 to avoid the unstable tails.
    """
    all_events = np.concatenate([train_events_t, test_events_t])
    lo = np.percentile(all_events, lo_pct)
    hi = np.percentile(all_events, hi_pct)
    return np.linspace(lo, hi, n_points)


def fold_metrics(model_name: str, fold: int, n_grid: int = 50):
    """
    Compute per-fold calibration metrics: IBS, IBLL, max-Brier.

    Returns dict with: fold, n_train, n_test, ibs, ibll, brier_curve,
                       eval_times, surv_test (test-set predicted S(t|x))
    """
    train = load_patient_level(model_name, fold, 'train')
    test = load_patient_level(model_name, fold, 'test')

    # Fit Breslow baseline from training risks + survival
    bre = BreslowEstimator()
    bre.fit(
        linear_predictor=train['risk'].values,
        event=train['c'].astype(bool).values,
        time=train['t'].values,
    )

    # Time grid: inner quantiles to avoid unstable tails. Also CLIP to test-set
    # time range so we don't extrapolate beyond observed test events.
    train_evt = train[train['c'].astype(bool)]['t'].values
    test_evt = test[test['c'].astype(bool)]['t'].values
    if len(test_evt) == 0:
        raise ValueError(f'fold {fold} test has no events!')
    eval_times = survival_grid(train_evt, test_evt, n_grid)
    # Clip to be safely inside test range
    eval_times = eval_times[(eval_times >= test_evt.min()) & (eval_times <= test_evt.max())]
    eval_times = eval_times[(eval_times >= train_evt.min()) & (eval_times <= train_evt.max())]

    # Predicted S(t|x) for each test patient at each eval time
    surv_funcs = bre.get_survival_function(test['risk'].values)
    surv_at_times = np.array([[sf(t) for t in eval_times] for sf in surv_funcs])

    # Build sksurv structured arrays
    surv_train = Surv.from_arrays(
        event=train['c'].astype(bool).values, time=train['t'].values)
    surv_test = Surv.from_arrays(
        event=test['c'].astype(bool).values, time=test['t'].values)

    # IBS (lower is better; in [0, ~0.25] for survival problems)
    ibs = integrated_brier_score(surv_train, surv_test, surv_at_times, eval_times)

    # Brier curve at each eval time
    _, brier_curve = brier_score(surv_train, surv_test, surv_at_times, eval_times)

    # IBLL — integrated binomial log-likelihood
    # Use the formula: BLL(t) = mean over test patients of:
    #   -[ ev_by_t * log(1 - S(t|x)) + (1 - ev_by_t) * log(S(t|x)) ]
    # where ev_by_t is the IPCW-weighted indicator. We approximate with a
    # simple version: for each eval time, compute the observed event-by-t
    # rate AT THAT TIME using only patients still at risk; compare to 1-S(t|x).
    # Bound S(t|x) away from 0/1 to avoid log singularities.
    S_clip = np.clip(surv_at_times, 1e-6, 1.0 - 1e-6)
    test_t_arr = test['t'].values
    test_c_arr = test['c'].astype(bool).values
    ibll_per_t = []
    for j, t in enumerate(eval_times):
        # IPCW-style: only patients with t_i > t OR (t_i <= t AND event)
        # Skip censored patients with t_i <= t (their event-by-t status is unknown)
        keep = (test_t_arr > t) | test_c_arr
        if keep.sum() == 0:
            continue
        ev_by_t = ((test_t_arr <= t) & test_c_arr).astype(float)
        bll = -(ev_by_t * np.log(1 - S_clip[:, j]) +
                (1 - ev_by_t) * np.log(S_clip[:, j]))
        ibll_per_t.append(bll[keep].mean())
    ibll = float(np.trapz(ibll_per_t, eval_times[:len(ibll_per_t)]) / (
        eval_times[len(ibll_per_t) - 1] - eval_times[0])) if len(ibll_per_t) > 1 else float('nan')

    return {
        'fold': fold,
        'n_train': len(train),
        'n_test': len(test),
        'ibs': float(ibs),
        'ibll': ibll,
        'brier_curve': np.asarray(brier_curve),
        'eval_times': eval_times,
        'surv_at_times': surv_at_times,
        'test_t': test_t_arr,
        'test_c': test_c_arr,
        'train': train,
        'test': test,
    }


def platt_scale_survival(grfn_or_pf_results, fold: int):
    """
    Post-hoc Platt scaling baseline (recommended by the audit paper).

    For each eval time t, fit a logistic regression on the training set:
        feature  = predicted_S(t|x_train)
        label    = event_by_t indicator (skip censored before t)
    Then apply the fitted logistic to test predicted_S(t|x_test) to get a
    recalibrated S_platt(t|x_test). Recompute IBS/IBLL with the recalibrated
    survival probabilities.
    """
    # We need training-set surv predictions too — recompute them from the
    # same Breslow model but applied to train risks.
    # For efficiency, re-fit Breslow + get train surv funcs.
    train = grfn_or_pf_results['train']
    test = grfn_or_pf_results['test']
    eval_times = grfn_or_pf_results['eval_times']

    bre = BreslowEstimator()
    bre.fit(train['risk'].values, train['c'].astype(bool).values, train['t'].values)
    surv_train_funcs = bre.get_survival_function(train['risk'].values)
    surv_train_at_t = np.array([[sf(t) for t in eval_times] for sf in surv_train_funcs])
    surv_test_at_t = grfn_or_pf_results['surv_at_times']

    # Recalibrate per eval time
    surv_test_platt = np.zeros_like(surv_test_at_t)
    train_t = train['t'].values; train_c = train['c'].astype(bool).values
    for j, t in enumerate(eval_times):
        keep = (train_t > t) | train_c
        ev_by_t = ((train_t <= t) & train_c).astype(int)
        X_tr = surv_train_at_t[keep, j].reshape(-1, 1)
        y_tr = 1 - ev_by_t[keep]   # P(survive to t) → 1 - event_by_t
        if len(np.unique(y_tr)) < 2:
            # Degenerate at this time
            surv_test_platt[:, j] = surv_test_at_t[:, j]
            continue
        lr = LogisticRegression()
        lr.fit(X_tr, y_tr)
        # Apply to test
        X_te = surv_test_at_t[:, j].reshape(-1, 1)
        surv_test_platt[:, j] = lr.predict_proba(X_te)[:, 1]

    # Recompute IBS/IBLL with Platt-scaled probabilities
    surv_train = Surv.from_arrays(
        event=train['c'].astype(bool).values, time=train['t'].values)
    surv_test = Surv.from_arrays(
        event=test['c'].astype(bool).values, time=test['t'].values)
    ibs_platt = integrated_brier_score(surv_train, surv_test, surv_test_platt, eval_times)
    return float(ibs_platt)


def main():
    rows = []
    for k in range(1, 16):
        try:
            grfn = fold_metrics('pathgraphomic_grfn', k)
            pf = fold_metrics('pathgraphomic_fusion', k)
            ibs_pf_platt = platt_scale_survival(pf, k)
            rows.append({
                'fold': k,
                'n_test': grfn['n_test'],
                'ibs_grfn': grfn['ibs'], 'ibll_grfn': grfn['ibll'],
                'ibs_pf':   pf['ibs'],   'ibll_pf':   pf['ibll'],
                'ibs_pf_platt': ibs_pf_platt,
                'delta_ibs': grfn['ibs'] - pf['ibs'],
                'delta_ibs_vs_platt': grfn['ibs'] - ibs_pf_platt,
            })
            print(f'fold {k:>2}  n_test={grfn["n_test"]:>3}  '
                  f'IBS  GRFN={grfn["ibs"]:.4f}  PF={pf["ibs"]:.4f}  '
                  f'PF+Platt={ibs_pf_platt:.4f}  Δ(GRFN-PF)={grfn["ibs"]-pf["ibs"]:+.4f}  '
                  f'Δ(GRFN-PF-Platt)={grfn["ibs"]-ibs_pf_platt:+.4f}  '
                  f'IBLL GRFN={grfn["ibll"]:.4f} PF={pf["ibll"]:.4f}')
        except Exception as e:
            print(f'fold {k}: FAILED {type(e).__name__}: {e}')
            rows.append({'fold': k, 'error': str(e)})

    df = pd.DataFrame(rows)
    print('\n=== AGGREGATE (15-fold paired) ===')

    if 'ibs_grfn' in df.columns and df['ibs_grfn'].notna().all():
        g_ibs = df['ibs_grfn'].values
        p_ibs = df['ibs_pf'].values
        pp_ibs = df['ibs_pf_platt'].values
        d1 = g_ibs - p_ibs       # GRFN vs PF
        d2 = g_ibs - pp_ibs      # GRFN vs PF+Platt

        print(f'IBS  PF (locked):        {p_ibs.mean():.4f} ± {p_ibs.std(ddof=1):.4f}')
        print(f'IBS  PF + Platt scaling: {pp_ibs.mean():.4f} ± {pp_ibs.std(ddof=1):.4f}')
        print(f'IBS  GRFN-Pathomic:      {g_ibs.mean():.4f} ± {g_ibs.std(ddof=1):.4f}')
        for label, d in [('GRFN vs PF', d1), ('GRFN vs PF+Platt', d2)]:
            t, p = ttest_rel(d, np.zeros_like(d))
            try:
                w, pw = wilcoxon(d, alternative='two-sided')
            except: pw = float('nan')
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = student.interval(0.95, df=len(d)-1, loc=d.mean(), scale=se)
            print(f'\n  Δ ({label:<22s}): mean={d.mean():+.4f}  '
                  f'paired-t p={p:.4f}  Wilcoxon p={pw:.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  '
                  f'GRFN wins (lower IBS) in {(d < 0).sum()}/15 folds')

        if df['ibll_grfn'].notna().all() and df['ibll_pf'].notna().all():
            g_ibll = df['ibll_grfn'].values
            p_ibll = df['ibll_pf'].values
            print(f'\nIBLL PF (locked):        {p_ibll.mean():.4f} ± {p_ibll.std(ddof=1):.4f}')
            print(f'IBLL GRFN-Pathomic:      {g_ibll.mean():.4f} ± {g_ibll.std(ddof=1):.4f}')
            d3 = g_ibll - p_ibll
            t, p = ttest_rel(d3, np.zeros_like(d3))
            print(f'  Δ IBLL (GRFN vs PF): mean={d3.mean():+.4f}  paired-t p={p:.4f}  '
                  f'GRFN wins in {(d3 < 0).sum()}/15 folds')

    out_path = '/home/sbarua/Region_based_segmentation/GRFN-Pathomic/gbmlgg/CALIBRATION_RESULTS.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(df, f)
    print(f'\nSaved per-fold metrics to: {out_path}')


if __name__ == '__main__':
    main()
