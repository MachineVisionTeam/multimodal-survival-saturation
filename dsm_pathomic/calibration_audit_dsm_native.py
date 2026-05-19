"""
DSM-Pathomic NATIVE calibration audit.

Unlike the previous Breslow-based audit (which is mathematically inappropriate
for DSM — Cox's exp(linear_predictor) semantics don't apply to a Weibull
mixture risk), this script computes IBS using DSM's NATIVE survival function:

    S(t | x) = sum_{m,k}  w_{m,k}(x)  *  S_Weibull(t ; alpha_{m,k}, beta_{m,k})

For each fold:
  - Reconstruct the trained DSM model
  - Run inference on the test set (via the standard PF data loader)
  - For each patient, compute S(t | x) at the IBS evaluation grid
  - Aggregate patch-level predictions to patient level
  - Compute IBS / IBLL with sksurv

This is the loss-function-lever calibration test in its honest form.

Author: 2026-05-19.
"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch

from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score, brier_score
from scipy.stats import ttest_rel, wilcoxon, t as student
from argparse import Namespace

# Make PF imports work
sys.path.insert(0, '/home/sbarua/Region_based_segmentation/pathomic_fusion_replica/PathomicFusion')
sys.path.insert(0, '/home/sbarua/Region_based_segmentation/DSM-Pathomic')

from networks import define_net
from data_loaders import PathgraphomicFastDatasetLoader
from utils import mixed_collate


ROOT = '/home/sbarua/Region_based_segmentation/pathomic_fusion_replica/PathomicFusion'
EXP_DIR = f'{ROOT}/runs_modern/TCGA_GBMLGG/surv_15_rnaseq'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def survival_grid(times_train, times_test, n_points=50, lo_pct=10.0, hi_pct=80.0):
    all_events = np.concatenate([times_train, times_test])
    lo = np.percentile(all_events, lo_pct)
    hi = np.percentile(all_events, hi_pct)
    grid = np.linspace(lo, hi, n_points)
    # Clip safely inside both event-time ranges
    grid = grid[(grid >= times_train.min()) & (grid <= times_train.max())]
    grid = grid[(grid >= times_test.min())  & (grid <= times_test.max())]
    return grid


def run_dsm_inference_for_fold(fold: int, eval_times: np.ndarray):
    """
    Load the trained DSM fold-k model + run inference on its test set,
    using DSM's NATIVE survival function. Returns patient-level S(t|x).
    """
    # Load checkpoint
    ckpt = torch.load(
        f'{EXP_DIR}/pathgraphomic_dsm/pathgraphomic_dsm_{fold}.pt',
        map_location='cpu', weights_only=False)

    # Reconstruct options from the saved opt namespace
    saved_opt = ckpt['opt']
    opt = Namespace(**vars(saved_opt) if hasattr(saved_opt, '__dict__') else saved_opt)
    opt.gpu_ids = [0] if DEVICE.type == 'cuda' else []

    # Build the model (skip encoder ckpt loading; we'll load the full state)
    # define_net needs k=None to skip encoder ckpt; then we load full ckpt
    model = define_net(opt, k=None)
    inner = model.module if hasattr(model, 'module') else model
    inner.load_state_dict(ckpt['model_state_dict'])
    inner.eval()
    if DEVICE.type == 'cuda':
        inner = inner.to(DEVICE)

    # Build test data loader using PF's exact setup
    test_loader = torch.utils.data.DataLoader(
        PathgraphomicFastDatasetLoader(opt, ckpt['data'], split='test', mode=opt.mode),
        batch_size=opt.batch_size, shuffle=False, collate_fn=mixed_collate)

    # Run inference, computing native S(t | x) per batch
    t_grid = torch.tensor(eval_times, dtype=torch.float32).to(DEVICE)
    all_S = []
    all_t = []
    all_c = []
    all_pat_idx = 0
    patnames = list(ckpt['data']['test']['x_patname'])
    all_pat = []

    with torch.no_grad():
        for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(test_loader):
            x_path = x_path.to(DEVICE)
            # Forward populates inner._cached_h_p/g/o
            _, _ = inner(x_path=x_path,
                          x_grph=x_grph.to(DEVICE),
                          x_omic=x_omic.to(DEVICE))
            # Native S(t | x)
            S = inner.survival_at_times(t_grid)                 # (B, T)
            all_S.append(S.cpu().numpy())
            all_t.append(survtime.cpu().numpy())
            all_c.append(censor.cpu().numpy())
            # Map this batch's patches to their patient names
            B = S.shape[0]
            all_pat.extend(patnames[all_pat_idx:all_pat_idx + B])
            all_pat_idx += B

    S_patch = np.concatenate(all_S, axis=0)                      # (N_patches, T)
    t_patch = np.concatenate(all_t, axis=0)
    c_patch = np.concatenate(all_c, axis=0).astype(int)

    # Aggregate to patient level: mean S per patient over its patches
    df = pd.DataFrame({'patient': all_pat, 't': t_patch, 'c': c_patch})
    df = df.assign(**{f'S_{j}': S_patch[:, j] for j in range(S_patch.shape[1])})
    agg_cols = {f'S_{j}': 'mean' for j in range(S_patch.shape[1])}
    agg_cols.update({'t': 'first', 'c': 'first'})
    df_pat = df.groupby('patient', as_index=False).agg(agg_cols).sort_values('t').reset_index(drop=True)

    S_pat = df_pat[[f'S_{j}' for j in range(S_patch.shape[1])]].values
    t_pat = df_pat['t'].values
    c_pat = df_pat['c'].astype(bool).values

    # Pull training survival info from saved data
    train = ckpt['data']['train']
    train_t = train['t']
    train_c = train['e'].astype(bool)

    return {
        'eval_times': eval_times,
        'S_test': S_pat,
        'test_t': t_pat, 'test_c': c_pat,
        'train_t': train_t, 'train_c': train_c,
    }


def compute_metrics(result_dict):
    """Given inference output, compute IBS + IBLL."""
    surv_train = Surv.from_arrays(event=result_dict['train_c'], time=result_dict['train_t'])
    surv_test = Surv.from_arrays(event=result_dict['test_c'], time=result_dict['test_t'])
    S_test = result_dict['S_test']
    eval_times = result_dict['eval_times']

    # Clip S to (0, 1) to avoid log(0) issues; sksurv handles the rest
    S_test = np.clip(S_test, 1e-6, 1 - 1e-6)

    ibs = integrated_brier_score(surv_train, surv_test, S_test, eval_times)

    # IBLL (same formulation as the GRFN audit)
    test_t = result_dict['test_t']
    test_c = result_dict['test_c']
    ibll_per_t = []
    for j, t in enumerate(eval_times):
        keep = (test_t > t) | test_c
        if keep.sum() == 0: continue
        ev_by_t = ((test_t <= t) & test_c).astype(float)
        bll = -(ev_by_t * np.log(1 - S_test[:, j]) + (1 - ev_by_t) * np.log(S_test[:, j]))
        ibll_per_t.append(bll[keep].mean())
    ibll = (float(np.trapezoid(ibll_per_t, eval_times[:len(ibll_per_t)]) /
                  (eval_times[len(ibll_per_t) - 1] - eval_times[0]))
            if len(ibll_per_t) > 1 else float('nan'))

    return float(ibs), ibll


def main():
    # Existing comparator numbers from GRFN audit
    grfn_pf = pickle.load(open(
        '/home/sbarua/Region_based_segmentation/GRFN-Pathomic/gbmlgg/CALIBRATION_RESULTS.pkl', 'rb'))

    print(f'{"Fold":<5}{"PF IBS":<11}{"PF+Platt":<11}{"GRFN IBS":<11}{"DSM-native IBS":<16}{"D(DSM-PF)":<12}{"D(DSM-GRFN)":<12}')
    rows = []
    for k in range(1, 16):
        try:
            ref = grfn_pf.iloc[k - 1]
            # Build eval-time grid matching the GRFN audit
            # (re-derive from saved data so train/test share the same grid)
            ckpt = torch.load(
                f'{EXP_DIR}/pathgraphomic_dsm/pathgraphomic_dsm_{k}.pt',
                map_location='cpu', weights_only=False)
            train_t = np.array(ckpt['data']['train']['t'])
            train_c = np.array(ckpt['data']['train']['e']).astype(bool)
            test_t = np.array(ckpt['data']['test']['t'])
            test_c = np.array(ckpt['data']['test']['e']).astype(bool)
            train_evt = train_t[train_c]
            test_evt = test_t[test_c]
            eval_times = survival_grid(train_evt, test_evt)

            r = run_dsm_inference_for_fold(k, eval_times)
            ibs_dsm, ibll_dsm = compute_metrics(r)

            d_pf = ibs_dsm - ref['ibs_pf']
            d_grfn = ibs_dsm - ref['ibs_grfn']
            print(f'{k:<5}{ref["ibs_pf"]:<11.4f}{ref["ibs_pf_platt"]:<11.4f}'
                  f'{ref["ibs_grfn"]:<11.4f}{ibs_dsm:<16.4f}{d_pf:+12.4f}{d_grfn:+12.4f}')
            rows.append({'fold': k, 'ibs_dsm': ibs_dsm, 'ibll_dsm': ibll_dsm,
                         'ibs_pf': ref['ibs_pf'], 'ibs_pf_platt': ref['ibs_pf_platt'],
                         'ibs_grfn': ref['ibs_grfn'], 'ibll_grfn': ref['ibll_grfn'],
                         'd_dsm_pf': d_pf, 'd_dsm_grfn': d_grfn})
        except Exception as e:
            print(f'fold {k} FAILED: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
            rows.append({'fold': k, 'error': str(e)})

    df = pd.DataFrame(rows)
    if 'ibs_dsm' in df.columns and df['ibs_dsm'].notna().all():
        d_ibs = df['ibs_dsm'].values
        p_ibs = df['ibs_pf'].values
        pp_ibs = df['ibs_pf_platt'].values
        g_ibs = df['ibs_grfn'].values
        d_ibll = df['ibll_dsm'].values
        g_ibll = df['ibll_grfn'].values

        print('\n=== AGGREGATE IBS (lower better, native DSM S(t|x), no Breslow) ===')
        print(f'  PF (locked, Cox+Breslow):    {p_ibs.mean():.4f} +/- {p_ibs.std(ddof=1):.4f}')
        print(f'  PF + Platt scaling:           {pp_ibs.mean():.4f} +/- {pp_ibs.std(ddof=1):.4f}')
        print(f'  GRFN-Pathomic (Cox+Breslow):  {g_ibs.mean():.4f} +/- {g_ibs.std(ddof=1):.4f}')
        print(f'  DSM-Pathomic (native S(t|x)): {d_ibs.mean():.4f} +/- {d_ibs.std(ddof=1):.4f}')

        for label, diff in [
            ('DSM vs PF        ', d_ibs - p_ibs),
            ('DSM vs PF + Platt', d_ibs - pp_ibs),
            ('DSM vs GRFN      ', d_ibs - g_ibs),
        ]:
            t, pt = ttest_rel(diff, np.zeros_like(diff))
            try: w, pw = wilcoxon(diff); ws = f'{pw:.4f}'
            except: ws = 'n/a'
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            ci = student.interval(0.95, df=len(diff)-1, loc=diff.mean(), scale=se)
            print(f'\n  D {label}: mean={diff.mean():+.4f}  paired-t p={pt:.4f}  '
                  f'Wilcoxon p={ws}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  '
                  f'DSM wins={int((diff < 0).sum())}/15')

        print(f'\n=== AGGREGATE IBLL (lower better) ===')
        print(f'  GRFN: {g_ibll.mean():.4f} +/- {g_ibll.std(ddof=1):.4f}')
        print(f'  DSM:  {d_ibll.mean():.4f} +/- {d_ibll.std(ddof=1):.4f}')

    out = '/home/sbarua/Region_based_segmentation/DSM-Pathomic/gbmlgg/CALIBRATION_RESULTS_native.pkl'
    with open(out, 'wb') as f:
        pickle.dump(df, f)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
