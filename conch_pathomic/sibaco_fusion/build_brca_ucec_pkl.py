"""
Build 2-modal CONCH-Bimodal PF data pkls for BRCA + UCEC, mirroring the BLCA
template (BLCA_st_conch_bimodal_flat.pkl) exactly.

Per the 2-modal amendment (commit 8cafd1e):
  - x_path: fold-specific 12288-d MMP CONCH signatures (aggregated/{cohort}/fold_k/).
            Fold-specific because the GMM is fit only on training-fold patches.
  - x_omic: globally z-scored curated panel (mut+CNA+RNA, 378/390-d). Global norm
            matches how BLCA's x_omic was prepared (identical across folds).
  - x_grph: dummy object array (2-modal, graph branch unused).
  - e: event indicator = 1 - censorship (cBioPortal censorship=1 means censored).
  - t: survival_months.  g: grade placeholder (0; survival task ignores it).

Output:
  data/TCGA_{BRCA,UCEC}/splits/{brca,ucec}_st_conch_bimodal_flat.pkl
  with top-level keys {data_pd, cv_splits, conch_bimodal_meta}, cv_splits 1..15.

Pre-registration anchor: 7b1256f + amendment 8cafd1e.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch

ST = '/mnt/storage7/Dataset_pathomicfusion'
AGG = '/home/sbarua/Region_based_segmentation/CONCH-Pathomic/aggregated'
N_FOLDS = 15


def load_omic(cohort: str) -> tuple[pd.DataFrame, list[str]]:
    """Global z-score every feature column. Return (df indexed by case_id, feat cols)."""
    fm = pd.read_csv(f'{ST}/{cohort}/data/TCGA_{cohort}/omic_panel/feature_matrix.csv',
                     index_col='case_id')
    feat_cols = list(fm.columns)
    # Global per-column z-score (matches BLCA's globally-normalized x_omic).
    mu = fm.mean(axis=0)
    sd = fm.std(axis=0).replace(0, 1.0)  # guard zero-variance columns
    fm_z = (fm - mu) / sd
    fm_z = fm_z.astype(np.float32)
    return fm_z, feat_cols


def load_clinical(cohort: str) -> pd.DataFrame:
    """case_id -> (t=survival_months, e=event=1-censorship, fold)."""
    m = pd.read_csv(f'{ST}/{cohort}/data/TCGA_{cohort}/splits/15foldcv/master_splits.csv')
    m = m.set_index('case_id')
    m['t'] = m['survival_months'].astype(float)
    m['e'] = (1.0 - m['censorship'].astype(float))  # event=1 means death observed
    m['g'] = 0.0                                      # grade placeholder
    return m[['t', 'e', 'g', 'fold']]


def build_cohort(cohort: str):
    print(f'\n=== Building {cohort} 2-modal pkl ===')
    omic_z, feat_cols = load_omic(cohort)
    clin = load_clinical(cohort)
    print(f'  omic features: {len(feat_cols)}d, cases with omic: {len(omic_z)}')
    print(f'  clinical cases: {len(clin)}')

    cv_splits = {}
    for k in range(1, N_FOLDS + 1):
        sig = torch.load(f'{AGG}/{cohort}/fold_{k}/patient_signatures.pt',
                         weights_only=False, map_location='cpu')
        sig_pids = set(sig.keys())

        # A patient is usable iff it has CONCH signature AND omic AND clinical.
        usable = sig_pids & set(omic_z.index) & set(clin.index)

        # test = fold==k-1 (0-indexed in master), train = the rest, intersected w/ usable
        test_pids, train_pids = [], []
        for pid in sorted(usable):
            if int(clin.loc[pid, 'fold']) == (k - 1):
                test_pids.append(pid)
            else:
                train_pids.append(pid)

        def pack(pids):
            x_path = np.stack([sig[p].numpy().astype(np.float32) for p in pids])  # (N,12288)
            x_path = x_path[:, None, :]                                            # (N,1,12288)
            x_omic = omic_z.loc[pids, feat_cols].values.astype(np.float32)         # (N,D)
            return {
                'x_patname': np.array(pids, dtype='<U64'),
                'x_path': x_path,
                'x_grph': np.array([0] * len(pids), dtype=object),  # dummy (2-modal)
                'x_omic': x_omic,
                'e': clin.loc[pids, 'e'].values.astype(np.float64),
                't': clin.loc[pids, 't'].values.astype(np.float64),
                'g': clin.loc[pids, 'g'].values.astype(np.float64),
            }

        cv_splits[k] = {'train': pack(train_pids), 'test': pack(test_pids)}
        if k == 1 or k == N_FOLDS:
            print(f'  fold {k}: train={len(train_pids)} test={len(test_pids)} '
                  f'x_path={cv_splits[k]["train"]["x_path"].shape} '
                  f'x_omic={cv_splits[k]["train"]["x_omic"].shape}')

    # sanity: every patient appears in exactly one test fold
    test_counts = {}
    for k in range(1, N_FOLDS + 1):
        for p in cv_splits[k]['test']['x_patname']:
            test_counts[p] = test_counts.get(p, 0) + 1
    assert all(c == 1 for c in test_counts.values()), 'partition violated'
    n_patients = len(test_counts)

    out = {
        'data_pd': None,
        'cv_splits': cv_splits,
        'conch_bimodal_meta': {
            'cohort': cohort,
            'modalities': 'path+omic (2-modal)',
            'omic_dim': len(feat_cols),
            'omic_features': feat_cols,
            'n_patients': n_patients,
            'n_folds': N_FOLDS,
            'path_signature': 'MMP K=16 CONCHv1.5, 12288-d, fold-specific GMM',
            'omic_norm': 'global per-column z-score',
            'prereg': '7b1256f + amendment 8cafd1e (2-modal)',
        },
    }
    out_path = f'{ST}/{cohort}/data/TCGA_{cohort}/splits/{cohort.lower()}_st_conch_bimodal_flat.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'  total patients (unique, in exactly 1 test fold): {n_patients}')
    print(f'  saved: {out_path}')


def main():
    for cohort in ['BRCA', 'UCEC']:
        build_cohort(cohort)


if __name__ == '__main__':
    main()
