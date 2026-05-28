"""
Convert our compact master_splits.csv (one 0-indexed test-fold column per case)
into the wide Train/Test format that aggregate_features.py expects
(header: ,1,2,...,15 ; each cell Train or Test).

Each patient is Test in exactly one fold (partition-based 15-fold, seed=42).
Output: data/TCGA_{COHORT}/splits/15foldcv/{cohort}_splits_wide.csv

Pre-registration anchor: 7b1256f + amendment 8cafd1e (2-modal).
"""
import pandas as pd

ST = '/mnt/storage7/Dataset_pathomicfusion'
N_FOLDS = 15

for cohort in ['BRCA', 'UCEC']:
    base = f'{ST}/{cohort}/data/TCGA_{cohort}'
    m = pd.read_csv(f'{base}/splits/15foldcv/master_splits.csv')
    rows = {}
    for _, r in m.iterrows():
        f = int(r['fold'])  # 0-indexed test fold
        rows[r['case_id']] = ['Test' if (k - 1) == f else 'Train'
                              for k in range(1, N_FOLDS + 1)]
    wide = pd.DataFrame.from_dict(rows, orient='index',
                                  columns=[str(k) for k in range(1, N_FOLDS + 1)])
    wide.index.name = ''
    out = f'{base}/splits/15foldcv/{cohort.lower()}_splits_wide.csv'
    wide.to_csv(out)
    assert ((wide == 'Test').sum(axis=1) == 1).all(), 'each patient must be Test once'
    print(f'{cohort}: {len(wide)} patients -> {out}')
