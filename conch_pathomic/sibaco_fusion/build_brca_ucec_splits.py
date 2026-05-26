"""
Build 15-fold splits for TCGA-BRCA and TCGA-UCEC per BRCA+UCEC SiBaCo pre-registration §3.4.

Locked rules:
- Source case list: MCAT-style clinical CSV (case_id, slide_id, site, survival, censorship).
- One slide per case (drop dup slides; keep alphabetically-first slide_id per case).
- Stratify by `tissue_source_site` (= 'site' column in MCAT CSV). Rare-site collapse rule
  below for sites with <5 cases (their stratum becomes a single 'RARE' bucket).
- 15 folds, sklearn StratifiedKFold, seed = 42.
- Output: one master CSV per cohort with case_id, slide_id, site, fold, split (train/test).
  Plus per-fold splits_k.csv files in SurvPath format for compatibility with later phases.

This script does NOT touch any PF code, training code, or feature extraction. It only
produces the case → fold mapping that downstream phases consume once features exist.

Output paths:
  data/TCGA_BRCA/splits/15foldcv/master_splits.csv
  data/TCGA_BRCA/splits/15foldcv/splits_{0..14}.csv  (SurvPath-format train,val)
  data/TCGA_UCEC/splits/15foldcv/master_splits.csv
  data/TCGA_UCEC/splits/15foldcv/splits_{0..14}.csv

Run:
  cd /home/sbarua/Region_based_segmentation/_publish/multimodal-survival-saturation
  source ~/.venv/bin/activate
  python conch_pathomic/sibaco_fusion/build_brca_ucec_splits.py

Pre-registration anchor: 7b1256f (BRCA+UCEC SiBaCo pre-reg).
"""

import os
import sys
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

MCAT_CSV_DIR = '/home/sbarua/Region_based_segmentation/mcat_replication/MCAT/dataset_csv'
PF_DATA_DIR = '/home/sbarua/Region_based_segmentation/pathomic_fusion_replica/PathomicFusion/data'

SEED = 42
N_FOLDS = 15
RARE_SITE_THRESHOLD = 5  # sites with fewer than this many cases get collapsed to 'RARE'


def load_clinical(cohort: str) -> pd.DataFrame:
    """Load MCAT clinical CSV for cohort and keep only clinical columns."""
    zip_path = os.path.join(MCAT_CSV_DIR, f'tcga_{cohort.lower()}_all_clean.csv.zip')
    with zipfile.ZipFile(zip_path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as f:
            df = pd.read_csv(f, low_memory=False)
    keep = ['case_id', 'slide_id', 'site', 'is_female', 'oncotree_code',
            'age', 'survival_months', 'censorship']
    return df[keep].copy()


def dedup_one_slide_per_case(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one slide per case_id. Tie-break: alphabetically-first slide_id."""
    df_sorted = df.sort_values(['case_id', 'slide_id'])
    out = df_sorted.drop_duplicates(subset='case_id', keep='first').reset_index(drop=True)
    return out


def stratification_buckets(df: pd.DataFrame) -> pd.Series:
    """Collapse rare sites (< RARE_SITE_THRESHOLD cases) into 'RARE'. Return strata series."""
    counts = df['site'].value_counts()
    rare = set(counts[counts < RARE_SITE_THRESHOLD].index)
    return df['site'].where(~df['site'].isin(rare), other='RARE')


def build_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a 'fold' column in [0, N_FOLDS).

    Each row gets the fold index of its TEST fold. Train for fold k = all rows where
    fold != k.
    """
    strata = stratification_buckets(df)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, test_idx) in enumerate(skf.split(df, strata)):
        fold[test_idx] = k
    assert (fold >= 0).all(), 'every row must be assigned to exactly one test fold'
    out = df.copy()
    out['fold'] = fold
    return out


def write_master(df_with_fold: pd.DataFrame, out_path: str) -> None:
    cols = ['case_id', 'slide_id', 'site', 'is_female', 'age',
            'survival_months', 'censorship', 'fold']
    df_with_fold[cols].to_csv(out_path, index=False)


def write_survpath_format(df_with_fold: pd.DataFrame, out_dir: str) -> None:
    """For each fold k, write splits_k.csv with two columns (train, val) of case_ids.
    Matches SurvPath split CSV format for downstream loader compatibility.
    """
    for k in range(N_FOLDS):
        test_cases = df_with_fold.loc[df_with_fold['fold'] == k, 'case_id'].values
        train_cases = df_with_fold.loc[df_with_fold['fold'] != k, 'case_id'].values
        # SurvPath CSVs are ragged: train column is longer than val. Pad with NaN.
        n = max(len(train_cases), len(test_cases))
        train_col = list(train_cases) + [np.nan] * (n - len(train_cases))
        val_col = list(test_cases) + [np.nan] * (n - len(test_cases))
        pd.DataFrame({'train': train_col, 'val': val_col}).to_csv(
            os.path.join(out_dir, f'splits_{k}.csv'), index=False)


def summarize(df_with_fold: pd.DataFrame, cohort: str) -> None:
    print(f"\n=== {cohort} 15-fold summary ===")
    print(f"  total cases: {len(df_with_fold)}")
    print(f"  total events: {(df_with_fold['censorship'] == 0).sum()}  "
          f"censored: {(df_with_fold['censorship'] == 1).sum()}")
    print(f"  per-fold size (test): "
          f"min={df_with_fold['fold'].value_counts().min()} "
          f"max={df_with_fold['fold'].value_counts().max()}")
    per_fold_events = df_with_fold.groupby('fold').apply(
        lambda g: (g['censorship'] == 0).sum(), include_groups=False)
    print(f"  per-fold events: min={per_fold_events.min()} "
          f"max={per_fold_events.max()} median={int(per_fold_events.median())}")
    strata = stratification_buckets(df_with_fold)
    print(f"  strata: {strata.nunique()} unique "
          f"(rare-site collapse threshold = {RARE_SITE_THRESHOLD})")


def main():
    for cohort in ['BRCA', 'UCEC']:
        df = load_clinical(cohort.lower())
        n_before = len(df)
        df = dedup_one_slide_per_case(df)
        print(f"{cohort}: rows {n_before} → {len(df)} after dedup (one slide per case)")

        df = build_folds(df)
        summarize(df, cohort)

        out_dir = os.path.join(PF_DATA_DIR, f'TCGA_{cohort}', 'splits', '15foldcv')
        os.makedirs(out_dir, exist_ok=True)
        write_master(df, os.path.join(out_dir, 'master_splits.csv'))
        write_survpath_format(df, out_dir)
        print(f"  wrote master_splits.csv + splits_0..14.csv → {out_dir}")


if __name__ == '__main__':
    main()
