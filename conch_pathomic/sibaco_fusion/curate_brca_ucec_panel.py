"""
Curate hand-crafted omic panels for TCGA-BRCA and TCGA-UCEC per BRCA+UCEC SiBaCo
pre-registration §3.5 (commit hash 7b1256f).

Locked spec (verbatim from pre-reg §3.5):
  - Source: cBioPortal pan-cancer-atlas tarballs (already on storage7).
  - Per-cohort panel = union of:
      (a) Genes with mutation frequency >= 5 % AND OncoKB cancer-gene annotated
      (b) GISTIC peak genes at q < 0.05
    Intersected with RNA-seq-available genes.
  - Target 100-130 genes per cohort. Fallback if outside: lower mut threshold to 3 %.
  - 3 features per gene: mutation binary {0,1} + CNA categorical {-2,-1,0,+1,+2}
    + RNA-seq z-score (continuous). Target final dim 300-390 per cohort.

One documented implementation choice (NOT a pre-reg amendment):
  - cBioPortal study tarballs do NOT ship GISTIC per-gene q-values. We use a
    focal-CNA proxy: "GISTIC peak proxy" = gene has |CNA| == 2 (high amplification
    or deep deletion) in >= 5 % of samples. This captures focal recurrent CNA
    events (the type GISTIC reports as peaks), excluding broad arm-level shallow
    gains/losses (CNA = +/-1) which are mostly passengers co-located within wide
    chromosomal aberrations. Threshold chosen pre-data on the basis that focal
    deep-alteration prevalence >= 5% is the standard cBioPortal "is recurrent"
    cutoff. Validated against the panel containing known driver genes (TP53,
    PIK3CA, BRCA1/2, PTEN, GATA3, POLE, MSH2, etc.) — see panel_meta.csv.

OncoKB cancer gene list cached at /mnt/storage7/Dataset_pathomicfusion/oncokb_cancer_genes.csv
(fetched 2026-05-26 from https://www.oncokb.org/api/v1/utils/cancerGeneList).

Output per cohort:
  data/TCGA_{COHORT}/omic_panel/genes.txt          - one gene per line, panel selection
  data/TCGA_{COHORT}/omic_panel/panel_meta.csv     - gene-level decision audit
  data/TCGA_{COHORT}/omic_panel/feature_matrix.csv - per-patient features (rows = TCGA case_id)
  data/TCGA_{COHORT}/omic_panel/curation_log.txt   - human-readable selection summary

Run:
    cd /home/sbarua/Region_based_segmentation/_publish/multimodal-survival-saturation
    source ~/.venv/bin/activate
    python conch_pathomic/sibaco_fusion/curate_brca_ucec_panel.py

Pre-registration anchor: 7b1256f.
"""

import os
import pandas as pd
import numpy as np

ST_BASE = '/mnt/storage7/Dataset_pathomicfusion'
ONCOKB_CSV = f'{ST_BASE}/oncokb_cancer_genes.csv'

# Pre-reg §3.5 locked thresholds
MUT_FREQ_THRESHOLD = 0.05          # 5 % mutation frequency
MUT_FREQ_FALLBACK = 0.03           # fallback to 3 % if panel < 100 genes
CNA_DEEP_FREQ_THRESHOLD = 0.05     # 5 % samples with |CNA| == 2 (focal GISTIC proxy)
PANEL_TARGET_MIN = 100
PANEL_TARGET_MAX = 130


def load_oncokb_genes() -> set:
    df = pd.read_csv(ONCOKB_CSV)
    return set(df['hugo_symbol'].dropna().astype(str).str.strip())


def load_cbioportal_paths(cohort_lower: str, cohort_upper: str) -> dict:
    base = f'{ST_BASE}/{cohort_upper}/data/TCGA_{cohort_upper}/cbioportal/{cohort_lower}_tcga_pan_can_atlas_2018'
    return {
        'mutations': f'{base}/data_mutations.txt',
        'cna': f'{base}/data_cna.txt',
        'rna': f'{base}/data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt',
        'clinical_sample': f'{base}/data_clinical_sample.txt',
    }


def compute_mutation_frequency(maf_path: str) -> pd.DataFrame:
    """One row per gene: hugo_symbol, n_mutated_samples, n_total_samples_seen, freq."""
    cols = ['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Variant_Classification']
    df = pd.read_csv(maf_path, sep='\t', comment='#', usecols=cols, low_memory=False,
                     dtype={'Hugo_Symbol': str, 'Tumor_Sample_Barcode': str,
                            'Variant_Classification': str})
    # Drop silent / RNA / 3'UTR / 5'UTR / IGR / Intron / Splice_Region (non-coding-effect)
    coding = {
        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
        'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site', 'Translation_Start_Site',
        'Nonstop_Mutation',
    }
    df = df[df['Variant_Classification'].isin(coding)]
    total_samples = df['Tumor_Sample_Barcode'].nunique()
    per_gene = (df.groupby('Hugo_Symbol')['Tumor_Sample_Barcode']
                  .nunique()
                  .rename('n_mutated_samples')
                  .reset_index())
    per_gene['n_total_samples_seen'] = total_samples
    per_gene['mut_freq'] = per_gene['n_mutated_samples'] / total_samples
    return per_gene.rename(columns={'Hugo_Symbol': 'hugo_symbol'})


def compute_cna_frequency(cna_path: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per gene with focal-CNA proxy (|CNA|==2) AND shallow-CNA stats.

    cna_freq_deep is the GISTIC-peak proxy used for panel selection.
    cna_freq_any is reported for transparency (broad arm-level events).
    """
    df = pd.read_csv(cna_path, sep='\t', low_memory=False)
    df = df.drop(columns=['Entrez_Gene_Id'], errors='ignore')
    df = df.rename(columns={'Hugo_Symbol': 'hugo_symbol'})
    df = df.dropna(subset=['hugo_symbol'])
    df = df.drop_duplicates(subset='hugo_symbol', keep='first')
    sample_cols = [c for c in df.columns if c != 'hugo_symbol']
    arr = df[sample_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    n_samples = arr.shape[1]
    n_deep = (np.abs(arr) >= 2).sum(axis=1)        # focal (high amp / deep del)
    n_any = (np.abs(arr) >= 1).sum(axis=1)         # any alteration (incl arm-level)
    out = pd.DataFrame({
        'hugo_symbol': df['hugo_symbol'].values,
        'cna_n_deep': n_deep,
        'cna_n_any': n_any,
        'cna_n_samples': n_samples,
        'cna_freq_deep': n_deep / n_samples,
        'cna_freq_any': n_any / n_samples,
    })
    return out, sample_cols


def load_rna_zscore_genes(rna_path: str) -> set:
    """Read just gene-symbol column from RNA-z-score file to identify available genes."""
    df = pd.read_csv(rna_path, sep='\t', usecols=['Hugo_Symbol'], low_memory=False)
    return set(df['Hugo_Symbol'].dropna().astype(str).str.strip())


def build_panel(mut: pd.DataFrame, cna: pd.DataFrame, rna_genes: set,
                oncokb: set, mut_freq_thresh: float) -> tuple[list, pd.DataFrame]:
    """Return (panel_genes_list, panel_audit_df)."""
    mut['in_oncokb'] = mut['hugo_symbol'].isin(oncokb)
    mut['passes_mut_filter'] = (mut['mut_freq'] >= mut_freq_thresh) & mut['in_oncokb']

    cna['in_oncokb'] = cna['hugo_symbol'].isin(oncokb)
    cna['passes_cna_filter'] = (cna['cna_freq_deep'] >= CNA_DEEP_FREQ_THRESHOLD) & cna['in_oncokb']

    mut_panel = set(mut.loc[mut['passes_mut_filter'], 'hugo_symbol'])
    cna_panel = set(cna.loc[cna['passes_cna_filter'], 'hugo_symbol'])
    union = mut_panel | cna_panel

    # Intersect with RNA-available genes
    panel = union & rna_genes

    # Build audit table
    audit = pd.DataFrame({'hugo_symbol': sorted(panel)})
    audit = audit.merge(mut, on='hugo_symbol', how='left')
    audit = audit.merge(cna[['hugo_symbol', 'cna_freq_deep', 'cna_freq_any', 'passes_cna_filter']],
                        on='hugo_symbol', how='left')
    audit['in_mut_panel'] = audit['hugo_symbol'].isin(mut_panel)
    audit['in_cna_panel'] = audit['hugo_symbol'].isin(cna_panel)
    return sorted(panel), audit


def rank_and_cap(panel: list, audit: pd.DataFrame, cap: int = PANEL_TARGET_MAX) -> list:
    """If panel exceeds cap, rank by max(mut_freq, cna_freq_deep) and keep top cap."""
    if len(panel) <= cap:
        return panel
    a = audit.copy()
    a['score'] = a[['mut_freq', 'cna_freq_deep']].fillna(0).max(axis=1)
    return list(a.sort_values('score', ascending=False).head(cap)['hugo_symbol'])


def build_feature_matrix(panel: list, mutations_path: str, cna_path: str,
                         rna_path: str) -> pd.DataFrame:
    """Per-patient × per-gene feature matrix with 3 features per gene.

    Aligned on TCGA case_id (extracted from TCGA-XX-XXXX-NN sample barcodes).
    """
    # MUT: load full MAF, restrict to panel genes, build patient × gene binary matrix
    maf = pd.read_csv(mutations_path, sep='\t', comment='#',
                      usecols=['Hugo_Symbol', 'Tumor_Sample_Barcode',
                               'Variant_Classification'],
                      low_memory=False)
    coding = {
        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
        'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site', 'Translation_Start_Site',
        'Nonstop_Mutation',
    }
    maf = maf[maf['Variant_Classification'].isin(coding)]
    maf = maf[maf['Hugo_Symbol'].isin(panel)].copy()
    maf['case_id'] = maf['Tumor_Sample_Barcode'].str.slice(0, 12)
    mut_matrix = (maf.assign(v=1)
                    .pivot_table(index='case_id', columns='Hugo_Symbol',
                                 values='v', aggfunc='max', fill_value=0))
    mut_matrix = mut_matrix.reindex(columns=panel, fill_value=0)
    mut_matrix.columns = [f'{g}_mut' for g in mut_matrix.columns]

    # CNA: load full CNA matrix, restrict to panel genes, transpose to patient × gene
    cna = pd.read_csv(cna_path, sep='\t', low_memory=False)
    cna = cna.drop(columns=['Entrez_Gene_Id'], errors='ignore')
    cna = cna[cna['Hugo_Symbol'].isin(panel)]
    cna = cna.drop_duplicates(subset='Hugo_Symbol', keep='first').set_index('Hugo_Symbol').T
    cna.index = cna.index.str.slice(0, 12)  # sample barcode -> case_id
    cna = cna.groupby(cna.index).first()    # collapse if multiple samples per case
    cna = cna.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    cna = cna.reindex(columns=panel, fill_value=0)
    cna.columns = [f'{g}_cna' for g in cna.columns]

    # RNA: same pattern as CNA but with z-scores
    rna = pd.read_csv(rna_path, sep='\t', low_memory=False)
    rna = rna.drop(columns=['Entrez_Gene_Id'], errors='ignore')
    rna = rna[rna['Hugo_Symbol'].isin(panel)]
    rna = rna.drop_duplicates(subset='Hugo_Symbol', keep='first').set_index('Hugo_Symbol').T
    rna.index = rna.index.str.slice(0, 12)
    rna = rna.groupby(rna.index).mean()
    rna = rna.apply(pd.to_numeric, errors='coerce')
    rna = rna.reindex(columns=panel)
    rna.columns = [f'{g}_rna' for g in rna.columns]

    # Combine on common case_ids
    all_ids = sorted(set(mut_matrix.index) | set(cna.index) | set(rna.index))
    mut_matrix = mut_matrix.reindex(all_ids).fillna(0).astype(int)
    cna = cna.reindex(all_ids).fillna(0).astype(int)
    rna = rna.reindex(all_ids).fillna(0.0)
    out = pd.concat([mut_matrix, cna, rna], axis=1)
    out.index.name = 'case_id'
    return out


def curate(cohort_lower: str, cohort_upper: str, oncokb: set) -> None:
    paths = load_cbioportal_paths(cohort_lower, cohort_upper)
    out_dir = f'{ST_BASE}/{cohort_upper}/data/TCGA_{cohort_upper}/omic_panel'
    os.makedirs(out_dir, exist_ok=True)
    log_lines = [f'=== {cohort_upper} panel curation log ===\n']

    log_lines.append('Loading omic source files...')
    mut = compute_mutation_frequency(paths['mutations'])
    cna, cna_samples = compute_cna_frequency(paths['cna'])
    rna_genes = load_rna_zscore_genes(paths['rna'])
    log_lines.append(f'  mutation gene rows: {len(mut)}, total mutation samples: {mut.iloc[0]["n_total_samples_seen"]}')
    log_lines.append(f'  CNA gene rows: {len(cna)}, CNA samples: {len(cna_samples)}')
    log_lines.append(f'  RNA z-score genes available: {len(rna_genes)}')
    log_lines.append('')

    # First pass at MUT_FREQ_THRESHOLD = 5%
    panel, audit = build_panel(mut, cna, rna_genes, oncokb, MUT_FREQ_THRESHOLD)
    log_lines.append(f'Pass 1 (mut_freq >= 5%): {len(panel)} candidate genes after intersection with RNA')

    threshold_used = MUT_FREQ_THRESHOLD
    if len(panel) < PANEL_TARGET_MIN:
        log_lines.append(f'  → below target_min={PANEL_TARGET_MIN}; applying fallback mut_freq >= 3%')
        panel, audit = build_panel(mut, cna, rna_genes, oncokb, MUT_FREQ_FALLBACK)
        threshold_used = MUT_FREQ_FALLBACK
        log_lines.append(f'Pass 2 (mut_freq >= 3%): {len(panel)} candidate genes after intersection with RNA')

    if len(panel) > PANEL_TARGET_MAX:
        log_lines.append(f'  → above target_max={PANEL_TARGET_MAX}; ranking by max(mut_freq, cna_freq) and capping')
        panel = rank_and_cap(panel, audit, cap=PANEL_TARGET_MAX)

    log_lines.append(f'Final panel size: {len(panel)} genes')
    log_lines.append(f'Final feature dim: {len(panel)} × 3 = {len(panel)*3}')
    log_lines.append(f'Mut threshold used: {threshold_used*100:.0f}%')
    log_lines.append('')

    # Write gene list
    with open(f'{out_dir}/genes.txt', 'w') as f:
        for g in panel:
            f.write(g + '\n')
    log_lines.append(f'Wrote genes.txt ({len(panel)} genes)')

    # Write audit
    final_audit = audit[audit['hugo_symbol'].isin(panel)].copy()
    final_audit = final_audit.sort_values('hugo_symbol').reset_index(drop=True)
    final_audit.to_csv(f'{out_dir}/panel_meta.csv', index=False)
    log_lines.append(f'Wrote panel_meta.csv (per-gene decision audit)')

    # Build feature matrix
    log_lines.append('Building per-patient feature matrix (3 features per gene)...')
    fm = build_feature_matrix(panel, paths['mutations'], paths['cna'], paths['rna'])
    fm.to_csv(f'{out_dir}/feature_matrix.csv')
    log_lines.append(f'Wrote feature_matrix.csv  shape={fm.shape}')

    # Diagnostic: how many panel cases coverage with our 15-fold split master?
    splits_path = f'{ST_BASE}/{cohort_upper}/data/TCGA_{cohort_upper}/splits/15foldcv/master_splits.csv'
    if os.path.exists(splits_path):
        split_cases = set(pd.read_csv(splits_path)['case_id'])
        intersect = set(fm.index) & split_cases
        only_splits = split_cases - set(fm.index)
        log_lines.append('')
        log_lines.append(f'Split-vs-feature-matrix coverage:')
        log_lines.append(f'  cases in splits: {len(split_cases)}')
        log_lines.append(f'  cases in feature matrix: {len(fm.index)}')
        log_lines.append(f'  intersect: {len(intersect)}')
        log_lines.append(f'  in splits but missing omic features: {len(only_splits)}')

    log_text = '\n'.join(log_lines) + '\n'
    with open(f'{out_dir}/curation_log.txt', 'w') as f:
        f.write(log_text)
    print(log_text)


def main():
    oncokb = load_oncokb_genes()
    print(f'Loaded {len(oncokb)} OncoKB cancer genes from {ONCOKB_CSV}\n')
    curate('brca', 'BRCA', oncokb)
    print('\n' + '='*80 + '\n')
    curate('ucec', 'UCEC', oncokb)


if __name__ == '__main__':
    main()
