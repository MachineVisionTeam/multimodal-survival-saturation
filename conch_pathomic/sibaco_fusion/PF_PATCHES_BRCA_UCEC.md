# PathomicFusion tree patches — BRCA + UCEC 2-modal wiring

These are the minimal additive patches applied to the PF tree to make
`pathomic_conch_flat` (Cell A') and `pathomic_conch_sibaco` (2-modal SiBaCo) modes
dataroot-aware. They DO NOT change behaviour for KIRC/GBMLGG/BLCA.

## Files patched

### `PathomicFusion/train_cv.py`

Made the 2-modal flat / SiBaCo split-pkl branch dataroot-aware so BRCA/UCEC each
load their own cohort pkl. Original BLCA branch is preserved as the `else`.

```python
elif opt.mode in ('pathomic_conch_flat', 'pathomic_conch_sibaco'):
    # 2-modal flat-omic CONCH-Bimodal: Cell A' baseline + SiBaCo share the SAME
    # pkl; only the fusion module differs. Dataroot-aware so BLCA/BRCA/UCEC each
    # load their own pkl. BRCA+UCEC added under SiBaCo 2x2 amendment (8cafd1e).
    if 'BRCA' in opt.dataroot:
        data_cv_path = '%s/splits/brca_st_conch_bimodal_flat.pkl' % opt.dataroot
    elif 'UCEC' in opt.dataroot:
        data_cv_path = '%s/splits/ucec_st_conch_bimodal_flat.pkl' % opt.dataroot
    else:
        data_cv_path = '%s/splits/BLCA_st_conch_bimodal_flat.pkl' % opt.dataroot
```

### `PathomicFusion/data_loaders.py` — UNCHANGED

The `pathomic_conch_flat` / `pathomic_conch_sibaco` modes already work for any
2-modal pkl with the BLCA-format `cv_splits` dict; BRCA/UCEC use identical structure.

### `PathomicFusion/networks.py` — UNCHANGED

The 2-modal SiBaCo dispatch via `opt.mode='pathomic_conch_sibaco'` already routes
to `PathomicCONCHBimodalSiBaCoNet`. The model is byte-identical to what BLCA used;
only `opt.input_size_omic` differs per cohort (BRCA=378, UCEC=390, BLCA=20430).

## Python environment additions

These packages were installed in `/home/sbarua/.venv` during Phase 8l smoke-testing.
They are dependencies of the existing PF code (not new), and were absent from the
venv at the start of Phase 8l (likely lost in a prior venv rebuild). All
SiBaCo/Cell-A' training requires them present:

```bash
pip install einops_exts   # required by CONCHv1.5 model loader (TRIDENT)
pip install tables        # required by PF networks.py
pip install torch_geometric  # required by PF networks.py (GraphNet imports)
pip install lifelines     # required by PF utils.py
```

## Training command template (per cohort)

For each of {BRCA, UCEC}, two runs (Cell A' baseline + SiBaCo) per cohort,
15 folds each, identical args except `--mode` and `--exp_name`:

```bash
cd PathomicFusion
CUDA_VISIBLE_DEVICES=0 python train_cv.py \
    --exp_name <baseline|sibaco>_<cohort> \
    --dataroot ./data/TCGA_<COHORT> \
    --checkpoints_dir ./runs_modern/TCGA_<COHORT>/ \
    --task surv --mode <pathomic_conch_flat|pathomic_conch_sibaco> \
    --model_name <pathomic_conch_flat|pathomic_conch_sibaco> \
    --niter 10 --niter_decay 20 --batch_size 32 \
    --use_vgg_features 1 --use_rnaseq 0 --input_size_omic <378|390> \
    --grph_dim 32 --omic_dim 32 --path_dim 32 \
    --measure 1 --skip 0 --use_bilinear 1 \
    --path_gate 1 --grph_gate 1 --omic_gate 0 \
    --fusion_type pofusion --gpu_ids 0
```

`input_size_omic` per cohort: **BRCA = 378, UCEC = 390** (matches `omic_panel/genes.txt`
size × 3 omic-types).
