================================================================================
MCAT REPLICATION — DEPRECATION PATCHES
Documentation of every modification made to mahmoodlab/MCAT
================================================================================

Source repo : https://github.com/mahmoodlab/MCAT
Cloned date : 2026-05-14
Cloned at   : /home/sbarua/Region_based_segmentation/mcat_replication/MCAT/

Environment (ours vs. MCAT-2021):
  | Component   | MCAT (2021)    | Ours (2026)              |
  |-------------|----------------|--------------------------|
  | Python      | 3.7.7          | 3.10.12                  |
  | PyTorch     | 1.3.1 / 1.6.0  | 2.3.1+cu121              |
  | CUDA        | 10.1           | 12.1                     |
  | torchvision | 0.4 / 0.7      | 0.18.1+cu121             |
  | h5py        | 2.10.0         | 3.16.0                   |
  | pandas      | 0.25.3 / 1.1.3 | 2.3.3                    |
  | scipy       | 1.4.1          | (newer)                  |

POLICY: keep MCAT's model architecture, hyperparameters, splits, genomics,
loss, and training schedule EXACTLY as published. Apply only the minimum
deprecation patches required to make 2021 code run on 2026 stack. Document
every patch here.

================================================================================
PATCH #1 — _LinearWithBias removed from PyTorch API
================================================================================

File   : MCAT/models/model_coattn.py
Lines  : 474 (and used at line 539)
Date   : 2026-05-15

Symptom:
    ImportError: cannot import name '_LinearWithBias' from
    'torch.nn.modules.linear'

Cause:
    `torch.nn.modules.linear._LinearWithBias` was a private class in PyTorch
    <=1.6 that wrapped `nn.Linear` with `bias=True` as the default. It was
    removed from PyTorch's public API in 1.7+. MCAT includes a frozen copy
    of the 2020-era `MultiheadAttention` class that imports this private
    symbol.

Fix:
    Replaced the import with an alias to `torch.nn.Linear`. Since
    `_LinearWithBias(in, out)` was functionally identical to
    `nn.Linear(in, out, bias=True)` (and bias=True is `nn.Linear`'s default),
    this is a zero-behavior-change patch.

Diff (before):
    from torch.nn.modules.linear import _LinearWithBias
    from torch.nn.init import xavier_uniform_

Diff (after):
    # PATCH (2026-05-15): `_LinearWithBias` was a private class in PyTorch <=1.6
    # and was removed in later versions. It was equivalent to `nn.Linear` with
    # the default `bias=True`, so we substitute `nn.Linear` directly.
    from torch.nn import Linear as _LinearWithBias
    from torch.nn.init import xavier_uniform_

Verification:
    After this patch, `from models.model_coattn import MCAT_Surv` succeeds.
    MCAT_Surv(omic_sizes=[100]*6, n_classes=4) constructs a 3.78M-param model.
    No behavioral change in forward pass.

================================================================================
PATCH #2 — missing --inst_loss argparse entry in main.py
================================================================================

File   : MCAT/main.py
Lines  : ~120 (added next to --bag_loss)
Date   : 2026-05-16

Symptom:
    AttributeError: 'Namespace' object has no attribute 'inst_loss'
    Raised at main.py:164 inside the settings dict construction.

Cause:
    main.py's settings dict references `args.inst_loss`, but the argparse
    block in the cloned repo never registers `--inst_loss`. Repo-side
    omission, not a deprecation. Confirmed by inspecting MCAT's own
    published `experiment_*.txt` logs (results/ICCV/AMILsm_*/), which all
    record `'inst_loss': None` — the field existed in the original repo
    with default=None.

Fix:
    Restored the missing argparse line with default=None and CLAM-style
    choices. inst_loss is referenced only in eval_utils.py (evaluation path,
    not training), so survival training is unaffected — this only un-breaks
    settings dict construction.

Diff (added after --bag_loss):
    parser.add_argument('--inst_loss', type=str,
                        choices=['svm', 'ce', None], default=None,
                        help='instance-level clustering loss function (default: None)')

Verification:
    main.py loads, args.inst_loss == None, settings dict built without
    error, all 4 GPU procs advance past arg parsing.

================================================================================
PATCH #3 — wrong dir name in dataset_survival.py (datasets_csv_sig typo)
================================================================================

File   : MCAT/ (filesystem-level symlink)
Date   : 2026-05-16

Symptom:
    FileNotFoundError: './dataset_csv_sig/signatures.csv'

Cause:
    datasets/dataset_survival.py:126 reads `./dataset_csv_sig/signatures.csv`
    but the directory on disk is `./datasets_csv_sig/` (extra plural 's').
    Repo-side typo, not a deprecation.

Fix:
    Symlinked the expected name to the on-disk name. Pure filesystem fix,
    no code change.

Diff:
    ln -s datasets_csv_sig dataset_csv_sig

Verification:
    `./dataset_csv_sig/signatures.csv` resolves; pd.read_csv loads the
    1,537-gene × 6-category signature table.

================================================================================
PATCH #4 — fast_cluster_ids.pkl loaded unconditionally for all modes
================================================================================

File   : MCAT/datasets/dataset_survival.py
Line   : 329 (Generic_Split.__init__)
Date   : 2026-05-16

Symptom:
    FileNotFoundError: '<data_dir>/fast_cluster_ids.pkl'
    Raised when constructing Generic_Split for mode='coattn'.

Cause:
    Generic_Split.__init__ unconditionally loads `fast_cluster_ids.pkl`,
    but the variable it sets (`self.fname2ids`) is only consumed at
    line 269 inside the `mode == 'cluster'` branch. The pkl is a
    cluster-mode artifact that MCAT does not ship for non-cluster modes;
    coattn/omic/pathomic modes never need it. Repo-side mode-coupling bug.

Fix:
    Wrapped the load in `if self.mode == 'cluster':`. Cluster mode behavior
    is preserved exactly; other modes skip the load.

Diff (line 329):
    -        with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
    -            self.fname2ids = pickle.load(handle)
    +        if self.mode == 'cluster':
    +            with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
    +                self.fname2ids = pickle.load(handle)

Verification:
    coattn mode constructs Generic_Split successfully, omic_sizes is
    populated from the gene signature intersection, training proceeds.

================================================================================
PATCH #5 — missing --testing argparse entry in main.py
================================================================================

File   : MCAT/main.py
Lines  : ~123 (added next to --inst_loss)
Date   : 2026-05-16

Symptom:
    AttributeError: 'Namespace' object has no attribute 'testing'
    Raised at utils/core_utils.py:178 inside get_split_loader(...).

Cause:
    Same class as PATCH #2: utils/core_utils.py references args.testing,
    but the cloned main.py never registers --testing.

Fix:
    Added the missing argparse line with default=False. testing=True is a
    debug-only path that short-circuits loader construction; default=False
    matches the normal training behavior.

Diff (added after --inst_loss):
    parser.add_argument('--testing', action='store_true', default=False,
                        help='Debugging flag (default: False)')

Verification:
    get_split_loader proceeds with testing=False, train/val DataLoaders
    constructed, first epoch's batch iteration begins. All 4 GPU procs
    pass arg parsing and reach the training loop (memory 2.5-4 GB/GPU
    observed during dataloader warmup).

================================================================================
PATCH #6 — final summary_survival not dispatched for coattn mode
================================================================================

File   : MCAT/utils/core_utils.py
Line   : 204 (inside train(), just after checkpoint save/load)
Date   : 2026-05-16

Symptom:
    ValueError: too many values to unpack (expected 5)
    Raised at utils/core_utils.py:333 inside summary_survival when the
    coattn collate yields a 10-tuple (path + 6 omics + label + event_time
    + c) but the function tries to unpack 5.

Cause:
    Per-epoch train/validate loops at lines 195-200 correctly branch on
    `args.mode == 'coattn'` and call train_loop_survival_coattn /
    validate_survival_coattn (defined in utils/coattn_train_utils.py).
    But the FINAL summary call at line 204 always invokes
    summary_survival, which expects the non-coattn 5-tuple collate.
    summary_survival_coattn already exists in coattn_train_utils.py (line
    133) and is imported via `from utils.coattn_train_utils import *` at
    line 18 — the dispatch is simply missing.

Fix:
    Added an `if args.mode == 'coattn':` branch mirroring the per-epoch
    pattern at lines 195-200.

Diff (line 204):
    -    results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)
    +    if args.mode == 'coattn':
    +        results_val_dict, val_cindex = summary_survival_coattn(model, val_loader, args.n_classes)
    +    else:
    +        results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)

Verification:
    First crash had completed 20 epochs of training cleanly on all 4 GPUs
    (confirmed by 20 epoch lines per log) — only the post-training
    summary aborted. With patch + PATCH #7 (np.asscalar), the summary
    runs to completion and writes split_latest_val_{i}_results.pkl.

================================================================================
PATCH #7 — np.asscalar removed in NumPy 1.23+
================================================================================

Files  : MCAT/utils/coattn_train_utils.py:161-163
         MCAT/utils/core_utils.py:351-353
         MCAT/utils/cluster_train_utils.py:177-179
Date   : 2026-05-16

Symptom:
    AttributeError: module 'numpy' has no attribute 'asscalar'.
    (Triggered inside the summary_*_survival* functions during the final
    per-fold evaluation pass.)

Cause:
    `np.asscalar(x)` was deprecated in NumPy 1.16 and removed in 1.23+.
    We run NumPy 2.2.6. MCAT calls it 3 times per summary function to
    convert 0-d/1-elem tensors and arrays into Python scalars for the
    saved patient_results dict.

Fix:
    Replaced each call with `.item()`, which is the idiomatic replacement
    and works uniformly across torch tensors (event_time, c) and 1-elem
    torch outputs (risk = -sum(survival, dim=1)). The downstream
    consumers (numpy assignment, concordance_index_censored, pickle)
    accept Python floats identically, so this is a zero-behavior-change
    patch.

Diff (one block, applied identically to all 3 files):
    -        risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy())
    -        event_time = np.asscalar(event_time)
    -        c = np.asscalar(c)
    +        risk = (-torch.sum(survival, dim=1)).item()
    +        event_time = event_time.item()
    +        c = c.item()

Verification:
    `grep -rn "np\.asscalar(" MCAT/utils/` returns no remaining call
    sites (only comment references in the patch headers). End-to-end
    re-run pending.

================================================================================
END (more patches to be added below as discovered during training)
================================================================================
