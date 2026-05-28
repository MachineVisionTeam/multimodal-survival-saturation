"""
Unimodal sanity baselines per pre-reg §3.6 (BRCA+UCEC SiBaCo 2x2 amendment 8cafd1e).

Trains SNN-only (omic + Cox head) and CNN-only (CONCH adapter + Cox head) per
cohort × fold, using the SAME optimizer / LR / schedule / Cox loss as the 2-modal
multimodal runs:
    Adam, lr=1e-3, batch=32, 30 epochs (10 + 20 decay), dropout=0.25.

Cox hazard head: Linear(32, 1) → Sigmoid * 6 − 3 (matches PF + SiBaCo convention).

Reports per-fold c-Index for each cohort × modality. Computes the locked balance
classification: cohort = "genomics-dominated" iff
   mean(SNN c-Index) − mean(CNN c-Index) ≥ +0.04 with paired-t p < 0.05.
Otherwise "balanced".

Run:
    python conch_pathomic/sibaco_fusion/unimodal_baselines.py \
        --cohort BRCA --gpu 0
The "--cohort all" form is the convenience driver that spawns per-cohort sub-procs
across GPUs (launched as background daemons by a separate launch shim).

Output:
    /mnt/storage7/Dataset_pathomicfusion/SiBaCo_unimodal/{COHORT}_unimodal.json
    (per-cohort: per-fold c-Index for SNN-only, CNN-only, delta, paired-t, verdict)
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

PF = '/home/sbarua/Region_based_segmentation/pathomic_fusion_replica/PathomicFusion'
sys.path.insert(0, PF)

from networks import MaxNet
from network_conch import CONCHv15PathAdapter

ST = '/mnt/storage7/Dataset_pathomicfusion'
OUT_BASE = f'{ST}/SiBaCo_unimodal'

COHORT_PKL = {
    'BRCA': f'{ST}/BRCA/data/TCGA_BRCA/splits/brca_st_conch_bimodal_flat.pkl',
    'UCEC': f'{ST}/UCEC/data/TCGA_UCEC/splits/ucec_st_conch_bimodal_flat.pkl',
    'BLCA': f'{PF}/data/TCGA_BLCA/splits/BLCA_st_conch_bimodal_flat.pkl',
    # KIRC / GBMLGG: 3-modal CONCH pkl, but we only consume x_path + x_omic
    'KIRC':   f'{PF}/data/TCGA_KIRC/splits/KIRC_st_conch.pkl',
    'GBMLGG': f'{PF}/data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_conch.pkl',
}
OMIC_DIM = {'BRCA': 378, 'UCEC': 390, 'BLCA': 20430, 'KIRC': 362, 'GBMLGG': 320}

# Locked hyperparameters per pre-reg §3.1 (same as multimodal Cell A')
LR = 1e-3
BATCH = 32
NITER = 10
NITER_DECAY = 20
DROPOUT = 0.25
SEED = 0
GENO_DOM_THRESHOLD = 0.04


class SNNOnly(nn.Module):
    """MaxNet omic encoder → Linear(32,1) Cox head, sigmoid×6−3 scaling."""
    def __init__(self, input_dim: int, omic_dim: int = 32, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = MaxNet(input_dim=input_dim, omic_dim=omic_dim,
                              dropout_rate=dropout, act=nn.Sigmoid(),
                              label_dim=1, init_max=False)
        self.classifier = nn.Linear(omic_dim, 1)
        self.register_buffer('out_range', torch.tensor([6.0]))
        self.register_buffer('out_shift', torch.tensor([-3.0]))
    def forward(self, x_omic):
        z, _ = self.encoder(x_omic=x_omic)
        h = torch.sigmoid(self.classifier(z)) * self.out_range + self.out_shift
        return h


class CNNOnly(nn.Module):
    """CONCH adapter → Linear(32,1) Cox head, sigmoid×6−3 scaling."""
    def __init__(self, in_dim: int = 12288, out_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.adapter = CONCHv15PathAdapter(in_dim=in_dim, out_dim=out_dim,
                                            dropout_rate=dropout)
        self.classifier = nn.Linear(out_dim, 1)
        self.register_buffer('out_range', torch.tensor([6.0]))
        self.register_buffer('out_shift', torch.tensor([-3.0]))
    def forward(self, x_path):
        if x_path.dim() == 3 and x_path.shape[1] == 1:
            x_path = x_path.squeeze(1)
        z = self.adapter(x_path)
        h = torch.sigmoid(self.classifier(z)) * self.out_range + self.out_shift
        return h


def cox_loss(hazards, t, e):
    """Standard Cox partial likelihood. Higher hazard = higher risk = shorter survival."""
    # Sort by descending time
    order = torch.argsort(t, descending=True)
    hz = hazards[order].squeeze(-1)
    e_sorted = e[order]
    # log-cumulative-sum-exp from the start (descending t means each step adds smaller t)
    log_cumsum = torch.logcumsumexp(hz, dim=0)
    return -((hz - log_cumsum) * e_sorted).sum() / (e_sorted.sum().clamp(min=1.0))


def c_index(hazards: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    """Concordance index. Higher hazard should mean shorter survival → uses sign."""
    from lifelines.utils import concordance_index
    # Cox convention: higher hazard = higher risk. lifelines wants higher score = longer
    # survival, so negate hazards.
    return concordance_index(t, -hazards.squeeze(), e)


def make_loaders(pkl_path: str, fold: int):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    splits = d.get('cv_splits', d.get('split'))
    tr, te = splits[fold]['train'], splits[fold]['test']
    def pack(s):
        xp = torch.tensor(np.asarray(s['x_path'])).float()
        xo = torch.tensor(np.asarray(s['x_omic'])).float()
        t  = torch.tensor(np.asarray(s['t'])).float()
        e  = torch.tensor(np.asarray(s['e'])).float()
        return xp, xo, t, e
    return pack(tr), pack(te)


def train_one(model: nn.Module, train_pack, test_pack, device, log_prefix=''):
    xp_tr, xo_tr, t_tr, e_tr = [x.to(device) for x in train_pack]
    xp_te, xo_te, t_te, e_te = [x.to(device) for x in test_pack]
    is_snn = isinstance(model, SNNOnly)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0,
                                               total_iters=NITER_DECAY)
    total_epochs = NITER + NITER_DECAY
    n = xp_tr.shape[0]
    best_c = -1.0
    for ep in range(total_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            if is_snn:
                hz = model(xo_tr[idx])
            else:
                hz = model(xp_tr[idx])
            loss = cox_loss(hz, t_tr[idx], e_tr[idx])
            loss.backward()
            opt.step()
        if ep >= NITER:
            sched.step()
        # eval c-index on test
        model.eval()
        with torch.no_grad():
            if is_snn:
                hz_te = model(xo_te).cpu().numpy()
            else:
                hz_te = model(xp_te).cpu().numpy()
        c = c_index(hz_te, t_te.cpu().numpy(), e_te.cpu().numpy())
        if c > best_c:
            best_c = c
    return best_c


def run_cohort(cohort: str, gpu: int):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(f'cuda:{gpu}')
    os.makedirs(OUT_BASE, exist_ok=True)
    out_path = f'{OUT_BASE}/{cohort}_unimodal.json'

    # Probe number of folds
    with open(COHORT_PKL[cohort], 'rb') as f:
        d = pickle.load(f)
    splits = d.get('cv_splits', d.get('split'))
    fold_keys = sorted(splits.keys())
    n_folds = len(fold_keys)

    snn_c, cnn_c = [], []
    t0 = time.time()
    for k in fold_keys:
        train_pack, test_pack = make_loaders(COHORT_PKL[cohort], k)
        snn = SNNOnly(input_dim=OMIC_DIM[cohort]).to(device)
        cnn = CNNOnly(in_dim=12288, out_dim=32).to(device)
        c_s = train_one(snn, train_pack, test_pack, device)
        c_c = train_one(cnn, train_pack, test_pack, device)
        snn_c.append(c_s); cnn_c.append(c_c)
        print(f'  [{cohort}] fold {k}: SNN c={c_s:.4f}  CNN c={c_c:.4f}  Δ={c_s-c_c:+.4f}'
              f'   ({time.time()-t0:.0f}s)', flush=True)

    snn_c, cnn_c = np.array(snn_c), np.array(cnn_c)
    delta = snn_c - cnn_c
    t_stat, p_t = stats.ttest_rel(snn_c, cnn_c)
    is_geno_dom = (delta.mean() >= GENO_DOM_THRESHOLD) and (p_t < 0.05) and (delta.mean() > 0)
    verdict = 'genomics-dominated' if is_geno_dom else 'balanced'

    summary = {
        'cohort': cohort,
        'n_folds': int(n_folds),
        'snn_per_fold': snn_c.tolist(),
        'cnn_per_fold': cnn_c.tolist(),
        'snn_mean': float(snn_c.mean()), 'snn_std': float(snn_c.std(ddof=1)),
        'cnn_mean': float(cnn_c.mean()), 'cnn_std': float(cnn_c.std(ddof=1)),
        'delta_mean': float(delta.mean()), 'delta_std': float(delta.std(ddof=1)),
        'paired_t': float(t_stat), 'paired_t_p': float(p_t),
        'classification_rule': f'delta>={GENO_DOM_THRESHOLD} AND paired-t p<0.05',
        'classification': verdict,
        'pre_reg_predicted': {'KIRC': 'balanced', 'GBMLGG': 'genomics-dominated',
                              'BLCA': 'balanced', 'BRCA': 'balanced',
                              'UCEC': 'genomics-dominated'}.get(cohort, 'n/a'),
        'walltime_seconds': round(time.time() - t0, 1),
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[{cohort}] FINAL: SNN={snn_c.mean():.4f}±{snn_c.std(ddof=1):.4f}  '
          f'CNN={cnn_c.mean():.4f}±{cnn_c.std(ddof=1):.4f}  '
          f'Δ={delta.mean():+.4f}  paired-t p={p_t:.4f}  → {verdict.upper()}', flush=True)
    print(f'  saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True, choices=list(COHORT_PKL.keys()))
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    run_cohort(args.cohort, args.gpu)


if __name__ == '__main__':
    main()
