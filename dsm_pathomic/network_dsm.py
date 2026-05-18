"""
DSM-Pathomic network wrapper — replaces PathgraphomicNet's TrilinearFusion_A
+ Linear classifier + sigmoid output with a modality-specific Deep Survival
Machines head (loss-function lever of the saturation paper).

Architecture spec: /home/sbarua/Region_based_segmentation/DSM-Pathomic/
DSM_Pathomic_ARCHITECTURE.txt §2-§3.

Strictly additive — this file does not modify any existing PF class. The
locked PF baseline (--mode pathgraphomic) runs byte-identical when this
file is not imported.

Key difference from PathgraphomicGRFNNet:
  - The DSM head is NOT a parameter-free combination rule. It computes its
    own loss (DSMFusion.nll_loss), which replaces CoxLoss entirely.
  - The forward output `hazard` is the Cox-comparable risk -E[T], used for
    c-Index ranking only. It is NOT passed into CoxLoss; train_test.py
    must dispatch to `compute_dsm_loss(t, censor)` instead when this model
    type is used.

Author: 2026-05-18.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.nn import Parameter

# Pull DSMFusion from the DSM-Pathomic experiment folder.
sys.path.insert(0, '/home/sbarua/Region_based_segmentation/DSM-Pathomic')
from dsm_fusion import DSMFusion

from networks import GraphNet, MaxNet
from utils import dfs_freeze


class PathgraphomicDSMNet(nn.Module):
    """
    Mirrors PathgraphomicNet (networks.py:534) but:
        - Replaces TrilinearFusion_A + Linear classifier with DSMFusion.
        - Replaces the Cox partial-likelihood loss with the DSM mixture
          negative log-likelihood (call compute_dsm_loss after forward()).
        - hazard = -E[T] (Cox-comparable scalar risk for c-Index ranking).
        - features stack: (μ-of-mixture, σ-of-mixture, max-weight) per
          modality for downstream calibration / explainability analysis.

    Ablation switches read from opt:
        opt.dsm_K              : Weibull components per modality (default 4)
        opt.dsm_alpha_min      : Weibull shape clamp lower (default 0.3)
        opt.dsm_alpha_max      : Weibull shape clamp upper (default 10.0)
        opt.dsm_beta_min       : Weibull scale clamp lower (default 1.0 day)
        opt.dsm_beta_max       : Weibull scale clamp upper (default 10000.0 days)
    """

    def __init__(self, opt, act, k):
        super(PathgraphomicDSMNet, self).__init__()

        # ---------------- locked encoders (identical to PathgraphomicNet) ----------------
        self.grph_net = GraphNet(
            grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate,
            use_edges=1, pooling_ratio=0.20,
            label_dim=opt.label_dim, init_max=False,
        )
        self.omic_net = MaxNet(
            input_dim=opt.input_size_omic, omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate, act=act,
            label_dim=opt.label_dim, init_max=False,
        )

        if k is not None:
            pt_fname = '_%d.pt' % k
            best_grph_ckpt = torch.load(os.path.join(
                opt.checkpoints_dir, opt.exp_name, 'graph', 'graph' + pt_fname),
                map_location=torch.device('cpu'))
            best_omic_ckpt = torch.load(os.path.join(
                opt.checkpoints_dir, opt.exp_name, 'omic', 'omic' + pt_fname),
                map_location=torch.device('cpu'))
            self.grph_net.load_state_dict(best_grph_ckpt['model_state_dict'])
            self.omic_net.load_state_dict(best_omic_ckpt['model_state_dict'])
            print("[DSM] Loaded encoders:\n",
                  os.path.join(opt.checkpoints_dir, opt.exp_name, 'graph', 'graph' + pt_fname), "\n",
                  os.path.join(opt.checkpoints_dir, opt.exp_name, 'omic', 'omic' + pt_fname))

        # ---------------- DSM fusion head ----------------
        assert opt.path_dim == opt.grph_dim == opt.omic_dim, \
            f'DSM expects equal modality dims; got {opt.path_dim}/{opt.grph_dim}/{opt.omic_dim}'

        self.fusion = DSMFusion(
            dim=opt.path_dim,
            K=getattr(opt, 'dsm_K', 4),
            alpha_min=getattr(opt, 'dsm_alpha_min', 0.3),
            alpha_max=getattr(opt, 'dsm_alpha_max', 10.0),
            beta_min=getattr(opt, 'dsm_beta_min', 1.0),
            beta_max=getattr(opt, 'dsm_beta_max', 10000.0),
        )

        # No output activation: DSM's Cox-comparable risk is -E[T], typically
        # in [-thousands, 0]. The sigmoid+[-3, +3] convention used by
        # PathgraphomicNet is irrelevant here because we don't feed `hazard`
        # into CoxLoss — we use DSMFusion.nll_loss directly.
        self.act = None
        self.output_range = Parameter(torch.FloatTensor([1.0]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([0.0]), requires_grad=False)

        # Freeze locked encoders, same as PathgraphomicNet.
        dfs_freeze(self.grph_net)
        dfs_freeze(self.omic_net)

        # Cache for loss computation (filled by forward, consumed by
        # compute_dsm_loss). Stored as attributes (not buffers) since
        # they're per-forward.
        self._cached_h_p = None
        self._cached_h_g = None
        self._cached_h_o = None

    def forward(self, **kwargs):
        """
        Forward pass mirrors PathgraphomicNet.forward signature.

        Returns
        -------
        features : (B, 6)  diagnostic vector [α_mean, β_mean (days), w_path_max,
                          w_graph_max, w_omic_max, num_components]. Used by
                          train_test.py's downstream pkl-saving; not used in
                          the loss.
        hazard   : (B, 1)  Cox-comparable risk = -E[T]. Used by c-Index
                          ranking only — NOT fed into CoxLoss.
        """
        path_vec = kwargs['x_path']
        grph_vec, _ = self.grph_net(x_grph=kwargs['x_grph'])
        omic_vec, _ = self.omic_net(x_omic=kwargs['x_omic'])

        # Cache for compute_dsm_loss
        self._cached_h_p = path_vec
        self._cached_h_g = grph_vec
        self._cached_h_o = omic_vec

        # Predict params + compute scalar risk
        alphas, betas, log_weights = self.fusion.predict_params(path_vec, grph_vec, omic_vec)
        weights = log_weights.exp()
        # E[T] = sum_k w_k * beta_k * Gamma(1 + 1/alpha_k)
        per_comp_mean = betas * torch.exp(torch.lgamma(1.0 + 1.0 / alphas))
        E_T = (weights * per_comp_mean).sum(dim=1)                 # (B,)
        hazard = (-E_T).unsqueeze(-1)                              # (B, 1)

        # Diagnostic features: per-modality max weight, mean alpha/beta
        K = self.fusion.K
        # Components 0..K-1 are path; K..2K-1 graph; 2K..3K-1 omic
        w_path = weights[:, 0:K].sum(dim=1)
        w_graph = weights[:, K:2*K].sum(dim=1)
        w_omic = weights[:, 2*K:3*K].sum(dim=1)
        alpha_mean = alphas.mean(dim=1)
        beta_mean = betas.mean(dim=1)
        ncomp = torch.full_like(alpha_mean, float(3*K))
        features = torch.stack([alpha_mean, beta_mean, w_path, w_graph, w_omic, ncomp], dim=1)

        return features, hazard

    # ------------------------------------------------------------------
    # Custom loss called by train_test.py instead of CoxLoss
    # ------------------------------------------------------------------

    def compute_dsm_loss(self, survtime: torch.Tensor, censor: torch.Tensor):
        """
        Returns the DSM mixture negative log-likelihood on the most recent
        forward batch.

        survtime : (B,)  PF's t variable (in days)
        censor   : (B,)  PF's local "censor" variable — actually the EVENT
                          indicator. See dsm_fusion.nll_loss docstring.
        """
        if self._cached_h_p is None:
            raise RuntimeError('compute_dsm_loss called before forward')
        return self.fusion.nll_loss(
            t=survtime, event=censor,
            h_p=self._cached_h_p, h_g=self._cached_h_g, h_o=self._cached_h_o,
        )

    # ------------------------------------------------------------------
    # Native S(t|x) for IBS / IBLL (no Breslow needed)
    # ------------------------------------------------------------------

    def survival_at_times(self, t_grid: torch.Tensor):
        """
        Predicted S(t | x) on the most-recent forward batch, evaluated at
        t_grid. Returns (B, T) — patient × time. The "no Breslow" promise.
        """
        if self._cached_h_p is None:
            raise RuntimeError('survival_at_times called before forward')
        return self.fusion.survival_at_times(
            t_grid, self._cached_h_p, self._cached_h_g, self._cached_h_o
        )

    def __hasattr__(self, name):
        """Custom hasattr expected by utils.regularize_MM_omic. Same as
        PathgraphomicNet.__hasattr__ at networks.py:571."""
        if '_parameters' in self.__dict__:
            if name in self.__dict__['_parameters']:
                return True
        if '_buffers' in self.__dict__:
            if name in self.__dict__['_buffers']:
                return True
        if '_modules' in self.__dict__:
            if name in self.__dict__['_modules']:
                return True
        return False
