"""
HACA v1.5 — Hazard-Anchored Cross-Attention.

Single architectural intervention on top of MCAT_Surv:

  1. Per-patch hazard head (small MLP, ~17K params, single shared across
     all 6 pathway queries).
  2. Hazard-anchored co-attention: adds  lambda(epoch) * log(r_j + eps)
     as a per-key bias to the pre-softmax co-attention logits, via
     nn.MultiheadAttention's native `attn_mask` parameter.

The lambda(epoch) value is supplied by the training loop (haca_train_utils.py)
according to the AAS (Adaptive Anchoring Schedule) warmup; this module is
stateless w.r.t. training progress.

Aux NLL-Surv head (optional, default OFF) is also present but only contributes
to the loss when the training loop passes aux_weight > 0. When aux_weight==0
the aux outputs are computed but multiplied by zero, so gradient does not
flow into aux_classifier weights — behaviour identical to "no aux head".

Author: 2026-05-17. Design: HACA_DESIGN_v1_5.md.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import SNN_Block, Attn_Net_Gated, BilinearFusion
from models.model_coattn import MultiheadAttention


class HACA_Surv(nn.Module):
    """
    HACA v1.5 = MCAT + per-patch hazard head + Bayesian-posterior co-attention.

    Args identical to MCAT_Surv plus:
        haca_eps: float = 1e-3
            Log stabiliser. log(r + eps) is bounded above -ln(eps) ≈ -6.9.
    """

    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600],
                 n_classes=4, model_size_wsi: str = 'small',
                 model_size_omic: str = 'small', dropout=0.25,
                 haca_eps: float = 1e-3):
        super(HACA_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.haca_eps = haca_eps
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

        ### FC Layer over WSI bag (identical to MCAT)
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.wsi_net = nn.Sequential(*fc)

        ### Genomic SNN per pathway (identical to MCAT)
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Patch-level hazard head — NEW (HACA)
        # Maps a 256-d patch embedding to a single sigmoid risk score r_j.
        # ~17K params: Linear(256,64) -> ReLU -> Dropout -> Linear(64,1)
        self.patch_hazard_mlp = nn.Sequential(
            nn.Linear(size[1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        ### Multihead Co-Attention (identical to MCAT)
        # We will pass attn_mask=lambda*log(r) to bias keys.
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        ### Path Transformer + Attention Head (identical to MCAT)
        path_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Omic Transformer + Attention Head (identical to MCAT)
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Fusion Layer (identical to MCAT)
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[
                nn.Linear(256 * 2, size[2]), nn.ReLU(),
                nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        ### Classifier (identical to MCAT)
        self.classifier = nn.Linear(size[2], n_classes)

        ### Auxiliary NLL-Surv head — NEW (HACA, off by default)
        # CORRECTED 2026-05-17 (Option C):
        # Previous version pooled h_path_bag via attention weights, which
        # depended on r_j ONLY through the softmax bias. That created a
        # gradient bottleneck — empirically r_j stayed in [0.49, 0.52]
        # across 20 epochs in both aux=0.0 and aux=0.1 pilots.
        #
        # The corrected aux head operates on a 4-d SUMMARY of r_j directly
        # (mean / std / max / top-10% mean), bypassing the softmax. This
        # guarantees a non-zero gradient back to patch_hazard_mlp from the
        # NLL-Surv loss, regardless of the state of the attention layer.
        self.aux_classifier = nn.Linear(4, n_classes)

    def _build_attn_bias(self, r_j: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Build (K, N) attention mask to add to pre-softmax coattn logits.

        r_j : (N,)            per-patch sigmoid scores in (0, 1)
        lam : float           lambda(epoch) from the AAS schedule
        returns : (K=6, N)    broadcast bias = lam * log(r_j + eps)
        """
        log_r = torch.log(r_j + self.haca_eps)         # (N,)
        K = len(self.omic_sizes)                        # 6
        bias = (lam * log_r).unsqueeze(0).expand(K, -1) # (K, N)
        return bias

    def forward(self, **kwargs):
        """
        Forward pass.

        Expected kwargs (from coattn collate, identical to MCAT):
            x_path : (N, 1024)
            x_omic1 ... x_omic6 : per-pathway gene vectors

        HACA-specific kwargs:
            haca_lambda : float, default 0.0
                Anchoring strength for this forward pass. The training loop
                passes the AAS-scheduled value; eval passes the final value.
                When 0.0, behaviour is byte-identical to vanilla MCAT
                (modulo the *unused* patch_hazard_mlp / aux_classifier).

        Returns:
            hazards      : (1, n_classes)  main NLL-Surv hazards
            S            : (1, n_classes)  survival fn = cumprod(1 - hazards)
            Y_hat        : (1, 1)          argmax-class
            attn_scores  : dict            {coattn, path, omic, haca_r (NEW)}
            aux_hazards  : (1, n_classes)  aux head hazards (unused if aux_weight=0)
            aux_S        : (1, n_classes)  aux survival fn
        """
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]
        haca_lambda = float(kwargs.get('haca_lambda', 0.0))

        # Project WSI patches and omics — identical to MCAT
        h_path_bag = self.wsi_net(x_path).unsqueeze(1)     # (N, 1, 256)
        h_omic = [self.sig_networks[idx](sig) for idx, sig in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(1)      # (6, 1, 256)

        # --- HACA addition: per-patch hazard scores and attention bias ---
        # h_path_bag is (N, 1, 256); squeeze the batch dim for the MLP.
        patch_logits = self.patch_hazard_mlp(h_path_bag.squeeze(1))   # (N, 1)
        r_j = torch.sigmoid(patch_logits).squeeze(-1)                 # (N,)
        attn_bias = self._build_attn_bias(r_j, haca_lambda)           # (6, N)

        # Hazard-anchored co-attention.
        # nn.MultiheadAttention's attn_mask is float, added to attention
        # logits BEFORE softmax. Shape (L_q=6, L_k=N) is the contract.
        h_path_coattn, A_coattn = self.coattn(
            h_omic_bag, h_path_bag, h_path_bag,
            attn_mask=attn_bias,
        )

        ### Path branch — identical to MCAT
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze()

        ### Omic branch — identical to MCAT
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()

        ### Fusion + main classifier — identical to MCAT
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        # --- HACA addition: aux head with DIRECT r_j supervision ---
        # CORRECTED 2026-05-17 (Option C):
        # Pool r_j into 4 summary statistics that have a DIRECT gradient
        # path back to patch_hazard_mlp, bypassing the coattn softmax.
        # This is what the design intended; the previous version
        # accidentally routed aux gradient through the same softmax-bias
        # bottleneck as the main loss, which is why r_j never broke
        # symmetry under aux=0.0 or aux=0.1.
        N_topk = max(1, r_j.numel() // 10)        # top 10% of patches
        r_summary = torch.stack([
            r_j.mean(),
            r_j.std(),
            r_j.max(),
            r_j.topk(N_topk).values.mean(),
        ])                                                              # (4,)
        aux_logits = self.aux_classifier(r_summary).unsqueeze(0)        # (1, n_classes)
        aux_hazards = torch.sigmoid(aux_logits)
        aux_S = torch.cumprod(1 - aux_hazards, dim=1)

        attention_scores = {
            'coattn': A_coattn,
            'path': A_path,
            'omic': A_omic,
            'haca_r': r_j.detach(),   # for diagnostic histograms; not in graph
        }

        return hazards, S, Y_hat, attention_scores, aux_hazards, aux_S
