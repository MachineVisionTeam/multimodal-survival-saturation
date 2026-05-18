"""
GRFN-Pathomic network wrapper — drop-in replacement for PathgraphomicNet's
fusion + classifier block, using the Gaussian Random Fuzzy Number combination
rule defined in /home/sbarua/Region_based_segmentation/GRFN-Pathomic/
grfn_fusion.py.

Architecture spec: GRFN-Pathomic/GRFN_Pathomic_ARCHITECTURE.txt §3.

Strictly additive — this file does not modify any existing PF class; it
only defines PathgraphomicGRFNNet. The locked PF baseline (--mode
pathgraphomic) runs byte-identical when this file is not imported.

Author: 2026-05-17.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.nn import Parameter

# Pull GRFNFusion from the GRFN-Pathomic experiment folder. Kept there
# (not duplicated) so the GRFN module is the single source of truth.
sys.path.insert(0, '/home/sbarua/Region_based_segmentation/GRFN-Pathomic')
from grfn_fusion import GRFNFusion

from networks import GraphNet, MaxNet
from utils import dfs_freeze


class PathgraphomicGRFNNet(nn.Module):
    """
    Mirrors PathgraphomicNet (networks.py:534) but:
        - Replaces TrilinearFusion_A + Linear classifier with GRFNFusion.
        - hazard = mu_f (the GRFN combination's fused mean, used directly
          as the Cox risk).
        - features = mu_f stacked with sigma_f^2 and h_f and the three r_m
          for downstream calibration / explainability analysis.

    Ablation switches read from opt:
        opt.grfn_freeze_r : bool      — set True to disable reliability discounting (A1)
        opt.grfn_freeze_h : bool      — set True to fix h_m at 1 (A2)
        opt.grfn_init_lh  : float     — init log_h. Default 0 -> h_init = 1.0
        opt.grfn_init_lsv : float     — init log sigma^2. Default 0 -> sigma^2_init = 1.0
    """

    def __init__(self, opt, act, k):
        super(PathgraphomicGRFNNet, self).__init__()

        # ---------------- locked encoders ----------------
        # Same as PathgraphomicNet (networks.py:537-546). Loads checkpoints
        # from the existing 'graph' and 'omic' single-modality runs so the
        # encoder side is byte-identical to the PF baseline.
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
            print("[GRFN] Loaded encoders:\n",
                  os.path.join(opt.checkpoints_dir, opt.exp_name, 'graph', 'graph' + pt_fname), "\n",
                  os.path.join(opt.checkpoints_dir, opt.exp_name, 'omic', 'omic' + pt_fname))

        # ---------------- GRFN fusion ----------------
        # All three modalities are 32-d in the locked PF baseline (path_dim,
        # grph_dim, omic_dim all == 32 by default).
        assert opt.path_dim == opt.grph_dim == opt.omic_dim, \
            f'GRFN expects equal modality dims; got {opt.path_dim}/{opt.grph_dim}/{opt.omic_dim}'

        self.fusion = GRFNFusion(
            dim=opt.path_dim,
            init_lh=getattr(opt, 'grfn_init_lh', 0.0),
            init_lsv=getattr(opt, 'grfn_init_lsv', 0.0),
            freeze_r=getattr(opt, 'grfn_freeze_r', False),
            freeze_h=getattr(opt, 'grfn_freeze_h', False),
        )

        # ---------------- output activation ----------------
        # Match PF's PathgraphomicNet output convention so the Cox loss
        # operates on a comparably-scaled risk:
        #     hazard = sigmoid(scalar_in) * 6 - 3   in [-3, 3]
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        # Freeze locked encoders, same as PathgraphomicNet (networks.py:552-553).
        dfs_freeze(self.grph_net)
        dfs_freeze(self.omic_net)

    def forward(self, **kwargs):
        # path features come pre-extracted from the locked PathNet checkpoint
        # (same as PathgraphomicNet expects).
        path_vec = kwargs['x_path']
        grph_vec, _ = self.grph_net(x_grph=kwargs['x_grph'])
        omic_vec, _ = self.omic_net(x_omic=kwargs['x_omic'])

        mu_f, sigma_f_sq, h_f, r = self.fusion(path_vec, grph_vec, omic_vec)
        # mu_f shape (B,)

        # Apply the same range-scaled sigmoid that PathgraphomicNet applies
        # to its Linear classifier output. This keeps the Cox loss seeing
        # values in [-3, 3], matching the locked baseline.
        hazard = mu_f.unsqueeze(-1)            # (B, 1)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        # Pack the GRFN side outputs as `features` so they survive
        # train_test.py's interface (which expects (features, hazard)).
        # We stack: [mu_f, sigma_f_sq, h_f, r_p, r_g, r_o] -> (B, 6).
        features = torch.stack(
            [mu_f, sigma_f_sq, h_f, r[:, 0], r[:, 1], r[:, 2]], dim=1)

        return features, hazard

    def __hasattr__(self, name):
        """
        Custom hasattr expected by utils.regularize_MM_omic (introspects
        the module to find 'omic_net' / 'grph_net' / 'fusion'). Same
        implementation as PathgraphomicNet.__hasattr__ at networks.py:571.
        """
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False
