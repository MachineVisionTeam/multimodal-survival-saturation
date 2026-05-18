"""
GRFN-Pathomic fusion module — Gaussian Random Fuzzy Number evidential
combination, designed as a drop-in replacement for Pathomic Fusion's
TrilinearFusion_A block.

Architecture spec: GRFN_Pathomic_ARCHITECTURE.txt §2 in this folder.

Inputs:  three patient-level 32-d modality embeddings (h_p, h_g, h_o)
         from the locked PF VGG / GCN / SNN encoders.

Outputs: mu_f      — Cox risk scalar per patient                    shape (B,)
         sigma_f_sq — propagated aleatoric variance                  shape (B,)
         h_f       — fused epistemic confidence (sum of disc. h)    shape (B,)
         r         — per-modality reliability per patient            shape (B, 3)
                     ordered as [r_path, r_graph, r_omic]

Cox training: pass mu_f as the risk to the existing CoxLoss in
pathomic_fusion_replica/PathomicFusion/utils.py — no other loss change.

Author: 2026-05-17.
"""

import torch
import torch.nn as nn


class GRFNFusion(nn.Module):
    """
    Closed-form GRFN combination of three modality-specific fuzzy survival
    predictions.

    Per-modality head emits (mu, log_sigma_sq, log_h).
    Reliability head emits 3 logits from the concatenated embedding,
    sigmoid-ed to per-modality r in (0, 1).

    Ablation switches:
        freeze_r : if True, r_m identically 1 (no reliability discounting)
                   — corresponds to ablation A1 in the architecture doc
        freeze_h : if True, h_m identically 1 (uniform epistemic precision)
                   — corresponds to ablation A2

    init_h : initial value of h_m at model-init. Default 1.0 (log_h = 0)
             which makes the trivial-limit check (uniform mean) clean.
    init_lsv : initial value of log sigma^2 at model-init. Default 0
             (sigma^2 = 1). Chosen so that early-training variance is on
             a comparable scale to early-training mu (both O(1)).
    """

    def __init__(self,
                 dim: int = 32,
                 init_lh: float = 0.0,
                 init_lsv: float = 0.0,
                 freeze_r: bool = False,
                 freeze_h: bool = False,
                 reliability_mode: str = 'contextual',
                 h_min: float = 1e-3,
                 h_max: float = 1e3,
                 sigma_sq_min: float = 1e-3,
                 sigma_sq_max: float = 1e3):
        """
        reliability_mode:
            'contextual' (default) — r_m = sigmoid(Linear(concat(h_p, h_g, h_o)))
                Each r_m has access to ALL three modalities. Reflects
                contextual reliability ("trust path more when graph/omic
                agree"). This is the GRFN-Pathomic main spec.
            'intrinsic' — r_m = sigmoid(Linear_m(h_m))
                Each r_m computed from its own modality only. Reviewer-
                friendly variant for ablation A1'. Reflects intrinsic
                per-modality uncertainty unconditioned on the others.

        h_min, h_max, sigma_sq_min, sigma_sq_max:
            Bound the per-modality precision h_m and aleatoric variance
            sigma_m^2 after the exp() transform. Prevents the dominant
            training-instability failure mode (h_m -> infinity when one
            modality dominates). Default range [1e-3, 1e3] covers six
            orders of magnitude — well beyond what any sane modality
            would actually express.
        """
        super().__init__()
        self.dim = dim
        self.freeze_r = freeze_r
        self.freeze_h = freeze_h
        self.reliability_mode = reliability_mode
        self.h_min = h_min
        self.h_max = h_max
        self.sigma_sq_min = sigma_sq_min
        self.sigma_sq_max = sigma_sq_max

        # Per-modality heads: 32 -> 3  (mu, log_sigma_sq, log_h)
        self.head_p = nn.Linear(dim, 3)
        self.head_g = nn.Linear(dim, 3)
        self.head_o = nn.Linear(dim, 3)

        # Reliability head(s)
        # Default 'contextual' mode: a single Linear(96, 3) takes concat
        # of all three embeddings -> 3 logits. This gives r_m access to
        # all modalities (contextual trustworthiness).
        # Alternative 'intrinsic' mode (ablation A1'): three Linear(32, 1)
        # heads, each r_m computed from its own modality only.
        if reliability_mode == 'contextual':
            self.reliability = nn.Linear(3 * dim, 3)
            self.reliability_p = None
            self.reliability_g = None
            self.reliability_o = None
        elif reliability_mode == 'intrinsic':
            self.reliability = None
            self.reliability_p = nn.Linear(dim, 1)
            self.reliability_g = nn.Linear(dim, 1)
            self.reliability_o = nn.Linear(dim, 1)
        else:
            raise ValueError(f'reliability_mode must be contextual or intrinsic, got {reliability_mode}')

        # Initialize so trivial-limit check is well-defined.
        # We want:
        #   - at init, mu_m has variance ~ same scale as h_m output dim
        #   - log_h initialized to init_lh (default 0) so h = 1 at init
        #   - log_sigma_sq initialized to init_lsv (default 0) so sigma^2 = 1
        for head in (self.head_p, self.head_g, self.head_o):
            # Reasonable Xavier init for the weight; bias set to control
            # the initial value of (mu, log_sigma_sq, log_h).
            nn.init.xavier_normal_(head.weight)
            with torch.no_grad():
                head.bias.zero_()
                head.bias[1] = init_lsv   # log_sigma_sq init
                head.bias[2] = init_lh    # log_h init

        # Reliability bias init at 0 -> sigmoid(0) = 0.5  (neutral
        # discounting at init). We want this — not 1, not 0 — so that the
        # model has room to push r up or down based on data.
        if self.reliability is not None:
            nn.init.xavier_normal_(self.reliability.weight)
            nn.init.zeros_(self.reliability.bias)
        else:
            for h in (self.reliability_p, self.reliability_g, self.reliability_o):
                nn.init.xavier_normal_(h.weight)
                nn.init.zeros_(h.bias)

    def forward(self, h_p: torch.Tensor, h_g: torch.Tensor, h_o: torch.Tensor):
        """
        h_p, h_g, h_o : (B, dim) each.

        Returns
        -------
        mu_f       : (B,)
        sigma_f_sq : (B,)
        h_f        : (B,)
        r          : (B, 3)  — [r_path, r_graph, r_omic]
        """
        # Per-modality (mu, log_sigma_sq, log_h)  — each (B, 3)
        p_out = self.head_p(h_p)
        g_out = self.head_g(h_g)
        o_out = self.head_o(h_o)

        # Stack into (B, K=3, 3)
        # ordering: dim=1 is modality (path, graph, omic)
        mu  = torch.stack([p_out[:, 0], g_out[:, 0], o_out[:, 0]], dim=1)  # (B, 3)
        lsv = torch.stack([p_out[:, 1], g_out[:, 1], o_out[:, 1]], dim=1)  # (B, 3)
        lh  = torch.stack([p_out[:, 2], g_out[:, 2], o_out[:, 2]], dim=1)  # (B, 3)

        # Bounded exp() per reviewer feedback (2026-05-17):
        # Unbounded h or sigma^2 is the dominant training-instability
        # failure mode (one modality dominating drives its h to inf,
        # starving gradient to the others). Clamp the OUTPUT of exp to
        # [1e-3, 1e3] — six orders of magnitude is more than any sane
        # modality would express, but bounded prevents pathologies.
        sigma_sq = torch.exp(lsv).clamp(self.sigma_sq_min, self.sigma_sq_max)  # (B, 3)
        h        = torch.exp(lh).clamp(self.h_min, self.h_max)                 # (B, 3)

        # Reliability  r_m in (0, 1)
        if self.freeze_r:
            r = torch.ones_like(h)
        elif self.reliability_mode == 'contextual':
            cat = torch.cat([h_p, h_g, h_o], dim=1)                       # (B, 3*dim)
            r = torch.sigmoid(self.reliability(cat))                      # (B, 3)
        else:  # intrinsic
            r_p = torch.sigmoid(self.reliability_p(h_p))                  # (B, 1)
            r_g = torch.sigmoid(self.reliability_g(h_g))                  # (B, 1)
            r_o = torch.sigmoid(self.reliability_o(h_o))                  # (B, 1)
            r = torch.cat([r_p, r_g, r_o], dim=1)                         # (B, 3)

        if self.freeze_h:
            h = torch.ones_like(h)

        # GRFN combination rule (EsurvFusion arXiv 2412.01215v1 Eq. 4)
        # ------------------------------------------------------------
        # h_f       = sum_m  r_m h_m
        # mu_f      = (sum_m r_m h_m mu_m) / h_f
        # sigma_f^2 = (sum_m r_m^2 h_m^2 sigma_m^2) / h_f^2
        #
        # Use clamp_min on h_f (cleaner than + eps; doesn't distort large
        # h_f values). Reviewer feedback (2026-05-17).
        rh = r * h                                                         # (B, 3)
        h_f_raw = rh.sum(dim=1)                                            # (B,)
        h_f = h_f_raw.clamp_min(1e-8)                                      # protected denominator
        mu_f = (rh * mu).sum(dim=1) / h_f                                  # (B,)
        sigma_f_sq = ((r * h) ** 2 * sigma_sq).sum(dim=1) / (h_f ** 2)     # (B,)

        # Return h_f_raw (un-clamped) for diagnostic logging — collapse
        # detection wants to see the TRUE sum of discounted precisions.
        return mu_f, sigma_f_sq, h_f_raw, r


# ----------------------------------------------------------------------------
# Module-level invariants exposed for the smoke test
# ----------------------------------------------------------------------------

def trivial_limit_mu_f(mu_p: torch.Tensor, mu_g: torch.Tensor, mu_o: torch.Tensor):
    """
    Closed-form expected mu_f when r_m = 1 and h_m = 1 for all m.

    With r=1, h=1: rh = 1, h_f = 3, mu_f = (mu_p + mu_g + mu_o) / 3.
    """
    return (mu_p + mu_g + mu_o) / 3.0


def trivial_limit_sigma_f_sq(s_p: torch.Tensor, s_g: torch.Tensor, s_o: torch.Tensor):
    """
    Closed-form expected sigma_f^2 when r_m = 1 and h_m = 1, given
    per-modality sigma^2 values s_p, s_g, s_o.

    With r=1, h=1: numerator = sum(sigma_m^2), denominator = 9.
    """
    return (s_p + s_g + s_o) / 9.0
