"""
DSM-Pathomic fusion + likelihood module.

Implements the modality-specific Deep Survival Machines configuration
described in DSM_Pathomic_ARCHITECTURE.txt §2. Pure PyTorch, no dependency
on auton-survival (whose 2021-era version constraints broke our PF env).

Reference for the DSM math:
  Nagpal, Li, Dubrawski (2021). "Deep Survival Machines: Fully Parametric
  Survival Regression and Representation Learning for Censored Data with
  Competing Risks." IEEE JBHI 25(8):3163-3175. arXiv:2003.01176.

Key design choice (the only piece marginally novel in this work):
  Per-modality component parameters. For K Weibull components per
  modality and M=3 modalities (path, graph, omic), the model emits 3K
  total components. The (alpha, beta) for components k ∈ {1..K} are
  predicted from h_path only; k ∈ {K+1..2K} from h_graph only; etc.
  Mixture weights w_{m,k}(x) are predicted from the CONCATENATED
  embedding (contextual gating). This preserves per-modality attribution
  while still letting the gate see all three modalities.

Author: 2026-05-18.
"""

import math

import torch
import torch.nn as nn


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp_min(x, eps))


class DSMFusion(nn.Module):
    """
    Modality-specific DSM head for Pathomic Fusion's three locked 32-d
    encoders.

    Args:
        dim          : per-modality embedding dim (32 for PF).
        K            : Weibull components per modality (default 4).
        alpha_min, alpha_max : clamp range for Weibull shape.
                       0.3 .. 10 covers decreasing/constant/increasing hazard
                       families without numerical pathology.
        beta_min, beta_max   : clamp range for Weibull scale (in TIME units —
                       days, for TCGA). 1 day .. 10000 days = 27 years.
        log_initial_beta     : initialization of log(beta), set near the
                       median observed survival time so the mixture starts
                       in a sane region (default log(1000) ≈ 6.9).
        gate_temp    : softmax temperature on the mixture-weight logits.
    """

    def __init__(self,
                 dim: int = 32,
                 K: int = 4,
                 alpha_min: float = 0.3,
                 alpha_max: float = 10.0,
                 beta_min: float = 1.0,
                 beta_max: float = 10000.0,
                 log_initial_alpha: float = 0.0,            # exp(0)=1 (exponential dist)
                 log_initial_beta: float = math.log(1000.0), # median glioma survival ~1000d
                 gate_temp: float = 1.0):
        super().__init__()
        self.dim = dim
        self.K = K
        self.M = 3                                  # path, graph, omic
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gate_temp = gate_temp

        # Initial-offset constants added INSIDE predict_params so they survive
        # any external weight-init scheme (e.g., PF's init_net which zeroes
        # all Linear biases). With these constants, the effective log_alpha
        # and log_beta start at the intended init values even when the
        # underlying Linear bias is 0.
        self._log_alpha_init = log_initial_alpha
        self._log_beta_init = log_initial_beta

        # Per-modality (alpha, beta) heads — 2K outputs each (K log_alpha + K log_beta)
        self.head_p = nn.Linear(dim, 2 * K)
        self.head_g = nn.Linear(dim, 2 * K)
        self.head_o = nn.Linear(dim, 2 * K)

        # Mixture-gate head — 3K outputs from concatenated 3*dim embedding
        self.gate = nn.Linear(self.M * dim, self.M * K)

        # Weight init: small xavier_normal so head outputs are near 0 at init.
        # We DON'T set biases here — PF's external init_net will zero them
        # anyway. The _log_*_init constants in predict_params provide the
        # intended initial location of alpha/beta.
        for head in (self.head_p, self.head_g, self.head_o):
            nn.init.xavier_normal_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
        nn.init.xavier_normal_(self.gate.weight, gain=0.1)
        nn.init.zeros_(self.gate.bias)               # uniform mixture at init

    # ------------------------------------------------------------------
    # forward / parameter prediction
    # ------------------------------------------------------------------

    def predict_params(self, h_p: torch.Tensor, h_g: torch.Tensor, h_o: torch.Tensor):
        """
        Predict the (alpha, beta, log_weights) for each component on a batch.

        Returns
        -------
        alphas       : (B, M*K)  Weibull shape per component (clamped)
        betas        : (B, M*K)  Weibull scale per component (clamped)
        log_weights  : (B, M*K)  log-softmax mixture weights (numerically stable)
        """
        B = h_p.shape[0]

        # Per-modality heads -> (log_alpha, log_beta) split.
        # Add the _log_*_init constants so the effective log_alpha and
        # log_beta start at the intended init values regardless of how the
        # head's Linear bias was initialized (PF's init_net zeroes biases).
        def head_to_params(head, h):
            out = head(h)                            # (B, 2K)
            la = out[:, :self.K] + self._log_alpha_init
            lb = out[:, self.K:] + self._log_beta_init
            a = torch.exp(la).clamp(self.alpha_min, self.alpha_max)
            b = torch.exp(lb).clamp(self.beta_min, self.beta_max)
            return a, b

        a_p, b_p = head_to_params(self.head_p, h_p)
        a_g, b_g = head_to_params(self.head_g, h_g)
        a_o, b_o = head_to_params(self.head_o, h_o)

        # Concatenate components in [path, graph, omic] order: (B, 3K)
        alphas = torch.cat([a_p, a_g, a_o], dim=1)
        betas = torch.cat([b_p, b_g, b_o], dim=1)

        # Mixture gate from concatenated embeddings
        logits = self.gate(torch.cat([h_p, h_g, h_o], dim=1)) / self.gate_temp
        log_weights = torch.log_softmax(logits, dim=1)   # (B, 3K)

        return alphas, betas, log_weights

    def forward(self, h_p: torch.Tensor, h_g: torch.Tensor, h_o: torch.Tensor):
        """
        Convenience forward returning a single dict with everything.
        """
        a, b, lw = self.predict_params(h_p, h_g, h_o)
        return {'alphas': a, 'betas': b, 'log_weights': lw}

    # ------------------------------------------------------------------
    # Per-component log-survival / log-density
    # ------------------------------------------------------------------

    @staticmethod
    def _log_S_weibull(t: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """
        log S(t) = - (t / beta)^alpha
        Shapes: t (B,) or (B,1); alpha, beta (B, K_total). Returns (B, K_total).
        """
        t = t.clamp_min(1e-6).unsqueeze(-1) if t.dim() == 1 else t.clamp_min(1e-6)
        # (t/beta) > 0, alpha > 0 -> well-defined power
        return -((t / beta) ** alpha)

    @staticmethod
    def _log_f_weibull(t: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """
        log f(t) = log(alpha) - log(beta) + (alpha - 1) log(t/beta) - (t/beta)^alpha
        Shapes as in _log_S_weibull.
        """
        t = t.clamp_min(1e-6).unsqueeze(-1) if t.dim() == 1 else t.clamp_min(1e-6)
        log_t_over_beta = torch.log(t) - torch.log(beta)
        return (torch.log(alpha) - torch.log(beta) +
                (alpha - 1.0) * log_t_over_beta -
                (t / beta) ** alpha)

    # ------------------------------------------------------------------
    # Mixture log-survival / log-density (numerically stable via logsumexp)
    # ------------------------------------------------------------------

    def log_survival(self, t: torch.Tensor, alphas: torch.Tensor,
                     betas: torch.Tensor, log_weights: torch.Tensor):
        """
        log S(t | x) = log [ sum_k w_k * S_k(t) ]
                     = logsumexp_k [ log_w_k + log_S_k(t) ]
        """
        log_S_k = self._log_S_weibull(t, alphas, betas)         # (B, M*K)
        return torch.logsumexp(log_weights + log_S_k, dim=1)    # (B,)

    def log_density(self, t: torch.Tensor, alphas: torch.Tensor,
                    betas: torch.Tensor, log_weights: torch.Tensor):
        """
        log f(t | x) = logsumexp_k [ log_w_k + log_f_k(t) ]
        """
        log_f_k = self._log_f_weibull(t, alphas, betas)         # (B, M*K)
        return torch.logsumexp(log_weights + log_f_k, dim=1)    # (B,)

    # ------------------------------------------------------------------
    # DSM negative log-likelihood loss
    # ------------------------------------------------------------------

    def nll_loss(self, t: torch.Tensor, event: torch.Tensor,
                 h_p: torch.Tensor, h_g: torch.Tensor, h_o: torch.Tensor):
        """
        DSM negative log-likelihood on one batch.

        t        : (B,)  survival times (positive)
        event    : (B,)  event indicator. 1 = event observed (uncensored),
                          0 = censored.
                          NOTE: Pathomic Fusion's local variable name 'censor'
                          actually carries this event-indicator convention
                          (verified against utils.CoxLoss line 358 which multi-
                          plies by `censor` — would zero-out censored cases if
                          censor=1 meant "censored"). We use the explicit
                          `event` parameter name here to avoid the confusion.
                          The PF call-site passes its 'censor' variable as
                          this `event` argument.
        h_p, h_g, h_o : (B, dim)  modality embeddings

        Returns
        -------
        loss : scalar mean negative log-likelihood
        """
        alphas, betas, log_weights = self.predict_params(h_p, h_g, h_o)

        log_S = self.log_survival(t, alphas, betas, log_weights)   # (B,)
        log_f = self.log_density(t, alphas, betas, log_weights)    # (B,)

        # Standard DSM NLL:
        #   event=1 (event observed)  => contributes log f(t)  (density at event time)
        #   event=0 (censored)        => contributes log S(t)  (survival to censor time)
        # NLL = -mean[ event * log_f + (1 - event) * log_S ]
        nll = -(event * log_f + (1.0 - event) * log_S).mean()
        return nll

    # ------------------------------------------------------------------
    # c-Index risk and S(t|x) for evaluation
    # ------------------------------------------------------------------

    def expected_time(self, h_p: torch.Tensor, h_g: torch.Tensor, h_o: torch.Tensor):
        """
        E[T | x] = sum_k w_k(x) * beta_k * Gamma(1 + 1/alpha_k)
        Returns (B,) expected survival times.
        """
        alphas, betas, log_weights = self.predict_params(h_p, h_g, h_o)
        weights = log_weights.exp()
        # torch.lgamma not directly broadcastable -> compute per-component
        per_comp_mean = betas * torch.exp(torch.lgamma(1.0 + 1.0 / alphas))   # (B, M*K)
        return (weights * per_comp_mean).sum(dim=1)

    def cox_risk(self, h_p: torch.Tensor, h_g: torch.Tensor, h_o: torch.Tensor):
        """
        Cox-comparable scalar risk: risk = -E[T]. Higher risk = shorter
        expected time, matching PF's convention for c-Index ranking.
        """
        return -self.expected_time(h_p, h_g, h_o)

    def survival_at_times(self, t_grid: torch.Tensor,
                          h_p: torch.Tensor, h_g: torch.Tensor, h_o: torch.Tensor):
        """
        Predicted S(t | x) for each (patient, t) pair.

        t_grid       : (T,) array of evaluation times
        h_p/g/o      : (B, dim) modality embeddings

        Returns
        -------
        S            : (B, T) survival probabilities. Computed natively
                       from the Weibull mixture — NO BRESLOW estimator needed.
                       This is the key calibration-vs-Cox advantage.
        """
        alphas, betas, log_weights = self.predict_params(h_p, h_g, h_o)
        T = t_grid.shape[0]
        B = h_p.shape[0]
        K_total = alphas.shape[1]

        # Broadcasting: t_grid (T,) → (1, T, 1); alphas (B, K_total) → (B, 1, K_total)
        t_b = t_grid.view(1, T, 1).clamp_min(1e-6)
        a_b = alphas.unsqueeze(1)
        b_b = betas.unsqueeze(1)
        log_S_k = -((t_b / b_b) ** a_b)                          # (B, T, K)

        log_w = log_weights.unsqueeze(1)                          # (B, 1, K)
        log_S = torch.logsumexp(log_w + log_S_k, dim=2)           # (B, T)
        return log_S.exp()
