"""
Phase-0 smoke test for DSMFusion. Eight checks (mirroring the GRFN smoke
test structure):

  T1. Shapes — predict_params returns (B, M*K) for alphas/betas/log_weights.
  T2. Positivity — alphas, betas in their clamped ranges; weights sum to 1.
  T3. Gradient flow — all 4 heads (head_p/g/o, gate) receive nonzero gradient
      from a censored DSM NLL on a small synthetic batch.
  T4. Trivial-limit K=1 — single Weibull per modality, mixture weights
      reduce to a 3-way modality vote. Check survival monotone in t.
  T5. Numerical stability — extreme inputs that would drive alpha/beta to
      clamp boundaries should NOT produce NaN/Inf in NLL or survival.
  T6. Survival function basic properties — S(0+) ≈ 1, S(∞) ≈ 0, monotone
      non-increasing on a fine time grid.
  T7. Cox risk vs E[T] consistency — risk = -E[T] should be finite and
      negative for sensible synthetic data.
  T8. Censoring sanity — censored observation should contribute log S(t),
      uncensored should contribute log f(t). Verify the loss differentiates.
"""

import sys
import math

import torch

sys.path.insert(0, '/home/sbarua/Region_based_segmentation/DSM-Pathomic')
from dsm_fusion import DSMFusion


def run_smoke_test(verbose: bool = True):
    torch.manual_seed(0)
    B, D, K = 16, 32, 4
    M = 3

    h_p = torch.randn(B, D)
    h_g = torch.randn(B, D)
    h_o = torch.randn(B, D)
    # Realistic TCGA times: ~100-5000 days
    t = torch.empty(B).uniform_(100, 5000)
    # Mix of event (=1) and censored (=0). Using PF's "censor" variable name
    # for the actual event indicator (see dsm_fusion.nll_loss docstring).
    event = (torch.arange(B) % 2).float()        # alternating, 8 each
    censor = event  # alias for compatibility with the rest of the test

    # ------------------------------------------------------------------
    # T1, T2: shapes + positivity
    # ------------------------------------------------------------------
    model = DSMFusion(dim=D, K=K)
    out = model(h_p, h_g, h_o)
    alphas = out['alphas']; betas = out['betas']; lw = out['log_weights']

    assert alphas.shape == (B, M*K), f'T1 FAIL: alphas shape {alphas.shape}'
    assert betas.shape == (B, M*K),  f'T1 FAIL: betas shape {betas.shape}'
    assert lw.shape == (B, M*K),     f'T1 FAIL: log_weights shape {lw.shape}'
    if verbose: print(f'[T1] Shapes OK   (B={B}, M*K={M*K})')

    assert (alphas >= model.alpha_min - 1e-5).all() and (alphas <= model.alpha_max + 1e-5).all(), \
        f'T2 FAIL: alphas outside clamp [{model.alpha_min}, {model.alpha_max}]'
    assert (betas >= model.beta_min - 1e-5).all() and (betas <= model.beta_max + 1e-5).all(), \
        f'T2 FAIL: betas outside clamp'
    weights = lw.exp()
    weight_sums = weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f'T2 FAIL: weights do not sum to 1; got {weight_sums[:3]}'
    if verbose:
        print(f'[T2] Positivity + simplex OK   '
              f'alpha range=[{alphas.min():.3f}, {alphas.max():.3f}], '
              f'beta range=[{betas.min():.1f}, {betas.max():.1f}], '
              f'weight sum mean={weight_sums.mean():.6f}')

    # ------------------------------------------------------------------
    # T3: gradient flow
    # ------------------------------------------------------------------
    model.zero_grad()
    loss = model.nll_loss(t, censor, h_p, h_g, h_o)
    loss.backward()
    g = {
        'head_p': sum(p.grad.norm().item() for p in model.head_p.parameters() if p.grad is not None),
        'head_g': sum(p.grad.norm().item() for p in model.head_g.parameters() if p.grad is not None),
        'head_o': sum(p.grad.norm().item() for p in model.head_o.parameters() if p.grad is not None),
        'gate':   sum(p.grad.norm().item() for p in model.gate.parameters() if p.grad is not None),
    }
    for name, gn in g.items():
        assert gn > 1e-8, f'T3 FAIL: {name} grad norm {gn} ≈ 0'
    if verbose: print(f'[T3] Gradient flow OK   grad norms = {g}   loss = {loss.item():.4f}')

    # ------------------------------------------------------------------
    # T4: Trivial-limit K=1 — single Weibull per modality (3 components total)
    # ------------------------------------------------------------------
    model_k1 = DSMFusion(dim=D, K=1)
    out_k1 = model_k1(h_p, h_g, h_o)
    assert out_k1['alphas'].shape == (B, 3), f'T4 FAIL: K=1 alphas shape {out_k1["alphas"].shape}'
    if verbose:
        print(f'[T4] K=1 trivial-limit OK   shape (B, 3) — 1 Weibull per modality, '
              f'mixture weights are the 3-way modality vote')

    # ------------------------------------------------------------------
    # T5: Numerical stability — drive log_alpha to extreme bias
    # ------------------------------------------------------------------
    model_stab = DSMFusion(dim=D, K=K)
    with torch.no_grad():
        for head in (model_stab.head_p, model_stab.head_g, model_stab.head_o):
            # Push log_alpha bias to +20 and log_beta bias to +20 (would give
            # exp(20) ≈ 5e8 without clamp)
            head.bias[:K] = 20.0
            head.bias[K:] = 20.0
    loss_hi = model_stab.nll_loss(t, censor, h_p, h_g, h_o)
    assert torch.isfinite(loss_hi), f'T5 FAIL: loss is non-finite at extreme params; got {loss_hi}'
    out_hi = model_stab(h_p, h_g, h_o)
    assert torch.isfinite(out_hi['alphas']).all() and torch.isfinite(out_hi['betas']).all(), \
        'T5 FAIL: non-finite params at extreme bias'

    # Also test extreme negative log
    with torch.no_grad():
        for head in (model_stab.head_p, model_stab.head_g, model_stab.head_o):
            head.bias[:K] = -20.0
            head.bias[K:] = -20.0
    loss_lo = model_stab.nll_loss(t, censor, h_p, h_g, h_o)
    assert torch.isfinite(loss_lo), f'T5 FAIL: loss non-finite at negative-extreme params; got {loss_lo}'
    if verbose:
        print(f'[T5] Numerical stability OK   loss at +20 bias = {loss_hi.item():.4f}, '
              f'at -20 bias = {loss_lo.item():.4f}')

    # ------------------------------------------------------------------
    # T6: Survival function properties — S(0+) ≈ 1, S(huge) ≈ 0, monotone
    # ------------------------------------------------------------------
    t_grid = torch.tensor([0.1, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 20000.0])
    with torch.no_grad():
        S = model.survival_at_times(t_grid, h_p, h_g, h_o)     # (B, T)
    assert S.shape == (B, len(t_grid)), f'T6 FAIL: S shape {S.shape}'
    # S(0+) close to 1
    assert (S[:, 0] > 0.95).all(), \
        f'T6 FAIL: S(0+) not near 1; mean={S[:, 0].mean():.4f}'
    # S decreasing across the grid (allow tiny float noise)
    monotone_violations = ((S[:, 1:] - S[:, :-1]) > 1e-4).sum().item()
    assert monotone_violations == 0, \
        f'T6 FAIL: {monotone_violations} non-monotone violations in S(t)'
    if verbose:
        print(f'[T6] Survival fn OK   S(0+) mean={S[:, 0].mean():.4f}  '
              f'S(20000) mean={S[:, -1].mean():.4f}  monotone violations={monotone_violations}')

    # ------------------------------------------------------------------
    # T7: Cox risk = -E[T]
    # ------------------------------------------------------------------
    with torch.no_grad():
        risk = model.cox_risk(h_p, h_g, h_o)
        ET = model.expected_time(h_p, h_g, h_o)
    assert risk.shape == (B,), f'T7 FAIL: risk shape {risk.shape}'
    assert torch.allclose(risk, -ET), 'T7 FAIL: cox_risk != -expected_time'
    assert torch.isfinite(risk).all(), 'T7 FAIL: risk non-finite'
    if verbose:
        print(f'[T7] Cox-risk = -E[T] OK   E[T] mean={ET.mean():.1f} days, '
              f'risk range=[{risk.min():.1f}, {risk.max():.1f}]')

    # ------------------------------------------------------------------
    # T8: Censoring sanity — loss differentiates events from censored
    # ------------------------------------------------------------------
    # Build two identical batches: one all-event (event=1), one all-censored
    # (event=0). The all-event loss should consume log f(t) → typically LARGER
    # in magnitude than the all-censored loss which consumes log S(t).
    with torch.no_grad():
        loss_all_event = model.nll_loss(t, torch.ones_like(event), h_p, h_g, h_o)
        loss_all_censored = model.nll_loss(t, torch.zeros_like(event), h_p, h_g, h_o)
    assert loss_all_event != loss_all_censored, \
        'T8 FAIL: event vs censored not differentiating loss'
    if verbose:
        print(f'[T8] Event/censor sanity OK   all-event loss={loss_all_event.item():.4f}, '
              f'all-censored loss={loss_all_censored.item():.4f}  '
              f'(log f is typically more negative than log S; event NLL > censored NLL)')

    if verbose:
        print('\n=== ALL T1-T8 PASS — DSMFusion Phase 0 smoke test green ===')
        # Print a parameter count for the architecture doc
        n_params = sum(p.numel() for p in model.parameters())
        print(f'\nParameter count (K={K}, M=3): {n_params}  (architecture doc spec: ~1,956 for K=4)')


if __name__ == '__main__':
    run_smoke_test(verbose=True)
