"""
Phase-0 smoke test for GRFNFusion. Six checks:

  T1. Shapes: forward output dims match spec.
  T2. Positivity: sigma_f^2 > 0, h_f > 0, r in (0, 1).
  T3. Gradient flow: all four heads receive non-zero gradient from a
      dummy Cox-style loss.
  T4. Trivial-limit reduction: freeze_r=True, freeze_h=True
      ->  mu_f == mean(mu_p, mu_g, mu_o)  exactly (within float tolerance).
  T5. Trivial-limit sigma: same setting
      ->  sigma_f^2 == mean(sigma^2) / 3.
  T6. Independent-modality lesion: zeroing the omic embedding
      changes mu_f if r_omic > 0 (sanity that omic actually contributes).

If T1-T6 all pass, the GRFN combination rule is implemented correctly and
Phase 1 (wire into PF training) can begin.
"""

import math
import sys

import torch

# Local import — file lives next to this test.
sys.path.insert(0, '/home/sbarua/Region_based_segmentation/GRFN-Pathomic')
from grfn_fusion import GRFNFusion, trivial_limit_mu_f, trivial_limit_sigma_f_sq


def run_smoke_test(verbose: bool = True):
    torch.manual_seed(0)
    B, D = 16, 32
    h_p = torch.randn(B, D, requires_grad=False)
    h_g = torch.randn(B, D, requires_grad=False)
    h_o = torch.randn(B, D, requires_grad=False)

    # ------------------------------------------------------------------
    # T1, T2, T3  — default GRFN, full forward + backward
    # ------------------------------------------------------------------
    model = GRFNFusion(dim=D, freeze_r=False, freeze_h=False)
    mu_f, sigma_f_sq, h_f, r = model(h_p, h_g, h_o)

    assert mu_f.shape == (B,),           f'T1 FAIL: mu_f shape {mu_f.shape}'
    assert sigma_f_sq.shape == (B,),     f'T1 FAIL: sigma_f_sq shape {sigma_f_sq.shape}'
    assert h_f.shape == (B,),            f'T1 FAIL: h_f shape {h_f.shape}'
    assert r.shape == (B, 3),            f'T1 FAIL: r shape {r.shape}'
    if verbose: print('[T1] Output shapes OK')

    assert (sigma_f_sq > 0).all().item(), 'T2 FAIL: sigma_f^2 not all positive'
    assert (h_f > 0).all().item(),        'T2 FAIL: h_f not all positive'
    assert (r > 0).all().item() and (r < 1).all().item(), 'T2 FAIL: r not in (0,1)'
    if verbose:
        print(f'[T2] Positivity OK   sigma_f^2 mean={sigma_f_sq.mean():.3f}, '
              f'h_f mean={h_f.mean():.3f}, r mean per-mod={r.mean(dim=0).tolist()}')

    # Cox-style scalar loss: maximize negative mean risk
    loss = -mu_f.mean()
    loss.backward()
    g_norms = {
        'head_p':       sum(p.grad.norm().item() for p in model.head_p.parameters()       if p.grad is not None),
        'head_g':       sum(p.grad.norm().item() for p in model.head_g.parameters()       if p.grad is not None),
        'head_o':       sum(p.grad.norm().item() for p in model.head_o.parameters()       if p.grad is not None),
        'reliability':  sum(p.grad.norm().item() for p in model.reliability.parameters()  if p.grad is not None),
    }
    for name, g in g_norms.items():
        assert g > 1e-8, f'T3 FAIL: {name} grad norm {g} ~= 0'
    if verbose: print(f'[T3] Gradient flow OK   grad norms = {g_norms}')

    # ------------------------------------------------------------------
    # T4, T5  — trivial-limit reduction
    # ------------------------------------------------------------------
    # Build a frozen GRFN where r=1 and h=1 always. mu_f should equal
    # the simple mean of the three modalities' mu outputs.
    triv = GRFNFusion(dim=D, freeze_r=True, freeze_h=True)
    # Sync heads' weights to the trained model so we can compare like-for-like
    triv.head_p.load_state_dict(model.head_p.state_dict())
    triv.head_g.load_state_dict(model.head_g.state_dict())
    triv.head_o.load_state_dict(model.head_o.state_dict())
    with torch.no_grad():
        mu_f_t, sigma_f_sq_t, h_f_t, r_t = triv(h_p, h_g, h_o)
        # Expected mu_f from trivial limit
        mu_p_only = triv.head_p(h_p)[:, 0]
        mu_g_only = triv.head_g(h_g)[:, 0]
        mu_o_only = triv.head_o(h_o)[:, 0]
        expected_mu_f = trivial_limit_mu_f(mu_p_only, mu_g_only, mu_o_only)

        sigma_p = torch.exp(triv.head_p(h_p)[:, 1])
        sigma_g = torch.exp(triv.head_g(h_g)[:, 1])
        sigma_o = torch.exp(triv.head_o(h_o)[:, 1])
        expected_sigma_f_sq = trivial_limit_sigma_f_sq(sigma_p, sigma_g, sigma_o)

    diff_mu = (mu_f_t - expected_mu_f).abs().max().item()
    diff_sigma = (sigma_f_sq_t - expected_sigma_f_sq).abs().max().item()
    assert diff_mu < 1e-5, f'T4 FAIL: max(|mu_f - mean(mu_m)|) = {diff_mu}'
    if verbose: print(f'[T4] Trivial-limit mu_f matches mean fusion  (max diff = {diff_mu:.2e})')
    assert diff_sigma < 1e-5, f'T5 FAIL: max(|sigma_f^2 - mean(s^2)/3|) = {diff_sigma}'
    if verbose: print(f'[T5] Trivial-limit sigma_f^2 matches mean/3   (max diff = {diff_sigma:.2e})')

    # Also check h_f = 3 and r = 1 under the freeze
    assert torch.allclose(h_f_t, torch.full_like(h_f_t, 3.0)), \
        f'T5b FAIL: h_f should be 3.0 under freeze_r/h, got mean={h_f_t.mean():.3f}'
    assert torch.allclose(r_t, torch.ones_like(r_t)), 'T5b FAIL: r should be 1 under freeze_r'
    if verbose: print('[T5b] Frozen-limit r=1, h_f=3  OK')

    # ------------------------------------------------------------------
    # T6  — modality lesion sanity: zeroing omic should shift mu_f
    # ------------------------------------------------------------------
    with torch.no_grad():
        mu_f_full, _, _, r_full = model(h_p, h_g, h_o)
        mu_f_no_o, _, _, r_no_o = model(h_p, h_g, torch.zeros_like(h_o))
    shift = (mu_f_full - mu_f_no_o).abs().mean().item()
    assert shift > 1e-4, f'T6 FAIL: zeroing omic produced no shift in mu_f ({shift:.2e})'
    if verbose:
        print(f'[T6] Modality-lesion sanity OK   mean |mu_f - mu_f(no omic)| = {shift:.4f}')
        print(f'      r_full mean per-mod   = {r_full.mean(dim=0).tolist()}')
        print(f'      r_no_omic mean per-mod = {r_no_o.mean(dim=0).tolist()}')

    # ------------------------------------------------------------------
    # T7  — numerical-stability: extreme h_m values
    # Reviewer (2026-05-17) flagged that unbounded h can blow up the
    # combination rule. We now clamp h to [1e-3, 1e3]. Verify that:
    #   (a) when all log_h are pushed to +20 (exp(20)≈5e8, clamped to 1e3),
    #       mu_f is still finite and tracks the highest-h modality's mu
    #   (b) when all log_h are pushed to -20 (exp(-20)≈2e-9, clamped to 1e-3),
    #       mu_f is still finite (no div-by-zero)
    # ------------------------------------------------------------------
    stab = GRFNFusion(dim=D, freeze_r=False, freeze_h=False)
    with torch.no_grad():
        # Hand-craft a payload that drives log_h to extremes via bias.
        # (We're testing the combination rule's robustness, not the heads.)
        # Force log_h = +20 (huge precision) for all heads.
        for head in (stab.head_p, stab.head_g, stab.head_o):
            head.bias[2] = 20.0
        mu_f_hi, _, h_f_hi, _ = stab(h_p, h_g, h_o)
        assert torch.isfinite(mu_f_hi).all(), 'T7a FAIL: mu_f non-finite when h saturates high'
        # h_f should be clamped near 3 × h_max
        assert (h_f_hi <= 3 * 1e3 + 1).all(), f'T7a FAIL: h_f overflowed clamp, max={h_f_hi.max():.2e}'

        # Force log_h = -20 (vanishing precision) for all heads.
        for head in (stab.head_p, stab.head_g, stab.head_o):
            head.bias[2] = -20.0
        mu_f_lo, _, h_f_lo, _ = stab(h_p, h_g, h_o)
        assert torch.isfinite(mu_f_lo).all(), 'T7b FAIL: mu_f non-finite when h vanishes'
    if verbose:
        print(f'[T7] Numerical-stability OK  '
              f'(h_high: mu_f∈[{mu_f_hi.min():.2f},{mu_f_hi.max():.2f}], '
              f'h_f≤{h_f_hi.max():.0f}; '
              f'h_low: mu_f∈[{mu_f_lo.min():.2f},{mu_f_lo.max():.2f}])')

    # ------------------------------------------------------------------
    # T8  — Intrinsic reliability variant (A1' ablation)
    # Reviewer flagged contextual r_m as worth disambiguating from
    # "intrinsic per-modality reliability." Verify the intrinsic variant
    # builds and forwards correctly, and that r_m depends ONLY on its
    # own modality embedding (lesion test).
    # ------------------------------------------------------------------
    intr = GRFNFusion(dim=D, reliability_mode='intrinsic')
    mu_f_i, sig_i, h_f_i, r_i = intr(h_p, h_g, h_o)
    assert r_i.shape == (B, 3), f'T8 FAIL: intrinsic r shape {r_i.shape}'
    # Lesion: changing h_o should NOT change r_p in intrinsic mode
    with torch.no_grad():
        _, _, _, r_full = intr(h_p, h_g, h_o)
        _, _, _, r_zero_o = intr(h_p, h_g, torch.zeros_like(h_o))
        r_path_diff = (r_full[:, 0] - r_zero_o[:, 0]).abs().max().item()
    assert r_path_diff < 1e-6, \
        f'T8 FAIL: intrinsic r_path changed when h_o lesioned ({r_path_diff:.2e}); should be 0'
    if verbose:
        print(f'[T8] Intrinsic reliability mode OK  '
              f'(r_path unchanged when h_o lesioned, diff={r_path_diff:.2e})')

    if verbose:
        print('\n=== ALL T1-T8 PASS — Phase 0 smoke test green ===')


if __name__ == '__main__':
    run_smoke_test(verbose=True)
