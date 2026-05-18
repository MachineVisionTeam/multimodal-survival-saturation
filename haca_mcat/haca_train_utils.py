"""
Training utilities for HACA v1.5.

Mirrors utils/coattn_train_utils.py with three additions:

  1. AAS (Adaptive Anchoring Schedule) computes per-epoch lambda and passes
     it as a haca_lambda kwarg into model.forward().

  2. HACA_Surv returns six outputs (hazards, S, Y_hat, A, aux_hazards, aux_S)
     so the unpacking is updated accordingly.

  3. Optional auxiliary NLL-Surv loss term:
         loss = NLL_Surv(main) + aux_weight * NLL_Surv(aux)
     aux_weight=0.0 (default) makes the aux term identically zero.

Also logs r_j distribution statistics each epoch (mean / std / fraction near
0.5) for the Phase-1 collapse diagnostic.

Design: HACA_DESIGN_v1_5.md.
"""

import os
from collections import OrderedDict

import numpy as np
import torch

from utils.utils import *
from sksurv.metrics import concordance_index_censored


# ----------------------------------------------------------------------------
# AAS — Adaptive Anchoring Schedule
# ----------------------------------------------------------------------------
def aas_lambda(epoch: int, lambda_final: float, warmup_epochs: int) -> float:
    """
    Linear warmup from 0 to lambda_final over the first `warmup_epochs` epochs,
    then constant at lambda_final.

    aas_lambda(0,   1.0, 5) == 0.0
    aas_lambda(1,   1.0, 5) == 0.2
    aas_lambda(5,   1.0, 5) == 1.0
    aas_lambda(19,  1.0, 5) == 1.0
    aas_lambda(0,   1.0, 0) == 1.0   # warmup disabled
    """
    if warmup_epochs <= 0:
        return float(lambda_final)
    return float(lambda_final) * min(1.0, epoch / warmup_epochs)


# ----------------------------------------------------------------------------
# Diagnostic: r_j distribution summary (epoch-level)
# ----------------------------------------------------------------------------
def _summarise_r(r_values: np.ndarray) -> str:
    """One-line stats string: mean, std, min, max, fraction in [0.45, 0.55]."""
    if len(r_values) == 0:
        return 'r: <no samples>'
    near_half = float(np.mean((r_values > 0.45) & (r_values < 0.55)))
    return ('r_j: mean={:.3f} std={:.3f} min={:.3f} max={:.3f} '
            'frac_near_0.5={:.2%}'.format(
                r_values.mean(), r_values.std(),
                r_values.min(), r_values.max(),
                near_half))


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train_loop_survival_haca(epoch, model, loader, optimizer, n_classes,
                              writer=None, loss_fn=None, reg_fn=None,
                              lambda_reg=0., gc=16,
                              haca_lambda_final=1.0, haca_warmup_epochs=5,
                              haca_aux_weight=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    cur_lambda = aas_lambda(epoch, haca_lambda_final, haca_warmup_epochs)
    print('  HACA epoch {} lambda = {:.4f}  (final={}, warmup={}, aux_weight={})'
          .format(epoch, cur_lambda, haca_lambda_final, haca_warmup_epochs, haca_aux_weight))

    # Sample r_j histograms across batches for the collapse diagnostic.
    r_sample_chunks = []

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3,
                    data_omic4, data_omic5, data_omic6,
                    label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        hazards, S, Y_hat, A, aux_hazards, aux_S = model(
            x_path=data_WSI,
            x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3,
            x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6,
            haca_lambda=cur_lambda,
        )

        loss_main = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        if haca_aux_weight > 0:
            loss_aux = loss_fn(hazards=aux_hazards, S=aux_S, Y=label, c=c)
            loss = loss_main + haca_aux_weight * loss_aux
        else:
            loss = loss_main
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        # Collect a small sample of r_j values for diagnostic histogram.
        # Cap at ~50 patches/batch to keep memory bounded.
        r_batch = A['haca_r'].detach().cpu().numpy()
        if len(r_batch) > 50:
            idx = np.random.choice(len(r_batch), 50, replace=False)
            r_batch = r_batch[idx]
        r_sample_chunks.append(r_batch)

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size:'
                  .format(batch_idx, loss_value + loss_reg, label.item(),
                          float(event_time), float(risk)))

        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times,
        all_risk_scores, tied_tol=1e-08)[0]

    r_all = np.concatenate(r_sample_chunks) if r_sample_chunks else np.array([])
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, '
          'train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))
    print('  ' + _summarise_r(r_all))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('haca/lambda', cur_lambda, epoch)
        if len(r_all):
            writer.add_scalar('haca/r_mean', float(r_all.mean()), epoch)
            writer.add_scalar('haca/r_std', float(r_all.std()), epoch)
            writer.add_histogram('haca/r_hist', r_all, epoch)


# ----------------------------------------------------------------------------
# Validation loop
# ----------------------------------------------------------------------------
def validate_survival_haca(cur, epoch, model, loader, n_classes,
                            early_stopping=None, monitor_cindex=None,
                            writer=None, loss_fn=None, reg_fn=None,
                            lambda_reg=0., results_dir=None,
                            haca_lambda_final=1.0, haca_warmup_epochs=5,
                            haca_aux_weight=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    cur_lambda = aas_lambda(epoch, haca_lambda_final, haca_warmup_epochs)

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3,
                    data_omic4, data_omic5, data_omic6,
                    label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            hazards, S, Y_hat, A, aux_hazards, aux_S = model(
                x_path=data_WSI,
                x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3,
                x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6,
                haca_lambda=cur_lambda,
            )

        loss_main = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        if haca_aux_weight > 0:
            loss_aux = loss_fn(hazards=aux_hazards, S=aux_S, Y=label, c=c, alpha=0)
            loss = loss_main + haca_aux_weight * loss_aux
        else:
            loss = loss_main
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times,
        all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model,
                       ckpt_name=os.path.join(
                           results_dir,
                           's_{}_minloss_checkpoint.pt'.format(cur)))
        if early_stopping.early_stop:
            print('Early stopping')
            return True

    return False


# ----------------------------------------------------------------------------
# Final-summary (post-training) loop
# ----------------------------------------------------------------------------
def summary_survival_haca(model, loader, n_classes,
                          haca_lambda_final=1.0, haca_warmup_epochs=5):
    """
    Per-slide risk extraction using the FINAL lambda (post-warmup).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    # At inference, AAS has long completed → use the final lambda.
    cur_lambda = float(haca_lambda_final)

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3,
                    data_omic4, data_omic5, data_omic6,
                    label, event_time, c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, A, aux_hazards, aux_S = model(
                x_path=data_WSI,
                x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3,
                x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6,
                haca_lambda=cur_lambda,
            )

        risk = (-torch.sum(survival, dim=1)).item()
        event_time = event_time.item()
        c = c.item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {
            'slide_id': np.array(slide_id),
            'risk': risk,
            'disc_label': label.item(),
            'survival': event_time,
            'censorship': c,
        }})

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times,
        all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index
