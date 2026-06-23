"""Ship trajectory evaluation metrics (ADE/FDE in nautical miles)."""

import torch
import numpy as np


def denormalize_positions(pred_st, mean_t, std_t):
    """Denormalize predicted positions back to nautical miles.

    Args:
        pred_st: (B, n_sample, pred_len, 2) or (B, pred_len, 2) normalized positions
        mean_t: (2,) tensor on same device
        std_t: (2,) tensor on same device

    Returns:
        pred_nm: same shape, in nautical miles (local coordinates)
    """
    return pred_st * std_t + mean_t


def compute_ade_fde(pred_nm, gt_nm):
    """Compute ADE and FDE for multiple samples, taking the min across samples.

    Args:
        pred_nm: (B, n_sample, pred_len, 2) predictions in NM
        gt_nm:   (B, pred_len, 2) ground truth in NM

    Returns:
        ade: scalar, mean of min-ADE across batch
        fde: scalar, mean of min-FDE across batch
    """
    gt_expanded = gt_nm.unsqueeze(1)  # (B, 1, pred_len, 2)
    errors = torch.norm(pred_nm - gt_expanded, dim=-1)  # (B, n_sample, pred_len)

    ade_per_sample = errors.mean(dim=-1)  # (B, n_sample)
    fde_per_sample = errors[:, :, -1]     # (B, n_sample)

    min_ade = ade_per_sample.min(dim=1).values.mean()  # scalar
    min_fde = fde_per_sample.min(dim=1).values.mean()  # scalar

    return min_ade.item(), min_fde.item()


@torch.no_grad()
def evaluate(model, dataloader, norm_params, extract_fn, n_sample=20, device="cuda"):
    """Run evaluation on a dataloader.

    Args:
        model: ShipMGF model
        dataloader: val or test DataLoader
        norm_params: normalization parameters from training set
        extract_fn: function to extract target ship data from batch
        n_sample: number of trajectory samples
        device: torch device

    Returns:
        dict with 'ade' and 'fde' in nautical miles
    """
    was_training = model.training
    model.eval()

    mean_t = torch.tensor(norm_params["mean"][:2], dtype=torch.float32, device=device)
    std_t = torch.tensor(norm_params["std"][:2], dtype=torch.float32, device=device)

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    for batch in dataloader:
        data_dict = extract_fn(batch)
        data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in data_dict.items()}

        pred_st = model.predict(data_dict, n_sample=n_sample)  # (B, n_sample, pred_len, 2)

        pred_nm = denormalize_positions(pred_st, mean_t, std_t)
        gt_nm = denormalize_positions(data_dict["gt_st"], mean_t, std_t)

        ade, fde = compute_ade_fde(pred_nm, gt_nm)
        batch_size = pred_st.shape[0]
        total_ade += ade * batch_size
        total_fde += fde * batch_size
        total_samples += batch_size

    if was_training:
        model.train()

    if total_samples == 0:
        return {"ade": float("nan"), "fde": float("nan")}

    return {
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
    }
