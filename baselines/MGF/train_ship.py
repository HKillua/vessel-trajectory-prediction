"""MGF Ship Trajectory Prediction — training script.

Usage:
    python train_ship.py --config config/ship_pred30.yml --exp v1
"""

import argparse
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure MGF src/ is on the path
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from data.ship_loader import (
    create_ship_dataloaders,
    extract_futures_for_clustering,
    extract_target_ship,
)
from models.ship_mgf import ShipMGF


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(log_path):
    logger = logging.getLogger("MGF_Ship")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# KMeans clustering for GMM base distribution
# ---------------------------------------------------------------------------
def run_clustering(data_root, n_clusters, pred_len, save_dir):
    """Run KMeans on training futures and return path to saved pickle."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"ship_kmeans_{n_clusters}_pred{pred_len}.pkl")

    if os.path.exists(out_path):
        print(f"[clustering] Reusing existing cluster model: {out_path}")
        return out_path

    print(f"[clustering] Extracting futures for KMeans (K={n_clusters}) ...")
    futures, base_dirs, _ = extract_futures_for_clustering(data_root)
    print(f"  Got {futures.shape[0]} trajectories, pred_len={pred_len}")

    # Rotate to align COG direction
    cos_a = base_dirs[:, 0]
    sin_a = base_dirs[:, 1]
    x, y = futures[:, :, 0], futures[:, :, 1]
    rot_x = cos_a[:, None] * x + sin_a[:, None] * y
    rot_y = -sin_a[:, None] * x + cos_a[:, None] * y
    rotated = np.stack([rot_x, rot_y], axis=-1)
    flat = rotated.reshape(rotated.shape[0], -1)

    from sklearn.cluster import KMeans

    print(f"[clustering] Running KMeans ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(flat)
    print(f"  Inertia: {kmeans.inertia_:.4f}")

    with open(out_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"[clustering] Saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, n_sample=20):
    """Compute ADE_min / FDE_min over a DataLoader."""
    model.eval()
    ade_list, fde_list = [], []

    for batch in loader:
        data = extract_target_ship(batch)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        sampled = model.predict(data, n_sample=n_sample)  # (B, n_sample, pred_len, 2)
        gt = data["gt_st"]  # (B, pred_len, 2)

        # ADE / FDE per sample
        diff = sampled - gt.unsqueeze(1)  # (B, n_sample, T, 2)
        dist = torch.norm(diff, dim=-1)  # (B, n_sample, T)
        ade = dist.mean(dim=-1)  # (B, n_sample)
        fde = dist[:, :, -1]  # (B, n_sample)

        ade_min = ade.min(dim=1).values.mean().item()
        fde_min = fde.min(dim=1).values.mean().item()
        ade_list.append(ade_min)
        fde_list.append(fde_min)

    ade_avg = float(np.mean(ade_list))
    fde_avg = float(np.mean(fde_list))
    return ade_avg, fde_avg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg, logger, exp_name, device):
    data_root = cfg["data"]["data_root"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    obs_len = cfg["data"]["obs_len"]
    pred_len = cfg["data"]["pred_len"]

    # ---- Clustering ----
    n_clusters = cfg["model"]["n_clusters"]
    cluster_dir = os.path.join(os.path.dirname(__file__), "src", "clustering", "models")
    cluster_path = run_clustering(data_root, n_clusters, pred_len, cluster_dir)

    # ---- Data ----
    logger.info("Loading data ...")
    loaders, norm_params = create_ship_dataloaders(data_root, batch_size, num_workers)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    test_loader = loaders.get("test")

    # ---- Model ----
    model_cfg = {
        "obs_len": obs_len,
        "pred_len": pred_len,
        "input_dim": cfg["model"]["input_dim"],
        "d_model": cfg["model"]["d_model"],
        "cond_dim": cfg["model"]["cond_dim"],
        "n_heads": cfg["model"]["n_heads"],
        "n_enc_layers": cfg["model"]["n_enc_layers"],
        "n_dec_layers": cfg["model"]["n_dec_layers"],
        "dropout": cfg["model"]["dropout"],
        "n_blocks": cfg["model"]["n_blocks"],
        "flow_hidden": cfg["model"]["flow_hidden"],
        "n_hidden": cfg["model"]["n_hidden"],
        "var_init": cfg["model"]["var_init"],
        "learn_var": cfg["model"]["learn_var"],
        "dequantize": cfg["training"]["dequantize"],
        "grad_clip": cfg["training"]["grad_clip"],
        "lr": cfg["training"]["lr"],
        "weight_decay": cfg["training"]["weight_decay"],
        "cluster_path": cluster_path,
    }
    model = ShipMGF(model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model params: {n_params:.2f}M")

    # ---- Training loop ----
    epochs = cfg["training"]["epochs"]
    eval_every = cfg["training"]["eval_every"]
    w_mse = cfg["training"]["w_mse"]
    w_mse_start = cfg["training"]["w_mse_start_epoch"]
    save_dir = os.path.join(cfg["training"]["save_dir"], exp_name)
    os.makedirs(save_dir, exist_ok=True)

    best_ade = float("inf")
    best_fde = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_epoch = []
        t0 = time.time()

        use_mse = (w_mse > 0) and (epoch >= w_mse_start)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            data = extract_target_ship(batch)
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

            if use_mse:
                info = model.update_mse(data, w_mse)
            else:
                info = model.update(data)
            loss_epoch.append(info["loss"])
            pbar.set_postfix(loss=info["loss"])

        avg_loss = float(np.mean(loss_epoch))
        dt = time.time() - t0

        # ---- Validation ----
        val_ade, val_fde = 0.0, 0.0
        if val_loader and epoch % eval_every == 0:
            val_ade, val_fde = evaluate(model, val_loader, device, n_sample=20)
            logger.info(
                f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                f"val_ADE={val_ade:.4f} val_FDE={val_fde:.4f} | "
                f"best_ADE={best_ade:.4f} best_FDE={best_fde:.4f} | {dt:.1f}s"
            )
            if val_ade < best_ade:
                best_ade = val_ade
                model.save(os.path.join(save_dir, "best_ade.pt"), epoch=epoch)
            if val_fde < best_fde:
                best_fde = val_fde
                model.save(os.path.join(save_dir, "best_fde.pt"), epoch=epoch)
        else:
            logger.info(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | {dt:.1f}s")

        # Save last
        model.save(os.path.join(save_dir, "last.pt"), epoch=epoch)

    # ---- Final test ----
    if test_loader:
        logger.info("Loading best ADE checkpoint for test ...")
        best_ade_path = os.path.join(save_dir, "best_ade.pt")
        if os.path.exists(best_ade_path):
            model.load(best_ade_path)
        test_ade, test_fde = evaluate(model, test_loader, device, n_sample=20)
        logger.info(f"Test ADE_min={test_ade:.4f} nm | FDE_min={test_fde:.4f} nm")

    logger.info(f"Training complete. Results saved to {save_dir}")
    return best_ade, best_fde


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MGF Ship Training")
    parser.add_argument("--config", type=str, default="config/ship_pred30.yml")
    parser.add_argument("--exp", type=str, default="v1", help="experiment name")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve data_root relative to vessel-trajectory-prediction root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg["data"]["data_root"] = os.path.join(project_root, cfg["data"]["data_root"])

    # Resolve save_dir
    cfg["training"]["save_dir"] = os.path.join(
        os.path.dirname(__file__), "results_ship", "mgf"
    )

    # Logger
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"mgf_ship_pred30_{args.exp}.log")
    logger = setup_logger(log_path)

    logger.info(f"Config: {args.config}")
    logger.info(f"Data root: {cfg['data']['data_root']}")
    logger.info(f"Experiment: {args.exp}")
    logger.info(f"Device: {device}")

    train(cfg, logger, args.exp, device)


if __name__ == "__main__":
    main()
