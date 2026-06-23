"""Training entry point for ShipMGF.

Usage:
    cd baselines/MGF
    python -m src.train_ship --config config/ship_pred30.yml
"""

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data.ship_loader import create_ship_dataloaders, extract_target_ship
from metrics.ship_metrics import evaluate
from models.ship_mgf import ShipMGF


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    data_cfg = raw_cfg["data"]
    model_cfg = raw_cfg["model"]
    train_cfg = raw_cfg["training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(project_root, data_cfg["data_root"])

    print(f"Loading data from {data_root} ...")
    loaders, norm_params = create_ship_dataloaders(
        data_root,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    if "train" not in loaders:
        print("ERROR: No training data found.")
        sys.exit(1)

    # --- Clustering ---
    mgf_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cluster_path = os.path.join(
        mgf_root, "src", "clustering", "models",
        f"ship_train_kmeans_{model_cfg['n_clusters']}_pred{data_cfg['pred_len']}.pkl"
    )

    if not os.path.exists(cluster_path):
        print(f"Cluster model not found at {cluster_path}")
        print("Running clustering first ...")
        from clustering.cluster_ship import main as cluster_main
        saved_argv = sys.argv
        try:
            sys.argv = [
                "cluster_ship",
                "--data_root", data_root,
                "--n_clusters", str(model_cfg["n_clusters"]),
                "--output_dir", os.path.join(mgf_root, "src", "clustering", "models"),
            ]
            cluster_main()
        finally:
            sys.argv = saved_argv
        print()

    # --- Model ---
    cfg = {
        **model_cfg,
        "obs_len": data_cfg["obs_len"],
        "pred_len": data_cfg["pred_len"],
        "cluster_path": cluster_path,
        "lr": train_cfg["lr"],
        "weight_decay": train_cfg["weight_decay"],
        "dequantize": train_cfg.get("dequantize", False),
    }

    model = ShipMGF(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    start_epoch = 0
    if args.resume:
        start_epoch = model.load(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # --- Output directory ---
    save_dir = os.path.join(project_root, train_cfg["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # --- Training ---
    epochs = train_cfg["epochs"]
    eval_every = train_cfg["eval_every"]
    w_mse = train_cfg.get("w_mse", 0.0)
    w_mse_start = train_cfg.get("w_mse_start_epoch", 0)

    best_ade = float("inf")
    best_fde = float("inf")

    print(f"\nTraining for {epochs} epochs, eval every {eval_every} ...")
    print(f"w_mse={w_mse}, w_mse starts at epoch {w_mse_start}")
    print(f"Save dir: {save_dir}\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_flow = 0.0
        epoch_mse = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loaders["train"]:
            data_dict = extract_target_ship(batch)
            data_dict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data_dict.items()
            }

            use_mse = w_mse > 0 and epoch >= w_mse_start
            if use_mse:
                result = model.update_mse(data_dict, w_mse)
                epoch_flow += result["flow_loss"]
                epoch_mse += result["mse_loss"]
            else:
                result = model.update(data_dict)

            epoch_loss += result["loss"]
            n_batches += 1

        dt = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)

        log_parts = [f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}"]
        if w_mse > 0 and epoch >= w_mse_start:
            log_parts.append(f"flow={epoch_flow/n_batches:.4f}")
            log_parts.append(f"mse={epoch_mse/n_batches:.4f}")
        log_parts.append(f"time={dt:.1f}s")
        print("  ".join(log_parts))

        # --- Evaluation ---
        if (epoch + 1) % eval_every == 0 and "val" in loaders:
            metrics = evaluate(
                model, loaders["val"], norm_params, extract_target_ship,
                n_sample=20, device=device,
            )
            ade, fde = metrics["ade"], metrics["fde"]
            print(f"  [Val] ADE={ade:.4f} NM  FDE={fde:.4f} NM")

            if ade < best_ade:
                best_ade = ade
                model.save(os.path.join(save_dir, "best_ade.pt"), epoch + 1)
                print(f"  -> New best ADE: {best_ade:.4f}")

            if fde < best_fde:
                best_fde = fde
                model.save(os.path.join(save_dir, "best_fde.pt"), epoch + 1)
                print(f"  -> New best FDE: {best_fde:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % (eval_every * 5) == 0:
            model.save(os.path.join(save_dir, f"epoch_{epoch+1}.pt"), epoch + 1)

    # --- Final save ---
    model.save(os.path.join(save_dir, "final.pt"), epochs)
    print(f"\nTraining complete. Best ADE={best_ade:.4f}, Best FDE={best_fde:.4f}")

    # --- Test evaluation ---
    if "test" in loaders:
        best_path = os.path.join(save_dir, "best_ade.pt")
        if os.path.exists(best_path):
            model.load(best_path)
        metrics = evaluate(
            model, loaders["test"], norm_params, extract_target_ship,
            n_sample=20, device=device,
        )
        print(f"[Test] ADE={metrics['ade']:.4f} NM  FDE={metrics['fde']:.4f} NM")


if __name__ == "__main__":
    main()
