#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import yaml
from mambadiff.models.mambadiff_ecr import MambaDiffECR
from data_provider.dataloader_multivessel import create_dataloaders

with open("mambadiff/configs/mambadiff_ecr.yaml") as f:
    cfg = yaml.safe_load(f)

model = MambaDiffECR(cfg).to("cpu")
model.init_diffusion(cfg, device="cpu")
enc_type = cfg["model"]["encoder_type"]
print(f"Encoder type: {enc_type}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

loaders = create_dataloaders("ship_trajectory_prediction/data/final/obs10_pred10", batch_size=4, num_workers=0)
batch = next(iter(loaders["train"]))
print(f"obs: {batch['obs'].shape}, pred: {batch['pred'].shape}")

model.eval()
with torch.no_grad():
    pred, anchor = model(batch)
print(f"pred: {pred.shape}, anchor: {anchor.shape}")
print("FORWARD PASS OK!")
