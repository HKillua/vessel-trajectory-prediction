"""Ship Flow Matching evaluation entry point."""

import os
import torch
import argparse
from data.dataloader_ship import build_ship_dataloaders
from utils.config import Config
from utils.utils import set_random_seed, log_config_to_file
from models.flow_matching import FlowMatcher
from models.backbone_ship import ShipMotionTransformer
from trainer.denoising_model_trainers import Trainer


def parse_config():
    parser = argparse.ArgumentParser(description='Ship Flow Matching Evaluation')

    parser.add_argument('--cfg', default='cfg/ship/fm.yml', type=str)
    parser.add_argument('--exp', default='', type=str)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='DMA')
    parser.add_argument('--pred_len', type=int, default=10, choices=[10, 20, 30])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--ckpt', type=str, default='best', choices=['best', 'last'])
    parser.add_argument('--sampling_steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    # These need to match the training run
    parser.add_argument('--fm_wrapper', type=str, default='direct')
    parser.add_argument('--fm_in_scaling', default=False, action='store_true')
    parser.add_argument('--sigma_data', type=float, default=0.13)
    parser.add_argument('--drop_method', default=None)
    parser.add_argument('--drop_logi_k', default=20.0, type=float)
    parser.add_argument('--drop_logi_m', default=0.5, type=float)
    parser.add_argument('--tied_noise', default=False, action='store_true')
    parser.add_argument('--loss_nn_mode', type=str, default='scene')
    parser.add_argument('--loss_reg_reduction', type=str, default='mean')

    return parser.parse_args()


def main():
    args = parse_config()

    cfg = Config(args.cfg, f'{args.exp}')
    assert hasattr(cfg, 'dataset') and cfg.dataset == 'ship', (
        f"cfg/ship/fm.yml must set 'dataset: ship', got '{getattr(cfg, 'dataset', 'MISSING')}'"
    )

    out_dim = args.pred_len * 2
    cfg.future_frames = args.pred_len
    cfg.MODEL.MODEL_OUT_DIM = out_dim
    cfg.MODEL.REGRESSION_MLPS = [128, 256, out_dim]

    cfg.sampling_steps = args.sampling_steps
    cfg.sigma_data = args.sigma_data
    cfg.fm_wrapper = args.fm_wrapper
    cfg.fm_rew_sqrt = False
    cfg.fm_in_scaling = args.fm_in_scaling
    cfg.t_schedule = 'logit_normal'
    cfg.logit_norm_mean = -0.5
    cfg.logit_norm_std = 1.5
    cfg.tied_noise = args.tied_noise
    cfg.drop_method = args.drop_method
    cfg.drop_logi_k = args.drop_logi_k
    cfg.drop_logi_m = args.drop_logi_m
    cfg.LOSS_NN_MODE = args.loss_nn_mode
    cfg.LOSS_REG_REDUCTION = args.loss_reg_reduction
    cfg.objective = 'pred_data'

    cfg.train_batch_size = args.batch_size
    cfg.test_batch_size = args.batch_size

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = f'_ship_{args.dataset_name}_pred{args.pred_len}_eval'
    logger = cfg.create_dirs(tag_suffix=tag)

    set_random_seed(args.seed)

    _, _, test_loader = build_ship_dataloaders(
        data_root=args.data_dir, cfg=cfg,
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
        num_workers=4,
    )

    model = ShipMotionTransformer(cfg.MODEL, logger=logger, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=logger)

    trainer = Trainer(
        cfg, denoiser,
        test_loader, test_loader,  # train_loader=test_loader: eval-only, no training
        logger=logger,
        gradient_accumulate_every=1,
    )

    trainer.test(mode=args.ckpt)


if __name__ == '__main__':
    main()
