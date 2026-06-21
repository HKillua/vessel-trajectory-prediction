"""Ship Flow Matching training entry point."""

import os
import torch
import argparse
from tensorboardX import SummaryWriter

from data.dataloader_ship import build_ship_dataloaders
from utils.config import Config
from utils.utils import set_random_seed, log_config_to_file

from models.flow_matching import FlowMatcher
from models.backbone_ship import ShipMotionTransformer
from trainer.denoising_model_trainers import Trainer


def parse_config():
    parser = argparse.ArgumentParser(description='Ship Flow Matching Training')

    parser.add_argument('--cfg', default='cfg/ship/fm.yml', type=str)
    parser.add_argument('--exp', default='', type=str)

    # Data configuration
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to obs{X}_pred{Y} directory with train/val/test splits')
    parser.add_argument('--dataset_name', type=str, default='NOAANY')
    parser.add_argument('--pred_len', type=int, default=30, choices=[10, 20, 30])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--checkpt_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    # FM parameters
    parser.add_argument('--sampling_steps', type=int, default=10)
    parser.add_argument('--t_schedule', type=str, default='logit_normal')
    parser.add_argument('--logit_norm_mean', default=-0.5, type=float)
    parser.add_argument('--logit_norm_std', default=1.5, type=float)
    parser.add_argument('--fm_wrapper', type=str, default='direct')
    parser.add_argument('--fm_in_scaling', default=False, action='store_true')
    parser.add_argument('--sigma_data', type=float, default=0.13)
    parser.add_argument('--drop_method', default=None)
    parser.add_argument('--drop_logi_k', default=20.0, type=float)
    parser.add_argument('--drop_logi_m', default=0.5, type=float)
    parser.add_argument('--tied_noise', default=False, action='store_true')

    # Loss configuration
    parser.add_argument('--loss_nn_mode', type=str, default='agent')
    parser.add_argument('--loss_reg_reduction', type=str, default='sum')
    parser.add_argument('--loss_reg_squared', default=False, action='store_true')

    # Optimization
    parser.add_argument('--init_lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)

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
    cfg.t_schedule = args.t_schedule
    cfg.logit_norm_mean = args.logit_norm_mean
    cfg.logit_norm_std = args.logit_norm_std
    cfg.tied_noise = args.tied_noise
    cfg.drop_method = args.drop_method
    cfg.drop_logi_k = args.drop_logi_k
    cfg.drop_logi_m = args.drop_logi_m
    cfg.LOSS_NN_MODE = args.loss_nn_mode
    cfg.LOSS_REG_REDUCTION = args.loss_reg_reduction
    cfg.LOSS_REG_SQUARED = args.loss_reg_squared
    cfg.objective = 'pred_data'

    if args.epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size
        cfg.test_batch_size = args.batch_size * 2
    cfg.checkpt_freq = args.checkpt_freq
    cfg.max_num_ckpts = 5

    if args.init_lr is not None:
        cfg.OPTIMIZATION.LR = args.init_lr
    if args.weight_decay is not None:
        cfg.OPTIMIZATION.WEIGHT_DECAY = args.weight_decay
    if args.warmup_epochs is not None:
        cfg.OPTIMIZATION.WARMUP_EPOCHS = args.warmup_epochs

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = f'_ship_{args.dataset_name}_pred{args.pred_len}'
    logger = cfg.create_dirs(tag_suffix=tag)

    set_random_seed(args.seed)

    log_config_to_file(cfg.yml_dict, logger=logger)

    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, '../tb'))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

    train_loader, val_loader, test_loader = build_ship_dataloaders(
        data_root=args.data_dir, cfg=cfg,
        batch_size_train=cfg.train_batch_size,
        batch_size_test=cfg.test_batch_size,
        num_workers=4,
    )

    model = ShipMotionTransformer(cfg.MODEL, logger=logger, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=logger)

    trainer = Trainer(
        cfg, denoiser,
        train_loader, test_loader,
        val_loader=val_loader,
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
    )

    trainer.train()


if __name__ == '__main__':
    main()
