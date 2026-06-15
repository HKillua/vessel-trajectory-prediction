import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mambadiff.trainer.trainer import ECRTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='mambadiff/configs/mambadiff_ecr.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='results/mambadiff')
    parser.add_argument('--phase', type=str, default='all', choices=['pretrain', 'train', 'all', 'test'])
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=0.5,
                        help='COLREGs guidance scale for test-time comparison (0=disabled)')
    parser.add_argument('--cfg_scale', type=float, default=0.0,
                        help='Classifier-free guidance scale (0=disabled, 1.5=typical)')
    parser.add_argument('--oracle_guidance', action='store_true',
                        help='Use ground truth neighbor trajectories for COLREGs guidance (ablation)')
    parser.add_argument('--no_report', action='store_true',
                        help='Skip auto-generating analysis report after training')
    args = parser.parse_args()

    trainer = ECRTrainer(args.cfg, device=args.device, log_dir=args.log_dir)

    if args.debug:
        trainer.cfg['training']['debug'] = True

    if args.phase == 'pretrain':
        trainer.pretrain_denoiser(epochs=args.pretrain_epochs)
    elif args.phase == 'train':
        trainer.train(pretrained_denoiser_path=args.pretrained)
    elif args.phase == 'all':
        trainer.pretrain_denoiser(epochs=args.pretrain_epochs)
        best_p1 = os.path.join(args.log_dir, 'checkpoint_phase1_best.pt')
        trainer.train(pretrained_denoiser_path=best_p1 if os.path.exists(best_p1) else None)
    elif args.phase == 'test':
        trainer.test(checkpoint_path=args.checkpoint, guidance_scale=args.guidance_scale,
                     cfg_scale=args.cfg_scale, use_gt_neighbors=args.oracle_guidance)

    if not args.no_report and args.phase == 'test':
        from mambadiff.analyze import generate_report
        report = generate_report(args.log_dir)
        report_path = os.path.join(args.log_dir, 'analysis_report.md')
        with open(report_path, 'w',encoding='utf-8') as f:
            f.write(report)
        print(f"Analysis report generated: {report_path}")


if __name__ == '__main__':
    main()