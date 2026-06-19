import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange


def visual(true, preds=None, name='./pic/test.pdf'):
    """Visualization stub - not used in training."""
    pass


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    # type1: step decay
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def vali(model, vali_set, vali_loader, criterion, args, device, ii):
    """Validation loop - computes MSE loss."""
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, batch in enumerate(vali_loader):
            batch_x = batch['past_traj'].permute(0, 2, 1)  # [B, 2, seq_len] -> [B, seq_len, 2]
            batch_y = batch['future_traj'].permute(0, 2, 1)

            from models.geohash import geohash_encoding
            geohash_codeing = geohash_encoding(batch_x).to(device=device).float()
            geohash_codeing = rearrange(geohash_codeing, 'b l p c -> b l (p c)')

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            total_gaussian = vali_set.generate_gaussian_map(
                nodes_current=batch_x,
                grid_size=args.grid_size,
                sigma_x=args.sigmaX,
                sigma_y=args.sigmaY,
            )
            total_gaussian = total_gaussian.to(device)

            outputs = model(batch_x, total_gaussian, geohash_codeing)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def test(model, test_set, test_loader, args, device, ii):
    """Test loop - computes ADE and FDE metrics."""
    model.eval()
    all_ade = []
    all_fde = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), desc='Testing'):
            batch_x = batch['past_traj'].permute(0, 2, 1)
            batch_y = batch['future_traj'].permute(0, 2, 1)

            from models.geohash import geohash_encoding
            geohash_codeing = geohash_encoding(batch_x).to(device=device).float()
            geohash_codeing = rearrange(geohash_codeing, 'b l p c -> b l (p c)')

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            total_gaussian = test_set.generate_gaussian_map(
                nodes_current=batch_x,
                grid_size=args.grid_size,
                sigma_x=args.sigmaX,
                sigma_y=args.sigmaY,
            )
            total_gaussian = total_gaussian.to(device)

            outputs = model(batch_x, total_gaussian, geohash_codeing)
            pred = outputs[:, -args.pred_len:, :]
            target = batch_y[:, -args.pred_len:, :]

            # ADE: average displacement error across all prediction steps
            ade = torch.sqrt(((pred - target) ** 2).sum(dim=-1)).mean(dim=-1)
            # FDE: final displacement error at last prediction step
            fde = torch.sqrt(((pred[:, -1, :] - target[:, -1, :]) ** 2).sum(dim=-1))

            all_ade.extend(ade.cpu().numpy().tolist())
            all_fde.extend(fde.cpu().numpy().tolist())

    ade_arr = np.array(all_ade)
    fde_arr = np.array(all_fde)
    print(f"Test ADE: {ade_arr.mean():.4f}, FDE: {fde_arr.mean():.4f}")
    return ade_arr, fde_arr
