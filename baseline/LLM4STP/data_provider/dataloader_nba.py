"""Stub for dataloader_nba - only seq_collate is needed by mainSTP_Geo.py."""
import torch


def seq_collate(batch):
    """Collate function for trajectory data.
    Each item is a dict with 'past_traj' and 'future_traj' tensors.
    """
    past_traj = torch.stack([item['past_traj'] for item in batch], dim=0)
    future_traj = torch.stack([item['future_traj'] for item in batch], dim=0)
    return {
        'past_traj': past_traj,
        'future_traj': future_traj,
    }


class NBADataset:
    """Stub - not used."""
    pass
