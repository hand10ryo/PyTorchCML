import torch
from torch import nn


class BaseTripletLoss(nn.Module):
    def __init__(self, margin: float = 1):
        super().__init__()
        self.margin = margin
        self.ReLU = nn.ReLU()

    def forward(self, pos_dist: torch.Tensor, neg_dist: torch.Tensor, weight=None) -> torch.Tensor:
        """
        Args:
            pos_dist : distance of pos pairs of size (n_batch, 1, 1)
            neg_dist : distance of pos pairs of size (n_batch, 1, n_neg_samples)
            weight : sample weight
        """
        raise NotImplementedError
