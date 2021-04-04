import torch
from torch import nn

from .BaseTripletLoss import BaseTripletLoss


class SumTripletLoss(BaseTripletLoss):
    def __init__(self, margin: float = 1):
        super().__init__(margin)

    def forward(self, pos_dist: torch.Tensor, neg_dist: torch.Tensor, weight=None) -> torch.Tensor:
        """
        Args:
            pos_dist : distance of pos pairs of size (n_batch, 1, 1)
            neg_dist : distance of pos pairs of size (n_batch, 1, n_neg_samples)
            weight : sample weight
        """
        tripletloss = self.ReLU(self.margin + pos_dist ** 2 - neg_dist ** 2)
        loss = torch.sum(tripletloss)
        return loss
