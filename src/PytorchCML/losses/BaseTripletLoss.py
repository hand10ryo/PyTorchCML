import torch
from torch import nn


class BaseTripletLoss(nn.Module):
    """Class of Abstract Loss for Triplet
    """

    def __init__(self, margin: float = 1):
        """ Set margin size and ReLU function

        Args:
            margin (float, optional): safe margin size. Defaults to 1.
        """
        super().__init__()
        self.margin = margin
        self.ReLU = nn.ReLU()

    def forward(self, pos_dist: torch.Tensor, neg_dist: torch.Tensor, weight=None) -> torch.Tensor:
        """ Method of forward

        Args:
            pos_dist (torch.Tensor): distance of pos pairs of size (n_batch, 1, 1)
            neg_dist (torch.Tensor): distance of pos pairs of size (n_batch, 1, n_neg_samples)
            weight ([type], optional): sample weight. Defaults to None.

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [description]
        """
        raise NotImplementedError
