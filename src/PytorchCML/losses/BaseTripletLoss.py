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

    def forward(self, user_emb: torch.Tensor,
                pos_item_emb: torch.Tensor,
                neg_item_emb: torch.Tensor) -> torch.Tensor:
        """ Method of forward

        Args:
            user_emb : embeddings of user size (n_batch, 1, d)
            pos_item_emb : embeddings of positive item size (n_batch, 1, d)
            neg_item_emb : embeddings of negative item size (n_batch, n_neg_samples, d)

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [description]
        """
        raise NotImplementedError
