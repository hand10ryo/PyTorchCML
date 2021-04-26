import torch
from torch import nn

from .BaseTripletLoss import BaseTripletLoss


class MinTripletLoss(BaseTripletLoss):
    def __init__(self, margin: float = 1, regularizers: list = []):
        """ Class of Triplet Loss taking minimum negative sample.
        """
        super().__init__(margin, regularizers)

    def forward(self, user_emb: torch.Tensor,
                pos_item_emb: torch.Tensor,
                neg_item_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_emb : embeddings of user size (n_batch, 1, d)
            pos_item_emb : embeddings of positive item size (n_batch, 1, d)
            neg_item_emb : embeddings of negative item size (n_batch, n_neg_samples, d)

        Return:
            loss : L = Σ [m + pos_dist^2 - min(neg_dist)^2]
        """
        embeddings_dict = {
            "user_emb": user_emb,
            "pos_item_emb": pos_item_emb,
            "neg_item_emb": neg_item_emb
        }

        pos_dist = torch.cdist(user_emb, pos_item_emb)
        neg_dist = torch.cdist(user_emb, neg_item_emb)

        min_neg_dist = torch.min(neg_dist, axis=2)
        pairwiseloss = self.ReLU(
            self.margin + pos_dist ** 2 - min_neg_dist.values ** 2)

        loss = torch.mean(pairwiseloss)
        reg = self.regularize(embeddings_dict)

        return loss + reg
