import torch
from torch import nn


class BaseTripletLoss(nn.Module):
    """Class of Abstract Loss for Triplet"""

    def __init__(self, margin: float = 1, regularizers: list = []):
        """Set margin size and ReLU function

        Args:
            margin (float, optional): safe margin size. Defaults to 1.
            regularizers (list, optional): list of regularizer
        """
        super().__init__()
        self.margin = margin
        self.ReLU = nn.ReLU()
        self.regularizers = regularizers

    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Method of forward

        Args:
            user_emb : embeddings of user size (n_batch, 1, d)
            pos_item_emb : embeddings of positive item size (n_batch, 1, d)
            neg_item_emb : embeddings of negative item size (n_batch, n_neg_samples, d)

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [description]


        ---- example code ---

        embeddings_dict = {
            "user_emb": user_emb,
            "pos_item_emb": pos_item_emb,
            "neg_item_emb": neg_item_emb,
        }


        loss = some_function(user_emb, pos_item_emb, neg_item_emb)
        reg = self.regularize(embeddings_dict)
        return loss + reg

        """

        raise NotImplementedError

    def regularize(self, embeddings_dict: dict):
        reg = 0
        for regularizer in self.regularizers:
            reg += regularizer(embeddings_dict)

        return reg
