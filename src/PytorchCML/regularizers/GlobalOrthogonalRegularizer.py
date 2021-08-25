import torch
from torch import nn

from .BaseRegularizer import BaseRegularizer


class GlobalOrthogonalRegularizer(BaseRegularizer):
    """Class of Global Orthogonal Regularization"""

    def __init__(self, weight: float = 1e-2):
        super().__init__(weight)
        self.ReLU = nn.ReLU()

    def forward(self, embeddings_dict: dict) -> torch.Tensor:
        """Method of comuting regularize term

        Args:
            embeddings_dict (dict): dictionary of embeddings which has pos_item_emb and neg_item_emb

        Returns:
            torch.Tensor: term of regularize
        """

        pos_item_emb = embeddings_dict["pos_item_embedding"]
        neg_item_emb = embeddings_dict["neg_item_embedding"]

        B, N, d = neg_item_emb.shape
        Q = B * N

        inner = torch.einsum("bid,bnd->bn", pos_item_emb, neg_item_emb)

        M1 = inner.sum() / Q
        M2 = (inner ** 2).sum() / Q

        LGOR = M1 ** 2 + self.ReLU(M2 - (1 / d))

        return self.weight * LGOR
