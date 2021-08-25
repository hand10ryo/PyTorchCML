import torch

from .BaseRegularizer import BaseRegularizer


class L2Regularizer(BaseRegularizer):
    """Class of L2 Regularization"""

    def forward(self, embeddings_dict: dict) -> torch.Tensor:
        """Method of comuting regularize term

        Args:
            embeddings_dict (dict): dictionary of embeddings which has pos_item_emb and neg_item_emb

        Returns:
            torch.Tensor: term of regularize
        """

        user_emb = embeddings_dict["user_embedding"]
        pos_item_emb = embeddings_dict["pos_item_embedding"]
        neg_item_emb = embeddings_dict["neg_item_embedding"]

        user_norm = (user_emb ** 2).sum()
        pos_item_norm = (pos_item_emb ** 2).sum()
        neg_item_norm = (neg_item_emb ** 2).sum()

        norm = user_norm + pos_item_norm + neg_item_norm
        return self.weight * norm
