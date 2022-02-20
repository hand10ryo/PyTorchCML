import torch
from torch import nn

from .BaseLoss import BaseLoss


class LogitPairwiseLoss(BaseLoss):
    """Class of pairwise logit loss for Logistic Matrix Factorization"""

    def __init__(self, regularizers: list = []):
        super().__init__(regularizers)
        self.LogSigmoid = nn.LogSigmoid()

    def main(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        """Method of forwarding main loss

        Args:
            embeddings_dict (dict): A dictionary of embddings which has following key and values.
                user_embedding : embeddings of user, size (n_batch, 1, d)
                pos_item_embedding : embeddings of positive item, size (n_batch, 1, d)
                neg_item_embedding : embeddings of negative item, size (n_batch, n_neg_samples, d)
                user_bias : bias of user, size (n_batch, 1)
                pos_item_bias : bias of positive item, size (n_batch, 1)
                neg_item_bias : bias of negative item, size (n_batch, n_neg_samples)

            batch (torch.Tensor) : A tensor of batch, size (n_batch, *).
            column_names (dict) : A dictionary that maps names to indices of rows of batch.
        """

        n_batch = embeddings_dict["user_embedding"].shape[0]
        n_neg = embeddings_dict["neg_item_bias"].shape[1]
        n_pos = 1

        pos_inner = torch.einsum(
            "nid,nid->n",
            embeddings_dict["user_embedding"],
            embeddings_dict["pos_item_embedding"],
        )

        neg_inner = torch.einsum(
            "nid,njd->nj",
            embeddings_dict["user_embedding"],
            embeddings_dict["neg_item_embedding"],
        )

        pos_bias = embeddings_dict["user_bias"] + embeddings_dict["pos_item_bias"]
        pos_y_hat = pos_inner + pos_bias.reshape(-1)

        neg_bias = embeddings_dict["user_bias"] + embeddings_dict["neg_item_bias"]
        neg_y_hat = neg_inner + neg_bias

        pos_loss = -nn.LogSigmoid()(pos_y_hat).sum()
        neg_loss = -nn.LogSigmoid()(-neg_y_hat).sum()

        loss = (pos_loss + neg_loss) / (n_batch * (n_pos + n_neg))

        return loss
