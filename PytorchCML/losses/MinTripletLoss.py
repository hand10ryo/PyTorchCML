import torch

from .BaseTripletLoss import BaseTripletLoss


class MinTripletLoss(BaseTripletLoss):
    def __init__(self, margin: float = 1, regularizers: list = []):
        """Class of Triplet Loss taking minimum negative sample."""
        super().__init__(margin, regularizers)

    def forward(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        """Method of forwarding loss

        Args:
            embeddings_dict (dict): A dictionary of embddings which has following key and values.
                user_embedding : embeddings of user, size (n_batch, d)
                pos_item_embedding : embeddings of positive item, size (n_batch, d)
                neg_item_embedding : embeddings of negative item, size (n_batch, n_neg_samples, d)

            batch (torch.Tensor) : A tensor of batch, size (n_batch, *).
            column_names (dict) : A dictionary that maps names to indices of rows of batch.

        Return:
            torch.Tensor: loss,  L = Σ [m + pos_dist^2 - min(neg_dist)^2]
        """

        pos_dist = torch.cdist(
            embeddings_dict["user_embedding"], embeddings_dict["pos_item_embedding"]
        )

        neg_dist = torch.cdist(
            embeddings_dict["user_embedding"], embeddings_dict["neg_item_embedding"]
        )

        min_neg_dist = torch.min(neg_dist, axis=2)
        pairwiseloss = self.ReLU(self.margin + pos_dist ** 2 - min_neg_dist.values ** 2)

        loss = torch.mean(pairwiseloss)
        reg = self.regularize(embeddings_dict)

        return loss + reg