from torch import nn


class BaseAdaptor(nn.Module):
    """Astract class of module for domain adaptation."""

    def __init__(self, weight):
        """Set some parameters

        Args:
            weight (float, optional): Loss weights for domain adaptation. Defaults to 1e-3.
        """
        super().__init__()
        self.weight = weight

    def forward(self, indices, embeddings):
        """Method to calculate loss for domain adaptation.

        Args:
            indices (torch.Tensor): Indices of users or items. size = (n_user, n_sample)
            embeddings (torch.Tensor): The embeddings corresponding to indices. size = (n_user, n_sample, n_dim)

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError
