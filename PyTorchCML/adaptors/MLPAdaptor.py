import torch

from torch import nn

from .BaseAdaptor import BaseAdaptor


class MLPAdaptor(BaseAdaptor):
    """Class of module for domain adaptation with MLP."""

    def __init__(
        self,
        features: torch.Tensor,
        n_dim: int = 20,
        n_hidden: list = [100],
        weight: float = 1e-3,
    ):
        """Set MLP model for domain adaptation.

        Args:
            features (torch.Tensor): A feature of users or items. size = (n_user, n_feature)
            n_dim (int, optional): A number of dimention of embeddings. Defaults to 20.
            n_hidden (list, optional): A list of numbers of neuron for each hidden layers. Defaults to [100].
            weight (float, optional): Loss weights for domain adaptation. Defaults to 1e-3.
        """
        super().__init__(weight)
        self.features_embeddings = nn.Embedding.from_pretrained(features)
        self.features_embeddings.weight.requires_grad = False

        self.n_input = features.shape[1]
        self.n_hidden = n_hidden
        self.n_output = n_dim

        projection_layers = [nn.Linear(self.n_input, self.n_hidden[0]), nn.ReLU()]
        for i in range(len(self.n_hidden) - 1):
            layer = [nn.Linear(self.n_hidden[i], self.n_hidden[i + 1]), nn.ReLU()]
            projection_layers += layer
        projection_layers += [nn.Linear(self.n_hidden[-1], self.n_output)]

        self.projector = nn.Sequential(*projection_layers)

    def forward(self, indices: torch.Tensor, embeddings: torch.Tensor):
        """Method to calculate loss for domain adaptation.

        Args:
            indices (torch.Tensor): Indices of users or items. size = (n_user, n_sample)
            embeddings (torch.Tensor): The embeddings corresponding to indices. size = (n_user, n_sample, n_dim)

        Returns:
            [torch.Tensor]: loss for domain adaptation. dim = 0.
        """
        features = self.features_embeddings(indices)
        projection = self.projector(features)
        dist = torch.sqrt(torch.pow(projection - embeddings, 2).sum(axis=2))
        return self.weight * dist.sum()
