import unittest

import torch
from torch import nn

from PyTorchCML.losses import MSEPairwiseLoss


class SampleRegularizer(nn.Module):
    def forward(self, embeding_dict: dict) -> torch.Tensor:
        return torch.ones(3).sum()


class TestMSEPairwiseLoss(unittest.TestCase):
    """Test LogitPairwiseLoss"""

    def test_forward(self):
        """
        test forward

        pos_inner = [[10], [10], [10]]
        neg_inner = [[-5, -5], [-5, -5], [-5, -5]]
        pos_bias = [3, 3, 3]
        neg_bias = [0, 0, 0]

        pos_r_hat = [13, 13, 13]
        neg_r_hat = [[-5, -5], [-5, -5], [-5, -5]]
        pos_loss = [-24, -24, -24]
        neg_loss = [[25, 25], [25, 25], [25, 25]]
        avg_loss = 26 / 3
        """
        embeddings_dict = {
            "user_embedding": torch.zeros(3, 5),
            "pos_item_embedding": torch.zeros(3, 5) * 2,
            "neg_item_embedding": -torch.zeros(3, 2, 5),
            "user_bias": torch.zeros(3, 1),
            "pos_item_bias": torch.zeros(3, 1) * 2,
            "neg_item_bias": -torch.zeros(3, 2),
        }
        batch = torch.ones([3, 2])
        column_names = {"user_id": 0, "item_id": 1}

        # without regularizer
        criterion = MSEPairwiseLoss()
        loss = criterion(embeddings_dict, batch, column_names).item()

        self.assertGreater(loss, 0)
        self.assertAlmostEqual(loss, 0.25, places=3)

        # with regularizer
        regs = [SampleRegularizer()]
        criterion = MSEPairwiseLoss(regularizers=regs)
        loss = criterion(embeddings_dict, batch, column_names).item()
        self.assertGreater(loss, 0)
        self.assertAlmostEqual(loss, 3.25, places=3)


if __name__ == "__main__":
    unittest.main()
