import unittest

import torch
from torch import nn

from PytorchCML.losses import MinTripletLoss


class SampleRegularizer(nn.Module):
    def forward(self, embeding_dict: dict) -> torch.Tensor:
        return torch.ones(3).sum()


class TestMinTripletLoss(unittest.TestCase):
    """Test MnTripletLoss"""

    def test_forward(self):
        """
        test forward
        """
        user_emb = torch.ones(3, 1, 5)
        pos_item_emb = torch.ones(3, 1, 5) * 2
        neg_item_emb = torch.ones(3, 1, 5)

        # without regularizer
        criterion = MinTripletLoss(margin=1)
        loss = criterion(user_emb, pos_item_emb, neg_item_emb).item()
        self.assertGreater(loss, 0)
        self.assertEqual(loss, 6)

        # with regularizer
        regs = [SampleRegularizer()]
        criterion = MinTripletLoss(margin=1, regularizers=regs)
        loss = criterion(user_emb, pos_item_emb, neg_item_emb).item()
        self.assertGreater(loss, 0)
        self.assertEqual(loss, 9)


if __name__ == '__main__':
    unittest.main()
