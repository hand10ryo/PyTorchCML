import unittest

import torch

from PytorchCML.losses import MinTripletLoss


class TestMinTripletLoss(unittest.TestCase):
    """Test MnTripletLoss"""

    def test_forward(self):
        """
        test forward
        """
        criterion = MinTripletLoss(margin=1)

        user_emb = torch.ones(3, 1, 5)
        pos_item_emb = torch.ones(3, 1, 5) * 2
        neg_item_emb = torch.ones(3, 1, 5)

        loss = criterion(user_emb, pos_item_emb, neg_item_emb).item()
        self.assertGreater(loss, 0)
        self.assertEqual(loss, 6)


if __name__ == '__main__':
    unittest.main()
