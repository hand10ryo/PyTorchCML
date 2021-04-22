import unittest

import torch

from PytorchCML.losses import MinTripletLoss


class TestMinTripletLoss(unittest.TestCase):
    def test_forward(self):
        """
        pos_dist = [[3], [3]]
        neg_dist = [[0,1,2], [3,4,5]]
        loss = [[10], [1]]
        avg_loss = 5.5
        """
        criterion = MinTripletLoss(margin=1)
        pos_dist_data = torch.ones(2, 1, 1) * 3
        neg_dist_data = torch.arange(6).reshape(2, 1, 3)
        loss = criterion(pos_dist_data, neg_dist_data).item()
        self.assertEqual(loss, 5.5)


if __name__ == '__main__':
    unittest.main()
