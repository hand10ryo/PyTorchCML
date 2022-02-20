import unittest

import torch
from torch import nn

from PyTorchCML.adaptors import MLPAdaptor


class TestMLPAdaptor(unittest.TestCase):
    """Test MLPAdaptor"""

    def test_forward(self):
        user_features = torch.ones(3, 4) / 10
        item_features = torch.ones(5, 6) / 10

        user_adaptor = MLPAdaptor(user_features, n_hidden=[10, 10], n_dim=4, weight=1)
        item_adaptor = MLPAdaptor(item_features, n_hidden=[10, 10], n_dim=4, weight=1)

        for param in user_adaptor.parameters():
            nn.init.constant_(param, 0.1)

        for param in item_adaptor.parameters():
            nn.init.constant_(param, 0.1)

        users = torch.LongTensor([[0], [1]])
        pos_items = torch.LongTensor([[2], [3]])
        neg_items = torch.LongTensor([[0, 1, 3, 4], [0, 2, 3, 4]])

        embeddings_dict = {
            "user_embedding": torch.ones(2, 1, 4) / 10,
            "pos_item_embedding": torch.ones(2, 1, 4) / 10,
            "neg_item_embedding": torch.ones(2, 4, 4) * 2 / 10,
        }

        user_loss = user_adaptor(users, embeddings_dict["user_embedding"])
        pos_item_loss = item_adaptor(pos_items, embeddings_dict["pos_item_embedding"])
        neg_item_loss = item_adaptor(neg_items, embeddings_dict["neg_item_embedding"])

        self.assertAlmostEqual(user_loss.item(), 0.96, places=5)
        self.assertAlmostEqual(pos_item_loss.item(), 1.04, places=5)
        self.assertAlmostEqual(neg_item_loss.item(), 2.56, places=5)
