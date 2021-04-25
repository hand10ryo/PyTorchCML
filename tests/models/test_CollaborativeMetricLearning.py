import unittest

import torch

from PytorchCML.models import CollaborativeMetricLearning


class TestCollaborativeMetricLearning(unittest.TestCase):

    def test_forward(self):
        users = torch.LongTensor([[0], [1]])
        pos_items = torch.LongTensor([[2], [3]])
        neg_items = torch.LongTensor([[1, 3, 4], [0, 1, 2]])

        model = CollaborativeMetricLearning(
            n_user=3, n_item=5, n_dim=10,
        )

        user_emb, pos_item_emb, neg_item_emb = model(
            users, pos_items, neg_items
        )

        # user_emb shape
        shape = user_emb.shape
        self.assertEqual(shape, (2, 1, 10))

        # pos_item_emb shape
        shape = pos_item_emb.shape
        self.assertEqual(shape, (2, 1, 10))

        # item_emb shape
        shape = neg_item_emb.shape
        self.assertEqual(shape, (2, 3, 10))

    def test_predict(self):
        user_item_pairs = torch.LongTensor([
            [0, 0], [1, 1], [2, 2]
        ])
        model = CollaborativeMetricLearning(
            n_user=3, n_item=5, n_dim=10,
        )

        y_hat = model.predict(user_item_pairs)

        # y_hat shape
        shape = y_hat.shape
        self.assertEqual(shape, torch.Size([3]))

    def test_spreadout_distance(self):
        pos_items = torch.LongTensor([[2], [3]])
        neg_items = torch.LongTensor([0, 1, 4])

        model = CollaborativeMetricLearning(
            n_user=3, n_item=5, n_dim=10,
        )

        so_dist = model.spreadout_distance(pos_items, neg_items)

        # y_hat shape
        shape = so_dist.shape
        self.assertEqual(shape, torch.Size([2, 3]))
