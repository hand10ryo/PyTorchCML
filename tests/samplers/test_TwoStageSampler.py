import unittest

import torch
import numpy as np

from PyTorchCML.samplers import TwoStageSampler


class TestTwoStageSampler(unittest.TestCase):
    """Test BaseSampler"""

    def test_get_pos_batch(self):
        """
        test get_pos_batch
        """
        train_set = torch.LongTensor([[0, 0], [0, 2], [1, 1], [1, 3], [2, 3], [2, 4]])

        sampler = TwoStageSampler(
            train_set,
            n_user=3,
            n_item=5,
            batch_size=3,
            n_neg_samples=2,
            # strict_negative=True,
        )
        pos_batch = sampler.get_pos_batch()

        # shape
        n, m = pos_batch.shape
        self.assertEqual(n, 3)
        self.assertEqual(m, 2)

        # range of user id and item id
        user_id_min = pos_batch[:, 0].min().item()
        item_id_min = pos_batch[:, 1].min().item()
        user_id_max = pos_batch[:, 0].max().item()
        item_id_max = pos_batch[:, 1].max().item()
        self.assertGreaterEqual(user_id_min, 0)
        self.assertGreaterEqual(item_id_min, 0)
        self.assertGreaterEqual(3, user_id_max)
        self.assertGreaterEqual(5, item_id_max)

        # pairwise weighted sampler
        pos_weight_pair = np.array([1, 1, 1, 1, 1, 1000])
        sampler = TwoStageSampler(
            train_set,
            n_user=3,
            n_item=5,
            pos_weight=pos_weight_pair,
            batch_size=100,
            n_neg_samples=2,
            strict_negative=True,
        )
        pos_batch = sampler.get_pos_batch()
        cnt_heavy = (pos_batch[:, 1] == 4).sum().item()
        cnt_lignt = (pos_batch[:, 1] == 0).sum().item()
        self.assertGreaterEqual(cnt_heavy, cnt_lignt)

        # item wise weighted sampler
        pos_weight_item = np.array([1, 1, 1, 1, 1000])
        sampler = TwoStageSampler(
            train_set,
            n_user=3,
            n_item=5,
            pos_weight=pos_weight_item,
            batch_size=100,
            n_neg_samples=2,
            strict_negative=True,
        )
        pos_batch = sampler.get_pos_batch()
        cnt_heavy = (pos_batch[:, 1] == 4).sum().item()
        cnt_lignt = (pos_batch[:, 1] == 0).sum().item()
        self.assertGreaterEqual(cnt_heavy, cnt_lignt)

    def test_get_neg_batch(self):
        """
        test get_neg_batch
        """
        train_set = torch.LongTensor([[0, 0], [0, 2], [1, 1], [1, 3], [2, 3], [2, 4]])
        interactions = {
            0: [0, 2],
            1: [1, 3],
            2: [3, 4],
        }
        sampler = TwoStageSampler(
            train_set,
            n_user=3,
            n_item=15,
            batch_size=3,
            n_neg_samples=2,
            n_neg_candidates=5,
            strict_negative=True,
        )
        pos_batch = sampler.get_pos_batch()
        users = pos_batch[:, 0:1]

        # two stage
        sampler.get_and_set_candidates()
        dist = torch.ones(3, 5)
        sampler.set_candidates_weight(dist, 3)
        neg_batch = sampler.get_neg_batch(users.reshape(-1))

        # shape
        n, m = neg_batch.shape
        self.assertEqual(n, 3)
        self.assertEqual(m, 2)

        # range of item id
        item_id_min = neg_batch.min().item()
        item_id_max = neg_batch.max().item()
        self.assertGreaterEqual(item_id_min, 0)
        self.assertGreaterEqual(15, item_id_max)

        # strict negative
        for k, u in enumerate(users):
            for i in interactions[u.item()]:
                self.assertNotIn(i, neg_batch[k])

        # weighted sampling
        neg_weight = np.array([1, 1, 1, 1, 100])
        sampler = TwoStageSampler(
            train_set,
            n_user=3,
            n_item=5,
            neg_weight=neg_weight,
            batch_size=100,
            n_neg_samples=2,
            n_neg_candidates=3,
            # strict_negative=True,
        )
        pos_batch = sampler.get_pos_batch()
        users = pos_batch[:, 0:1]

        # two stage
        sampler.get_and_set_candidates()
        dist = torch.ones(100, 3)
        sampler.set_candidates_weight(dist, 3)
        neg_batch = sampler.get_neg_batch(users.reshape(-1))

        cnt_heavy = (neg_batch == 4).sum().item()
        cnt_lignt = (neg_batch == 0).sum().item()
        self.assertGreaterEqual(cnt_heavy, cnt_lignt)


if __name__ == "__main__":
    unittest.main()
