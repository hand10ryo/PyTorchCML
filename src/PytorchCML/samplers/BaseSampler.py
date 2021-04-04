from typing import Optional

import numpy as np
import torch
from torch import nn


class BaseSampler:
    def __init__(self, train_set: np.ndarray,
                 pos_weight: Optional[np.ndarray] = None,
                 neg_weight: Optional[np.ndarray] = None,
                 device: Optional[torch.device] = None,
                 batch_size: int = 256, n_neg_samples: int = 10):

        self.train_set = torch.LongTensor(train_set).to(device)
        self.n_neg_samples = n_neg_samples
        self.batch_size = batch_size
        self.n_user = np.unique(train_set[:, 0]).shape[0]
        self.n_item = np.unique(train_set[:, 1]).shape[0]
        self.device = device
        self.two_stage = False

        # device
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # set pos weight
        if pos_weight is not None:  # weighted
            if len(pos_weight) == len(train_set):
                pos_weight_pair = pos_weight

            elif len(pos_weight) == self.n_item:
                pos_weight_pair = pos_weight[train_set[:, 1]]

            elif len(pos_weight) == self.n_user:
                pos_weight_pair = pos_weight[train_set[:, 0]]

            else:
                raise NotImplementedError

        else:  # uniform
            pos_weight_pair = torch.ones(train_set.shape[0])

        self.pos_sampler = torch.distributions.categorical.Categorical(
            probs=torch.Tensor(pos_weight_pair).to(device))

        # set neg weight
        if neg_weight is None:  # uniorm
            neg_weight_item = np.ones(self.n_item) / self.n_item
        elif len(neg_weight) == self.n_item:  # weighted
            neg_weight_item = neg_weight
        else:
            raise NotImplementedError

        self.neg_sampler = torch.distributions.categorical.Categorical(
            probs=torch.Tensor(neg_weight_item).to(device))

    def get_pos_batch(self) -> torch.Tensor:
        batch_indices = self.pos_sampler.sample([self.batch_size])
        batch = self.train_set[batch_indices]
        return batch

    def get_neg_batch(self) -> torch.Tensor:
        return self.neg_sampler.sample([self.batch_size, self.n_neg_samples])
