from typing import Optional

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from scipy.sparse import csr_matrix


class BaseSampler:
    def __init__(
        self,
        train_set: np.ndarray,
        n_user: Optional[int] = None,
        n_item: Optional[int] = None,
        pos_weight: Optional[np.ndarray] = None,
        neg_weight: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 256,
        n_neg_samples: int = 10,
        strict_negative: bool = False,
    ):
        """Class of Base Sampler for get positive and negative batch.
        Args:
            train_set (np.ndarray): [description]
            n_user (Optional[int], optional): [description]. Defaults to None.
            n_item (Optional[int], optional): [description]. Defaults to None.
            pos_weight (Optional[np.ndarray], optional): [description]. Defaults to None.
            neg_weight (Optional[np.ndarray], optional): [description]. Defaults to None.
            device (Optional[torch.device], optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 256.
            n_neg_samples (int, optional): [description]. Defaults to 10.
            strict_negative (bool, optional): [description]. Defaults to False.

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]
        """

        self.train_set = torch.LongTensor(train_set).to(device)
        self.train_matrix = csr_matrix(
            (np.ones(train_set.shape[0]), (train_set[:, 0], train_set[:, 1])),
            [n_user, n_item],
        )
        self.n_neg_samples = n_neg_samples
        self.batch_size = batch_size
        if n_user is None:
            self.n_user = np.unique(train_set[:, 0]).shape[0]
        else:
            self.n_user = n_user

        if n_user is None:
            self.n_item = np.unique(train_set[:, 1]).shape[0]
        else:
            self.n_item = n_item
        self.device = device
        self.strict_negative = strict_negative
        self.two_stage = False

        # device
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.pos_weight_pair = torch.Tensor(pos_weight_pair).to(device)
        self.pos_sampler = Categorical(probs=self.pos_weight_pair)

        # set neg weight
        if neg_weight is None:  # uniorm
            neg_weight_item = np.ones(self.n_item)
        elif len(neg_weight) == self.n_item:  # weighted
            neg_weight_item = neg_weight
        else:
            raise NotImplementedError

        self.neg_weight_item = torch.Tensor(neg_weight_item).to(device)
        self.neg_sampler = Categorical(probs=self.neg_weight_item)

    def get_pos_batch(self) -> torch.Tensor:
        """Method for positive sampling.

        Returns:
            torch.Tensor: positive batch.
        """
        batch_indices = self.pos_sampler.sample([self.batch_size])
        batch = self.train_set[batch_indices]
        return batch

    def get_neg_batch(self, users: torch.Tensor) -> torch.Tensor:
        """Method of negative sampling

        Args:
            users (torch.Tensor): indices of users in pos pairs.

        Returns:
            torch.Tensor: negative samples.
        """

        if self.strict_negative:
            pos_item_mask = torch.Tensor(self.train_matrix[users.to("cpu")].A)
            weight = torch.einsum(
                "i,ni->ni", self.neg_weight_item, 1 - pos_item_mask.to(self.device)
            )
            neg_sampler = Categorical(probs=weight)
            neg_samples = neg_sampler.sample([self.n_neg_samples]).T
            return neg_samples

        else:
            return self.neg_sampler.sample([self.batch_size, self.n_neg_samples])
