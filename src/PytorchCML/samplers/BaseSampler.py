from typing import Optional, Union

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from scipy.sparse import csr_matrix

from ..models import BaseEmbeddingModel


class BaseSampler:
    def __init__(
        self,
        train_set: torch.Tensor,
        n_user: Optional[int] = None,
        n_item: Optional[int] = None,
        pos_weight: Optional[np.ndarray] = None,
        neg_weight: Union[np.ndarray, BaseEmbeddingModel] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 256,
        n_neg_samples: int = 10,
        strict_negative: bool = False,
        neutral: torch.Tensor = torch.Tensor([]),
    ):
        """Class of Base Sampler for get positive and negative batch.
        Args:
            train_set (torch.Tensor): training interaction data which columns are [user_id, item_id]
            n_user (Optional[int], optional): A number of user considered. Defaults to None.
            n_item (Optional[int], optional): A number of item considered. Defaults to None.
            pos_weight (Optional[np.ndarray], optional): Sampling weight for positive pair. Defaults to None.
            neg_weight (Optional[np.ndarray], optional): Sampling weight for negative item. Defaults to None.
            device (Optional[torch.device], optional): Device name. Defaults to None.
            batch_size (int, optional): Length of mini-batch. Defaults to 256.
            n_neg_samples (int, optional): A number of negative samples. Defaults to 10.
            strict_negative (bool, optional): If removing positive items from negative samples or not. Defaults to False.

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]
        """
        # set positive pairs
        self.train_set = train_set

        # set some hyperparameters
        self.n_neg_samples = n_neg_samples
        self.batch_size = batch_size
        self.strict_negative = strict_negative
        self.two_stage = False

        if n_user is None:
            self.n_user = np.unique(train_set[:, 0].cpu()).shape[0]
        else:
            self.n_user = n_user

        if n_user is None:
            self.n_item = np.unique(train_set[:, 1].cpu()).shape[0]
        else:
            self.n_item = n_item

        # set flag matrix whose element indicates the pair is not negative.
        train_set_cpu = train_set.cpu()
        neutral_cpu = neutral.cpu()
        not_negative = torch.cat([train_set_cpu, neutral_cpu])
        self.not_negative_flag = csr_matrix(
            (np.ones(not_negative.shape[0]), (not_negative[:, 0], not_negative[:, 1])),
            [n_user, n_item],
        )
        self.not_negative_flag.sum_duplicates()
        self.not_negative_flag.data[:] = 1

        # device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # set pos weight
        if pos_weight is not None:  # weighted
            if len(pos_weight) == len(train_set):
                pos_weight_pair = pos_weight

            elif len(pos_weight) == self.n_item:
                pos_weight_pair = pos_weight[train_set[:, 1].cpu()]

            elif len(pos_weight) == self.n_user:
                pos_weight_pair = pos_weight[train_set[:, 0].cpu()]

            else:
                raise NotImplementedError

        else:  # uniform
            pos_weight_pair = torch.ones(train_set.shape[0])

        self.pos_weight_pair = torch.Tensor(pos_weight_pair).to(self.device)
        self.pos_sampler = Categorical(probs=self.pos_weight_pair)

        # set neg weight
        if neg_weight is None:  # uniform
            self.negative_weighted_by_model = False
            self.neg_item_weight = torch.ones(self.n_item).to(self.device)

        elif isinstance(neg_weight, BaseEmbeddingModel):  # user-item weighted
            self.negative_weighted_by_model = True
            self.neg_weight_model = neg_weight

        elif len(neg_weight) == self.n_item:  # item weighted
            self.negative_weighted_by_model = False
            self.neg_item_weight = torch.Tensor(neg_weight).to(self.device)

        else:
            raise NotImplementedError

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

        if self.negative_weighted_by_model and self.strict_negative:
            flag = torch.Tensor(self.not_negative_flag[users.to("cpu")].A)
            mask = 1 - flag.to(self.device)
            weight = self.neg_weight_model.get_item_weight(users)
            weight *= mask
            neg_sampler = Categorical(probs=weight)
            neg_samples = neg_sampler.sample([self.n_neg_samples]).T

        elif self.negative_weighted_by_model and not self.strict_negative:
            weight = self.neg_weight_model.get_item_weight(users)
            neg_sampler = Categorical(probs=weight)
            neg_samples = neg_sampler.sample([self.n_neg_samples]).T

        elif not self.negative_weighted_by_model and self.strict_negative:
            flag = torch.Tensor(self.not_negative_flag[users.to("cpu")].A)
            mask = 1 - flag.to(self.device)
            weight = mask * self.neg_item_weight
            neg_sampler = Categorical(probs=weight)
            neg_samples = neg_sampler.sample([self.n_neg_samples]).T

        else:
            neg_sampler = Categorical(probs=self.neg_item_weight)
            neg_samples = neg_sampler.sample([self.batch_size, self.n_neg_samples])

        return neg_samples
