from typing import Optional

import numpy as np
import torch

from torch.distributions.categorical import Categorical

from .BaseSampler import BaseSampler


class TwoStageSampler(BaseSampler):
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
        n_neg_candidates=200,
    ):
        """Class of Two Stage Sampler for CML.

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
            n_neg_candidates (int, optional): A number of candidates in 1st stage negative sampling. Defaults to 200.
        """
        super().__init__(
            train_set,
            n_user,
            n_item,
            pos_weight,
            neg_weight,
            device,
            batch_size,
            n_neg_samples,
            strict_negative,
        )

        self.two_stage = True
        self.n_neg_candidates = n_neg_candidates
        self.neg_candidate_sampler = Categorical(probs=self.neg_item_weight)

    def get_pos_batch(self) -> torch.Tensor:
        """Method for positive sampling.

        Returns:
            torch.Tensor: positive batch.
        """
        batch_indices = self.pos_sampler.sample([self.batch_size])
        batch = self.train_set[batch_indices]
        return batch

    def get_and_set_candidates(self) -> torch.Tensor:
        """Method of getting and setting candidates for 2nd stage.

        Returns:
            torch.Tensor: Indices of items of negative sample candidate.
        """
        self.candidates = self.neg_candidate_sampler.sample([self.n_neg_candidates])
        return self.candidates

    def set_candidates_weight(self, dist: torch.Tensor, dim: int):
        """Method of calclating sampling weight for 2nd stage sampling.

        Args:
            dist (torch.Tensor) : spreadout distance (dot product) matrix, size = (n_batch, n_neg_candidates)
            dim (int): A number of dimention of embeddings.
        """
        # draw beta
        beta = (
            torch.distributions.beta.Beta((dim - 1) / 2, 1 / 2)
            .sample([1])
            .to(self.device)
        )

        # make mask
        mask = (dist > 0) * (dist < 0.99)

        # calc weight
        alpha = 1 - (dim - 1) / 2
        log_neg_dist = torch.log(1 - torch.square(dist))  # + 1e-6)
        log_beta = torch.log(beta)
        self.candidates_weight = torch.exp(alpha * log_neg_dist + log_beta)

        # fill zero by mask
        self.candidates_weight[~mask] = 0

        # all zero -> uniform
        self.candidates_weight[self.candidates_weight.sum(axis=1) == 0] = 1

    def get_neg_batch(self, users: torch.Tensor) -> torch.Tensor:
        """Method of negative sampling

        Args:
            users (torch.Tensor): indices of users in pos pairs.

        Returns:
            torch.Tensor: negative samples.
        """

        if self.strict_negative:
            pos_item_mask = torch.Tensor(self.not_negative_flag[users.to("cpu")].A)
            pos_item_mask_candidate = pos_item_mask[:, self.candidates].to(self.device)
            weight = (1 - pos_item_mask_candidate) * self.candidates_weight
            zero_indices = weight.sum(axis=1) <= 1e-10
            weight[zero_indices.reshape(-1)] = 1

        else:
            weight = self.candidates_weight

        neg_sampler = Categorical(probs=weight)
        neg_indices = neg_sampler.sample([self.n_neg_samples]).T
        neg_items = self.candidates[neg_indices]

        return neg_items
