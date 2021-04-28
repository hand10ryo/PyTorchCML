from typing import Optional, Union

from torch import optim

from ..evaluators import BaseEvaluator
from ..losses import BasePairwiseLoss, BaseTripletLoss
from ..models import BaseEmbeddingModel
from ..samplers import BaseSampler


class BaseTrainer:
    """Class of abstract trainer for redommend system in implicit feedback setting."""

    def __init__(
        self,
        model: BaseEmbeddingModel,
        optimizer: optim,
        criterion: Union[BasePairwiseLoss, BaseTripletLoss],
        sampler: BaseSampler,
    ):
        """Set components for learning recommend system.

        Args:
            model (BaseEmbeddingModel): embedding model
            optimizer (optim): pytorch optimizer
            criterion (Union[BasePairwiseLoss, BaseTripletLoss]): loss function
            sampler (BaseSampler): sampler
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.sampler = sampler

    def fit(
        self,
        n_batch: int = 500,
        n_epoch: int = 10,
        valid_evaluator: Optional[BaseEvaluator] = None,
        valid_per_epoch: int = 5,
    ):

        raise NotImplementedError
