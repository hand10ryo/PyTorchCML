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
        column_names: Optional[dict] = None,
    ):
        """Set components for learning recommend system.

        Args:
            model (BaseEmbeddingModel): embedding model
            optimizer (optim): pytorch optimizer
            criterion (Union[BasePairwiseLoss, BaseTripletLoss]): loss function
            sampler (BaseSampler): sampler
            column_names (Optional[dict]): sampler
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.sampler = sampler

        if column_names is not None:
            self.column_names = column_names
        else:
            self.column_names = {"user_id": 0, "item_id": 1}

    def fit(
        self,
        n_batch: int = 500,
        n_epoch: int = 10,
        valid_evaluator: Optional[BaseEvaluator] = None,
        valid_per_epoch: int = 5,
    ):

        raise NotImplementedError
