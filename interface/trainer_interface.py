from abc import ABC, abstractmethod


class MerlinArthurInterface(ABC):
    """This serves as the blueprint for defining the MerlinArthurTrainer class."""

    @abstractmethod
    def __init__(self, arguments):
        raise NotImplementedError

    @abstractmethod
    def prepare_dataset(self, arguments):
        raise NotImplementedError

    @abstractmethod
    def initialize_arthur(self, arthur):
        raise NotImplementedError

    @abstractmethod
    def initialize_feature_selectors(self, feaute_selectors):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self, lr_arthur, lr_merlin, lr_morgana):
        raise NotImplementedError

    @abstractmethod
    def regular_train(self, max_epochs):
        raise NotImplementedError

    @abstractmethod
    def train_min_max(self):
        raise NotImplementedError

    @abstractmethod
    def _train_min_max_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _test_min_max_epoch(self):
        raise NotImplementedError
