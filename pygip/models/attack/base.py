from abc import ABC, abstractmethod
from typing import Union, Optional

import torch

from pygip.datasets import Dataset
from pygip.utils.hardware import get_device


class BaseAttack(ABC):
    """Abstract base class for attack models.

    This class provides a common interface for various attack strategies on graph-based
    machine learning models. It handles device management, dataset loading, and
    compatibility checks to ensure that the attack can be executed on the given
    dataset and model API type.

    Attributes:
        supported_api_types (set): A set of strings representing the supported API
            types (e.g., 'pyg', 'dgl').
        supported_datasets (set): A set of strings representing the names of
            supported dataset classes.
        device (torch.device): The computing device (CPU or GPU) to be used for
            the attack.
        dataset (Dataset): The dataset object containing graph data and metadata.
        graph_dataset: The raw graph dataset from the underlying library.
        graph_data: The primary graph data structure.
        num_nodes (int): The number of nodes in the graph.
        num_features (int): The number of features per node.
        num_classes (int): The number of classes for node classification.
        attack_node_fraction (float, optional): The fraction of nodes to be
            targeted by the attack.
        model_path (str, optional): The path to a pre-trained target model.
    """
    supported_api_types = set()
    supported_datasets = set()

    def __init__(self, dataset: Dataset, attack_node_fraction: float = None, model_path: str = None,
                 device: Optional[Union[str, torch.device]] = None):
        """Initializes the BaseAttack.

        Args:
            dataset (Dataset): The dataset to be attacked.
            attack_node_fraction (float, optional): The fraction of nodes to
                target in the attack. Defaults to None.
            model_path (str, optional): The path to a pre-trained model file.
                Defaults to None.
            device (Union[str, torch.device], optional): The device to run the
                attack on. If None, it will be automatically selected.
                Defaults to None.
        """
        self.device = torch.device(device) if device else get_device()
        print(f"Using device: {self.device}")

        # graph data
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data

        # meta data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # params
        self.attack_node_fraction = attack_node_fraction
        self.model_path = model_path

        self._check_dataset_compatibility()

    def _check_dataset_compatibility(self):
        """Checks if the dataset is compatible with the attack.

        Raises:
            ValueError: If the dataset's API type or class name is not in the
                list of supported types.
        """
        cls_name = self.dataset.__class__.__name__

        if self.supported_api_types and self.dataset.api_type not in self.supported_api_types:
            raise ValueError(
                f"API type '{self.dataset.api_type}' is not supported. Supported: {self.supported_api_types}")

        if self.supported_datasets and cls_name not in self.supported_datasets:
            raise ValueError(f"Dataset '{cls_name}' is not supported. Supported: {self.supported_datasets}")

    @abstractmethod
    def attack(self):
        """
        Execute the attack.
        """
        raise NotImplementedError

    def _load_model(self, model_path):
        """
        Load a pre-trained model.
        """
        raise NotImplementedError

    def _train_target_model(self):
        """
        Train the target model if not provided.
        """
        raise NotImplementedError

    def _train_attack_model(self):
        """
        Train the attack model.
        """
        raise NotImplementedError
