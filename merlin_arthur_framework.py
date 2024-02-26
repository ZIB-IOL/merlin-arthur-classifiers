import itertools
from datetime import date
from types import ModuleType
from typing import List, Mapping, Optional, Tuple

import numpy as np
import torch
from captum.attr import (GradientShap, GuidedBackprop, GuidedGradCam,
                         IntegratedGradients, Saliency)
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm

import wandb
from config_files.config_dataclass import *
from custom_datasets.datasets import (CustomMNIST, CustomUCICensus,
                                      SVHNDataset, read_census_data)
from feature_selection import FeatureSelector
from interface.trainer_interface import MerlinArthurInterface
from models import (DeeperCNN, SimpleCNN, SimpleNet, UCICensusClassifier,
                    saliency_model)
from utilities.metrics import (Averager, EntropyCalculator, IoUCalculator,
                               MaskRegularizer, MetricsCollector,
                               MorganaCriterion, PointingGameCalculator,
                               RMACalculator, RRACalculator, SaliencyLoss,
                               categorical_accuracy, get_accuracy)
from utilities.util import SaveBestModel  # type: ignore


class MerlinArthurTrainer:
    def __init__(
        self, trainer_cfg: TrainerConfig, boolean_cfg: BooleanConfig, logger: Optional[ModuleType] = None
    ) -> None:
        """Initializes MerlinArthurTrainer.

        Args:
            trainer_config (TrainerConfig): TrainerConfig object containing all the parameters.
            boolean_config (BooleanConfig): BooleanConfig object containing all the boolean parameters.
            logger (Optional[ModuleType]): Logger to use. Defaults to None.

            These config classes include:
                approach (str): Approach to use, either `sfw`, ``unet`, `regular`
                save_model (Optional[bool], optional): Whether to save the model. Defaults to False.
                save_masks (Optional[bool], optional): Whether to save the masks. Defaults to True.
                debug (Optional[bool], optional): Whether to run in debug mode. Defaults to False.
                add_mask_channel (Optional[bool], optional): Whether to add mask channel to the input. Defaults to False.
                binary_classification (Optional[bool], optional): Whether to perform binary classification. Defaults to False.
                other_loss (Optional[bool], optional): Whether to use other loss. Defaults to False.
                destroyer_loss_active (Optional[bool], optional): Whether to use destroyer loss. Defaults to True.
                use_grayscale (Optional[bool], optional): Whether to use grayscale. Defaults to None.
                calculate_iou (Optional[bool], optional): Whether to calculate IoU. Defaults to False.

        Raises:
            AssertionError: If approach is not supported
            AssertionError: If logger is not of type SummaryWriter, `wandb` or None
            AssertionError: If save_best_model is not of type bool
        """
        self.trainer_cfg = trainer_cfg
        self.logger = logger
        # Check if approach is supported
        approaches = ["sfw", "unet", "regular", "mask_optimization", "posthoc"]
        assert (
            trainer_cfg.approach in approaches
        ), f"Approach must be one of {approaches}, got approach='{trainer_cfg.approach}'"
        assert isinstance(logger, (ModuleType, type(None))), "Logger must be of type `wandb`, or None"

        # Assign trainer config parameters
        self.approach = trainer_cfg.approach
        self.debug = trainer_cfg.debug
        self.epochs = trainer_cfg.epochs
        self.save_masks = trainer_cfg.save_masks
        self.add_mask_channel = trainer_cfg.add_mask_channel
        # Assing boolean config parameters
        self.save_model = boolean_cfg.save_model
        self.use_amp = boolean_cfg.use_amp
        self.cuda_benchmark = boolean_cfg.cuda_benchmark
        # Initialize parameters
        self.batch_size = self.gamma = None
        self.arthur = self.merlin = self.morgana = None
        self.scheduler_arthur = self.scheduler_merlin = self.scheduler_morgana = None
        self.optimizer_arthur = self.optimizer_merlin = self.optimizer_morgana = None
        self.weight_decay_merlin = self.weight_decay_morgana = 0
        self.data_type = None

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on Device:", self.device)

        today = date.today()
        self.current_date = today.strftime("%b-%d-%Y")

    def _load_mnist(self) -> None:
        """Loads MNIST dataset."""
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if self.approach in ("sfw", "mask_optimization"):
            mnist_dataset = CustomMNIST
        elif self.approach in ("unet", "regular"):
            mnist_dataset = MNIST
        else:
            raise NotImplementedError(f"Approach `{self.approach}` not implemented for MNIST.")
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_data = mnist_dataset(root=".data/", train=True, download=True, transform=self.transforms)
        self.test_data = mnist_dataset(root=".data/", train=False, download=True, transform=self.transforms)
        self.data_type = "image"

        self.train_data, self.test_data = self._reduce_dataset_to_targets(
            self.train_data, self.test_data, self.list_of_targets
        )
        if self.debug:
            self.train_data.data = self.train_data.data[: 2 * self.batch_size]  # type: ignore
            self.test_data.data = self.test_data.data[: 2 * self.batch_size]  # type: ignore

    def _load_svhn(self) -> None:
        """Loads SVHN dataset."""
        assert self.batch_size is not None, "Batch size must be specified for SVHN."

        # Transformations
        target_transform = transforms.ToTensor()
        transforms_train = transforms.Compose(
            [
                transforms.Resize((self.resized_height, self.resized_width)),
                transforms.ToTensor(),
            ]
        )
        transforms_test = transforms.Compose(
            [
                transforms.Resize((self.resized_height, self.resized_width)),
                transforms.ToTensor(),
            ]
        )

        if self.svhn_autoaugment is True:
            # AutoAugment is only applied to training data
            transforms_train.transforms.insert(1, transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN))  # type: ignore

        if self.add_normalization is True:
            if self.use_grayscale is True:
                transforms_train.transforms.append(transforms.Normalize((0.5,), (0.5,)))  # type: ignore
                transforms_test.transforms.append(transforms.Normalize((0.5,), (0.5,)))  # type: ignore
            else:
                transforms_train.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # type: ignore
                transforms_test.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # type: ignore

        self.train_data = SVHNDataset(
            ".data/svhn_format1/",
            split="train",
            target_digit=self.svhn_cfg.svhn_target,
            transform=transforms_train,
            target_transform=target_transform,
            balance=True,
            max_samples=2 * self.batch_size if self.debug is True else None,
            one_vs_two=self.one_vs_two,
            use_grayscale=self.use_grayscale,
        )
        self.test_data = SVHNDataset(
            ".data/svhn_format1/",
            split="test",
            target_digit=self.svhn_target,
            transform=transforms_test,
            target_transform=target_transform,
            balance=True,
            max_samples=2 * self.batch_size if self.debug else None,
            one_vs_two=self.one_vs_two,
            use_grayscale=self.use_grayscale,
        )

        if self.add_extra_data is True:
            # Add extra data to training data from "extra" split
            train_data_extra = SVHNDataset(
                ".data/svhn_format1/",
                split="extra",
                target_digit=self.svhn_target,
                transform=transforms_train,
                target_transform=target_transform,
                balance=True,
                max_samples=20000 if self.extra_data_max_samples is None else self.extra_data_max_samples,
                one_vs_two=self.one_vs_two,
                use_grayscale=self.use_grayscale,
            )
            self.train_data = ConcatDataset([self.train_data, train_data_extra])

        self.data_type = "image"

    def _load_uci_census(self) -> None:
        """Loads UCI Census dataset."""
        # assert "read_pre_processed" in kwargs, f"read_pre_processed must be specified for UCI-Census"
        # assert "remove_corr_feat" in kwargs, f"remove_corr_feat must be specified for UCI-Census"
        # assert "download" in kwargs, f"download must be specified for UCI-Census"
        # assert "drop_sex_feat" in kwargs, f"drop_sex_feat must be specified for UCI-Census"
        # assert self.batch_size is not None, "Batch size must be specified for UCI-Census."

        target_class = "income" if ">50K" in self.list_of_targets else "sex_target"
        self.train_data, self.test_data = read_census_data(
            PATH=".data/adult",
            target_class=target_class, # either "income" or `sex_target`
            read_pre_processed=False, # Set to True if you want to use the pre-processed data (much faster!)
            remove_corr_feat=True, # Set to True if you want to remove correlated features (e.g., marital status)
            download=False, # Set to True if you want to download the data
            drop_sex_feat=False, # Set to True if you want to drop the feature `sex`
        )
        if self.debug:
            self.train_data = self.train_data[: 2 * self.batch_size]
            self.test_data = self.test_data[: 2 * self.batch_size]

        self.train_data = CustomUCICensus(dataset=self.train_data, target_class=target_class)
        self.test_data = CustomUCICensus(dataset=self.test_data, target_class=target_class)
        self.data_type = "categorical"

        # raise NotImplementedError("UCI Census has been removed from this implementation.")

    def prepare_dataset(self, dataset_cfg: DatasetConfig, svhn_cfg: SvhnConfig, **kwargs) -> None:
        """Initializes Dataset with according preprocessing steps"""
        self.dataset_cfg = dataset_cfg
        self.svhn_cfg = svhn_cfg

        self.batch_size = self.dataset_cfg.batch_size
        self.list_of_targets = self.dataset_cfg.targets
        self.num_targets = len(self.list_of_targets)  # excluding "Don't Know!"
        self.string_of_targets = "".join(str(target) for target in self.list_of_targets)
        self.dataset = self.dataset_cfg.dataset
        self.add_normalization = self.dataset_cfg.add_normalization
        self.original_image_ratio = self.dataset_cfg.original_image_ratio

        self.use_grayscale = self.svhn_cfg.use_grayscale
        self.svhn_autoaugment = self.svhn_cfg.svhn_autoaugment
        self.resized_height = self.svhn_cfg.resized_height
        self.resized_width = self.svhn_cfg.resized_width
        self.add_extra_data = self.svhn_cfg.add_extra_data
        self.extra_data_max_samples = self.svhn_cfg.extra_data_max_samples
        self.svhn_target = self.svhn_cfg.svhn_target
        self.one_vs_two = self.svhn_cfg.one_vs_two

        # fmt: off
        # Check if dataset args are supported
        allowed_datasets = ("MNIST", "UCI-Census", "SVHN")
        assert (self.dataset in allowed_datasets
        ), f"MerlinArthurTrainer is only implemented for {allowed_datasets}, got dataset='{self.dataset}'" 
        assert (self.num_targets == 2), f"MerlinArthurTrainer is only implemented for 2 targets, got num_targets={self.num_targets}"
        assert self.batch_size > 0, f"Batch size must be greater than 0, got batch_size={self.batch_size}"
        # fmt: on

        # Initialize Dataset and Dataloader
        if self.dataset == "MNIST":
            self._load_mnist()
        elif self.dataset == "SVHN":
            self._load_svhn()
        elif self.dataset == "UCI-Census":
            self._load_uci_census()
        else:
            raise NotImplementedError(f"Dataset `{self.dataset}` not implemented.")

        # Initialize Dataloader for training and testing
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def _reduce_dataset_to_targets(
        self, train_data: Dataset, test_data: Dataset, list_of_targets: List[str]
    ) -> Tuple[Dataset, Dataset]:
        """Reduces dataset to datapoints which correspond to targets of interest.

        Only tested for MNIST so far. Should not be applied to other datasets or len(list_of_targets) != 2.

        Iterates through chosen labels and redefines the dataset containing only the datapoints corresponding to the given targets.
        More specifically, relabeling the reduced dataset with incremental labels starting from 0.

        Args:
            train_data (Dataset): Training Dataset.
            test_data (Dataset): Test Dataset
            list_of_targets (List[str]): List of target labels as strings.

        Returns:
            Tuple[Dataset, Dataset]: Dataset containing only images of corresponding labels.
        """
        list_of_targets = sorted(list_of_targets)
        train_idxs = torch.zeros_like(train_data.targets)  # type: ignore
        test_idxs = torch.zeros_like(test_data.targets)  # type: ignore
        label_count = 0
        for label in list_of_targets:
            label_index = int(label)
            train_idxs = torch.logical_or(train_idxs, train_data.targets == label_index)  # type: ignore
            test_idxs = torch.logical_or(test_idxs, test_data.targets == label_index)  # type: ignore
            train_data.targets[train_data.targets == label_index] = label_count  # type: ignore
            test_data.targets[test_data.targets == label_index] = label_count  # type: ignore
            label_count += 1
        train_data.data = train_data.data[train_idxs]  # type: ignore
        train_data.targets = train_data.targets[train_idxs]  # type: ignore
        test_data.data = test_data.data[test_idxs]  # type: ignore
        test_data.targets = test_data.targets[test_idxs]  # type: ignore

        return train_data, test_data

    def initialize_arthur(self, arthur_cfg: ArthurConfig) -> None:
        """Initializes Arthur model.

        Check config_dataclass.py for more information on the config parameters.

        Args:
            arthur_cfg (ArthurConfig): Config for Arthur model.
        """
        self.arthur_cfg = arthur_cfg
        self.model_arthur = arthur_cfg.model_arthur
        self.imagenet_pretrained = arthur_cfg.imagenet_pretrained
        self.reduce_kernel_size = arthur_cfg.reduce_kernel_size
        self.binary_classification = arthur_cfg.binary_classification
        self.lr = arthur_cfg.lr
        self.weight_decay = arthur_cfg.weight_decay
        self.pretrained_arthur = arthur_cfg.pretrained_arthur
        self.pretrained_path = arthur_cfg.pretrained_path
        self.hidden_size = arthur_cfg.hidden_size
        self.num_layers = arthur_cfg.num_layers
        self.scheduler_arthur_step_size = arthur_cfg.scheduler_arthur_step_size
        self.scheduler_arthur = arthur_cfg.scheduler_arthur

        in_channels = 3 if self.dataset == "SVHN" else 1
        output_dim = 1 if self.binary_classification else 3

        if self.model_arthur == "SimpleCNN":
            self.arthur = SimpleCNN(output_dim, in_channels).to(self.device)
        elif self.model_arthur == "DeeperCNN":
            self.arthur = DeeperCNN(output_dim, in_channels).to(self.device)
        elif self.model_arthur == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if self.imagenet_pretrained else None
            self.arthur = models.resnet18(weights)
            self._update_kernel_size_and_fc(output_dim, 512)
        elif self.model_arthur == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT if self.imagenet_pretrained is True else None
            self.arthur = models.wide_resnet50_2(weights)
            self._update_kernel_size_and_fc(output_dim, 2048)
        elif self.model_arthur == "UCICensusClassifier":
            self.arthur = UCICensusClassifier(
                input_size=11 * 41, output_size=output_dim, num_layers=3, hidden_size=50  # type: ignore
            )
        elif self.model_arthur == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if self.imagenet_pretrained is True else None
            self.arthur = models.vgg16(weights)
            # Update output dimension to 1 for binary classification
            in_features = self.arthur.classifier[6].in_features
            self.arthur.classifier[6] = torch.nn.Linear(in_features, output_dim)  # type: ignore
        else:
            raise NotImplementedError(f"Arthur model `{self.model_arthur}` not implemented.")

        if self.pretrained_arthur is True:
            assert self.pretrained_path is not None, "Path to pretrained Arthur model must be provided."
            state_dict = torch.load(self.pretrained_path)
            self.arthur.load_state_dict(state_dict["model_state_dict"])
        self.arthur = self.arthur.to(self.device)
        self.num_param = sum(p.numel() for p in self.arthur.parameters() if p.requires_grad)

    def _update_kernel_size_and_fc(self, output_dim: int, fc_out_features: int) -> None:
        """Private method to update kernel size and fully connected layer of ResNet model.

        Args:
            output_dim (int): The output dimension of the model.
            fc_out_features (int): The output features of the fully connected layer.

        Returns:
            None
        """
        if self.reduce_kernel_size is True:
            if self.add_mask_channel and not self.use_grayscale:
                conv1_in_channels = 4
            elif not self.use_grayscale:
                conv1_in_channels = 3
            else:
                conv1_in_channels = 1
            self.arthur.conv1 = torch.nn.Conv2d(conv1_in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)  # type: ignore
        self.arthur.fc = torch.nn.Linear(fc_out_features, output_dim)  # type: ignore

    def configure_arthur_optimizer(self) -> None:
        """Initializes optimizer for Arthur."""
        assert self.arthur is not None, "Arthur model must be initialized first."
        # Initialize optimizer and scheduler
        self.optimizer_arthur = torch.optim.Adam(self.arthur.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_arthur is True and self.scheduler_arthur_step_size is not None:
            self.scheduler_arthur = torch.optim.lr_scheduler.StepLR(
                self.optimizer_arthur, step_size=self.scheduler_arthur_step_size, gamma=0.1
            )
        else:
            self.scheduler_arthur = None

    def initialize_feature_selectors(
        self,
        feature_selector_cfg: FeatureSelectorConfig,
        mask_optim_cfg: Optional[MaskOptimizationConfig],
        unet_cfg: Optional[UnetConfig],
        metrics_and_penalties_cfg: MetricsPenaltiesConfig,
        **kwargs,
    ) -> None:
        """Initializes feature selectors Merlin and Morgana. Either U-Net or SFW can be used.

        Args:
            feature_selector_cfg (FeatureSelectorConfig): Config for feature selectors.

            Args in feature_selector_cfg:
                segmentation_method: Optional[str] = None
                mask_size: Optional[int] = None
                lr_merlin: Optional[float] = None
                lr_morgana: Optional[float] = None
                gamma: Optional[float] = None
                gaussian: bool = False
                only_on_class: bool = False
        """
        # Assing configs
        self.fs_config = feature_selector_cfg
        self.mask_optim_cfg = mask_optim_cfg
        self.unet_cfg = unet_cfg
        self.metrics_and_penalties_cfg = metrics_and_penalties_cfg
        assert self.unet_cfg is not None
        # Assign values of feature selector config
        self.segmentation_method = feature_selector_cfg.segmentation_method  # e.g., "topk"
        self.mask_size = feature_selector_cfg.mask_size
        self.lr_merlin = feature_selector_cfg.lr_merlin
        self.lr_morgana = feature_selector_cfg.lr_morgana
        self.gamma = feature_selector_cfg.gamma
        self.gaussian = feature_selector_cfg.gaussian
        self.only_on_class = feature_selector_cfg.only_on_class
        # Assign values of metrics and penalties config
        self.other_loss = metrics_and_penalties_cfg.other_loss
        self.destroyer_loss_active = metrics_and_penalties_cfg.destroyer_loss_active
        self.calculate_iou = metrics_and_penalties_cfg.calculate_iou
        # Assing values of unet config
        self.saliency_mapper = self.unet_cfg.saliency_mapper
        self.steps_morgana = self.unet_cfg.steps_morgana

        # Initialize Merlin
        merlin_model = self._get_feature_selection_model()
        self.merlin = FeatureSelector(
            mode="merlin",
            model=merlin_model,
            use_amp=self.use_amp,
            binary_classification=self.binary_classification,
            add_mask_channel=self.add_mask_channel,
            fs_config=self.fs_config,
            mask_optim_cfg=self.mask_optim_cfg,
            unet_cfg=self.unet_cfg,
            metrics_and_penalties_cfg=self.metrics_and_penalties_cfg,
        )
        # Initialize Morgana
        if self.unet_cfg and self.unet_cfg.parameter_sharing:
            morgana_model = self.merlin.model
        else:
            morgana_model = self._get_feature_selection_model()
        self.morgana = FeatureSelector(
            mode="morgana",
            model=morgana_model,
            use_amp=self.use_amp,
            binary_classification=self.binary_classification,
            add_mask_channel=self.add_mask_channel,
            fs_config=self.fs_config,
            mask_optim_cfg=self.mask_optim_cfg,
            unet_cfg=self.unet_cfg,
            metrics_and_penalties_cfg=self.metrics_and_penalties_cfg,
        )
        # Set data type
        self.merlin.data_type = self.data_type  # type: ignore
        self.morgana.data_type = self.data_type  # type: ignore
        # Move to device
        self.merlin = self.merlin.to(self.device)
        self.morgana = self.morgana.to(self.device)
        # Check if mask sizes are equal
        assert self.merlin.mask_size == self.morgana.mask_size, "Mask sizes of Merlin and Morgana must be equal."
        self.mask_size = self.merlin.mask_size
        # Set segmentation method
        self.segmentation_method = self.segmentation_method
        # Set brute force mode (if applicable)
        self.brute_force_merlin = kwargs.get("brute_force_merlin", False)
        self.brute_force_morgana = kwargs.get("brute_force_morgana", False)

    def _get_feature_selection_model(self):
        """Private method to get feature selection model."""
        if self.approach in ("sfw", "mask_optimization", "posthoc"):
            return self.approach
        elif self.approach == "unet" and not self.saliency_mapper:
            return SimpleNet(
                n_channels=3 if self.dataset == "SVHN" else 1,
                apply_sigmoid=True if self.segmentation_method == "topk" else False,
            )
        elif self.approach == "unet" and self.saliency_mapper:
            return saliency_model()
        else:
            raise NotImplementedError(f"Model `{self.approach}` not implemented.")

    def configure_unet_optimizers(self, **kwargs):
        """Configures optimizers for each player, respectively."""
        assert self.merlin is not None, "Merlin must be initialized first."
        assert self.morgana is not None, "Morgana must be initialized first."
        assert self.lr_merlin is not None, "Learning rate for Merlin must be set."
        assert self.lr_morgana is not None, "Learning rate for Morgana must be set."
        assert self.weight_decay_merlin is not None, "Weight decay for Merlin must be set."
        assert self.weight_decay_morgana is not None, "Weight decay for Morgana must be set."

        # Set optimizers
        self.optimizer_merlin = torch.optim.Adam(
            self.merlin.parameters(), lr=self.lr_merlin, weight_decay=self.weight_decay_merlin
        )
        self.optimizer_morgana = torch.optim.Adam(
            self.morgana.parameters(),
            lr=self.lr_morgana,
            weight_decay=self.weight_decay_morgana,
            maximize=False if self.binary_classification is True else True,  # type: ignore
        )

    def calculate_conditional_entropy(self) -> MetricsCollector:
        """Calculates conditional entropy."""
        if self.approach in ("sfw", "mask_optimization"):
            self.fixed_loader = [(x, y, s_1, s_2, ind) for x, y, s_1, s_2, ind in self.test_loader]
            self.fixed_loader += [(x, y, s_1, s_2, ind) for x, y, s_1, s_2, ind in self.train_loader]
        elif self.approach == "unet":
            self.fixed_loader = [(x, y) for x, y in self.test_loader]
            self.fixed_loader += [(x, y) for x, y in self.train_loader]
        else:
            raise ValueError(f"Approach {self.approach} is not supported.")

        if self.arthur is None:
            raise ValueError("Arthur model is not initialized.")

        assert self.segmentation_method is not None, "Segmentation method must be set."
        assert self.mask_size is not None, "Mask size must be set."

        if self.approach == "unet":
            # Load pretrained state dict
            state_dict = torch.load(self.arthur_cfg.pretrained_path)
            # Remove 'model.' prefix from state_dict keys
            unet_state_dict = {k.replace('model.', ''): v for k, v in state_dict["model_state_dict_merlin"].items()}
            
            self.merlin.model.load_state_dict(unet_state_dict)

        self.print_trainer_details()
        self.entropy_calculator = EntropyCalculator(
            self.arthur,
            self.merlin,
            self.morgana,
            self.segmentation_method,
            self.fixed_loader,
            self.test_loader,
            self.mask_size,
            self.num_targets,
            brute_force_merlin=self.brute_force_merlin,
            brute_force_morgana=self.brute_force_morgana,
        )

        return self.entropy_calculator.calc_conditional_entropy(tol=1e-5)

    def print_trainer_details(self):
        """Prints all details of the model.

        Including the number of parameters, all optimization parameters, mask size, approach, segmentation method
        and all details nicely formatted.
        """
        if self.arthur is None:
            raise ValueError("Arthur model is not initialized.")
        print("")
        print("Model Details:")
        print("-------------")
        print("Number of parameters in Arthur: ", sum(p.numel() for p in self.arthur.parameters() if p.requires_grad))
        print("-------------")
        if isinstance(self.merlin, FeatureSelector) and isinstance(self.merlin.model, torch.nn.Module):
            print(
                "Number of parameters in Merlin: ",
                sum(p.numel() for p in self.merlin.model.parameters() if p.requires_grad),
            )
        elif isinstance(self.merlin, FeatureSelector) and self.merlin.sfw_configuration is True:
            # Print SFW configuration
            print("SFW Merlin Details:")
            print("-------------")
            print("SFW Learning Rate:", self.merlin.sfw_learning_rate)
            print("SFW Momentum:", self.merlin.sfw_momentum)
            print("SFW Max Iterations:", self.merlin.sfw_max_iterations)
            print("SFW Stoptol:", self.merlin.sfw_stoptol)
            print("-------------")

        if isinstance(self.morgana, FeatureSelector) and isinstance(self.morgana.model, torch.nn.Module):
            print(
                "Number of parameters in Morgana: ",
                sum(p.numel() for p in self.morgana.model.parameters() if p.requires_grad),
            )
            print("-------------")
        elif isinstance(self.morgana, FeatureSelector) and self.morgana.sfw_configuration is True:
            # Print SFW configuration
            print("SFW Morgana Details:")
            print("-------------")
            print("SFW Learning Rate:", self.morgana.sfw_learning_rate)
            print("SFW Momentum:", self.morgana.sfw_momentum)
            print("SFW Max Iterations:", self.morgana.sfw_max_iterations)
            print("SFW Stoptol:", self.morgana.sfw_stoptol)
            print("-------------")

        print("Approach: ", self.approach.upper())
        print("Number of Epochs: ", self.epochs) if isinstance(self.epochs, int) else None
        print("Gamma: ", self.gamma) if isinstance(self.gamma, float) else None
        print("Batch Size: ", self.batch_size)
        print("Number of Targets: ", self.num_targets)
        print("Target classes:", self.string_of_targets)
        print("Dataset:", self.dataset)

        if self.approach in ("sfw", "unet"):
            print("Mask size: ", self.mask_size)
            print("Segmentation method: ", self.segmentation_method)
            print("-------------")

        # print nicely some information about the optmizer and scheduler
        if self.optimizer_arthur is not None:
            print("Arthur optimizer: ", self.optimizer_arthur)
            if self.scheduler_arthur is not None:
                print("Arthur scheduler: ", self.scheduler_arthur.__class__)
                print("Arthur scheduler step size: ", self.scheduler_arthur.step_size)  # type: ignore
                print("Arthur scheduler gamma: ", self.scheduler_arthur.gamma)  # type: ignore
            print("-------------")

        if self.optimizer_merlin is not None:
            print("Merlin optimizer: ", self.optimizer_merlin)
            if self.scheduler_merlin is not None:
                print("Merlin scheduler: ", self.scheduler_merlin.__class__)
                print("Merlin scheduler step size: ", self.scheduler_merlin.step_size)  # type: ignore
                print("Merlin scheduler gamma: ", self.scheduler_merlin.gamma)  # type: ignore
            print("-------------")

        if self.optimizer_morgana is not None:
            print("Morgana optimizer: ", self.optimizer_morgana)
            if self.scheduler_morgana is not None:
                print("Morgana scheduler: ", self.scheduler_morgana.__class__)
                print("Morgana scheduler step size: ", self.scheduler_morgana.step_size)  # type: ignore
                print("Morgana scheduler gamma: ", self.scheduler_morgana.gamma)  # type: ignore
            print("-------------")
        print("")

    def regular_train(self, max_epochs: int = 10) -> bool:
        """Trains and tests Arthur without Merlin and Morgana."""
        assert self.arthur is not None, "Arthur model must be initialized."
        assert self.optimizer_arthur is not None, "Optimizer for Arthur must be initialized."
        self.print_trainer_details()
        self.max_epochs = max_epochs
        loss_criterion = (
            torch.nn.BCEWithLogitsLoss() if self.binary_classification is True else torch.nn.CrossEntropyLoss()
        )
        save_best_model = SaveBestModel() if self.save_model is True else None
        for self.epoch in range(self.max_epochs):
            # Loop over epochs with train and test routines
            train_acc_avg, train_loss_avg = self._regular_train_epoch(loss_criterion)
            test_acc_avg, test_loss_avg = self._regular_test_epoch(loss_criterion)
            # Log epoch result
            if isinstance(self.logger, ModuleType):
                self.logger.log(
                    {
                        "train/acc": train_acc_avg.result(),
                        "train/loss": train_loss_avg.result(),
                        "test/acc": test_acc_avg.result(),
                        "test/loss": test_loss_avg.result(),
                    }
                )
            self._print_epoch_results(train_acc_avg, train_loss_avg, test_acc_avg, test_loss_avg)
            # Save model
            model_state = {"model_state_dict": self.arthur.state_dict()}
            optimizer_state = {"optimizer_state_dict": self.optimizer_arthur.state_dict()}
            if isinstance(save_best_model, SaveBestModel):
                save_best_model(
                    test_loss_avg.result(),
                    self.epoch,
                    model_state,
                    optimizer_state,
                    log_dir=f"runs/reg_train/{self.dataset}/state_dict/reg_train_{self.arthur.__class__.__name__}_{self.dataset}_{self.current_date}.pth",
                )
        return True

    def _regular_train_epoch(self, loss_criterion) -> Tuple[Averager, Averager]:
        """Training epoch for Arthur on regular training."""

        from captum._utils.common import _run_forward
        from typing import Any, Callable, Union, Tuple
        from torch import Tensor
        import torch

        # This is the same as the default compute_gradients function in captum._utils.gradient, except
        # setting create_graph=True when calling torch.autograd.grad
                
        assert self.optimizer_arthur is not None, "Optimizer for Arthur must be initialized."
        assert self.arthur is not None, "Arthur model must be initialized."
        # Set model to train mode
        self.arthur.train()
        # Initialize metrics
        train_loss_avg = Averager()
        train_acc_avg = Averager()
        # Loop over batches
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch in pbar:
            # Training loop
            x_input = batch[0].to(self.device)
            y_true = batch[1].to(self.device)
            batch_size = x_input.shape[0]

            if self.add_mask_channel is True:
                # Add mask channel to input
                ones = torch.ones_like(x_input[:, 0, :, :])
                x_input = torch.cat((x_input, ones.unsqueeze(1)), dim=1)

            # Inference
            output = self.arthur(x_input)
            if self.binary_classification is True:
                # Adapt tensors for BCEWithLogitsLoss
                output = output.squeeze()
                y_true = y_true.float()
            loss = loss_criterion(output, y_true)

            # Backpropagation
            loss.backward()
            self.optimizer_arthur.step()
            self.optimizer_arthur.zero_grad()
            # Metrics
            accuracy = categorical_accuracy(output, y_true, self.binary_classification)
            train_acc_avg.add(accuracy)
            train_loss_avg.add(loss.item())
            # Monitoring
            pbar.set_postfix({"Train Loss": train_loss_avg.result()})

        return train_acc_avg, train_loss_avg

    def _regular_test_epoch(self, loss_criterion) -> Tuple[Averager, Averager]:
        """Test epoch for Arthur on regular training"""
        assert self.arthur is not None, "Arthur model must be initialized."
        # Set model to eval mode
        self.arthur.eval()
        # Initialize metrics
        test_loss_avg = Averager()
        test_acc_avg = Averager()
        # Loop over batches
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Test Epoch {self.epoch + 1}/{self.max_epochs}")
            for batch in pbar:
                # Test loop
                x_input = batch[0].to(self.device)
                y_true = batch[1].to(self.device)

                if self.add_mask_channel is True:
                    # Add mask channel to input
                    ones = torch.ones_like(x_input[:, 0, :, :])
                    x_input = torch.cat((x_input, ones.unsqueeze(1)), dim=1)

                # Inference
                output = self.arthur(x_input)

                if self.binary_classification is True:
                    # Adapt tensors for BCEWithLogitsLoss
                    output = output.squeeze()
                    y_true = y_true.float()
                loss = loss_criterion(output, y_true)

                # Metrics
                loss = loss_criterion(output, y_true)
                accuracy = categorical_accuracy(output, y_true, self.binary_classification)
                test_acc_avg.add(accuracy)
                test_loss_avg.add(loss.item())
                # Monitoring
                pbar.set_postfix({"Test Loss": test_loss_avg.result()})

        return test_acc_avg, test_loss_avg

    def _print_epoch_results(
        self, train_acc_avg: Averager, train_loss_avg: Averager, test_acc_avg: Averager, test_loss_avg: Averager
    ):
        """Print results of epoch.

        Args:
            train_acc_avg (Averager): Average training accuracy.
            train_loss_avg (Averager): Average training loss.
            test_acc_avg (Averager): Average test accuracy.
            test_loss_avg (Averager): Average test loss.
        """
        # Calculate average loss and accuracy results of epoch
        train_acc = train_acc_avg.result()
        train_loss = train_loss_avg.result()
        test_acc = test_acc_avg.result()
        test_loss = test_loss_avg.result()
        # Print average results
        print(f"Train loss: {train_loss:1.4f} | Train Accuracy: {train_acc*100:2.2f}%")
        print(f"Test loss: {test_loss:1.4f} | Test Accuracy: {test_acc*100:2.2f}%")
        print(10 * "---")

    def train_min_max(self, max_epochs: int, gamma: float, steps_morgana: int) -> bool:
        """Train Arthur with MinMax training.

        This is the main training loop.
        It will train Arthur for the given number of epochs.
        It will also save the best model and optimizer states.
        It will also print the results of the training.

        Args:
            max_epochs (int): Number of epochs to train Arthur.
            mask_size (int): Size of the mask.
            gamma (float): Weight of the Morgana loss.
        Returns:
            None
        """
        # Check if all necessary parameters are set
        assert self.arthur is not None, "Arthur model must be initialized."
        assert isinstance(self.merlin, FeatureSelector) and isinstance(self.morgana, FeatureSelector)
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        assert isinstance(self.optimizer_merlin, torch.optim.Optimizer), "Merlin Optimizer not set."
        assert isinstance(self.optimizer_morgana, torch.optim.Optimizer), "Morgana Optimizer not set."
        self.print_trainer_details()
        # Set training parameters
        self.max_epochs = max_epochs
        self.gamma = gamma
        self.steps_morgana = steps_morgana
        save_best_model = SaveBestModel() if self.save_model is True else None
        # Loop over epochs
        for self.epoch in range(self.max_epochs):
            train_metrics = self._train_min_max_epoch()
            test_metrics = self._test_min_max_epoch()
            # LR Sechudler Step
            if isinstance(self.scheduler_arthur, torch.optim.lr_scheduler._LRScheduler):
                self.scheduler_arthur.step()
            if isinstance(self.scheduler_merlin, torch.optim.lr_scheduler._LRScheduler):
                self.scheduler_merlin.step()
            if isinstance(self.scheduler_morgana, torch.optim.lr_scheduler._LRScheduler):
                self.scheduler_morgana.step()
            # Print results
            train_metrics.print_results("Train")
            test_metrics.print_results("Test")
            print(10 * "---")
            # Log results
            if isinstance(self.logger, ModuleType):
                self.logger.log(
                    {
                        "train/loss": train_metrics.average("loss"),
                        "train/completeness": train_metrics.average("completeness"),
                        "train/completeness_continuous": train_metrics.average("completeness_continuous"),
                        "train/soundness": train_metrics.average("soundness"),
                        "train/soundness_continuous": train_metrics.average("soundness_continuous"),
                        "test/loss": test_metrics.average("loss"),
                        "test/completeness": test_metrics.average("completeness"),
                        "test/soundness": test_metrics.average("soundness")
                    },
                    step=self.epoch,
                )
            # Save model and optimizer states
            model_state = {
                "model_state_dict_arthur": self.arthur.state_dict(),
                "model_state_dict_merlin": self.merlin.state_dict(),
                "model_state_dict_morgana": self.morgana.state_dict(),
            }
            optimizer_state = {
                "optimizer_state_dict_arthur": self.optimizer_arthur.state_dict(),
                "optimizer_state_dict_merlin": self.optimizer_merlin.state_dict(),
                "optimizer_state_dict_morgana": self.optimizer_morgana.state_dict(),
            }
            if isinstance(save_best_model, SaveBestModel):
                save_best_model(
                    test_metrics.average("loss"),
                    self.epoch,
                    model_state,
                    optimizer_state,
                    log_dir=f"runs/min_max_train/merlin-arthur-framework/{self.dataset}/unet_approach/mask{self.mask_size}/state_dict/minmax_targets{self.string_of_targets}_{self.mask_size}_Params_{self.num_param:1.2E}_gamma{self.gamma}.pth",
                )

        return True

    def _train_min_max_epoch(self) -> MetricsCollector:
        """Train epoch for Arthur on MinMax training.

        Returns:
            MetricsCollector: Metrics collector with the results of the epoch.
        """
        # Check if all necessary parameters are set
        assert self.arthur is not None, "Arthur model must be initialized."
        assert isinstance(self.merlin, FeatureSelector) and isinstance(
            self.morgana, FeatureSelector
        ), f"Merlin and Morgana need to be FeatureSelectors, got Merlin={type(self.merlin)}, Morgana={type(self.morgana)}"
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        assert isinstance(self.optimizer_merlin, torch.optim.Optimizer), "Merlin Optimizer not set."
        assert isinstance(self.optimizer_morgana, torch.optim.Optimizer), "Morgana Optimizer not set."
        assert self.segmentation_method is not None, "Segmentation method not set."
        assert self.mask_size is not None, "Mask size not set."

        # Set models to train mode
        self.merlin.train()
        self.morgana.train()
        metrics_collector = MetricsCollector(
            "loss", "completeness", "soundness", "completeness_continuous", "soundness_continuous"
        )
        self.arthur.eval()  # NOTE: Arthur is in eval mode to prevent batchnorm from updating
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch in pbar:
            x_input = batch[0].to(self.device)
            y_true = batch[1].to(self.device)

            # Initialize inputs for Merlin and Morgana
            x_merlin = x_input
            y_true_merlin = y_true
            x_morgana = x_input
            y_true_morgana = y_true

            # If only on class is set, only use the samples of that class
            if self.merlin.only_on_class is True:
                # Merlin only for label 1
                x_merlin = x_input[y_true == 1]
                y_true_merlin = y_true[y_true == 1]
            if self.morgana.only_on_class is True:
                # Morgana only for label 0
                x_morgana = x_input[y_true == 0]
                y_true_morgana = y_true[y_true == 0]

            if self.segmentation_method == "soft_topk":
                # U-Net outputs
                output_merlin = self.merlin(x_merlin, y_true_merlin)
                output_morgana = self.morgana(x_morgana, y_true_morgana)
                # Normalize outputs
                merlin_output_normalized = self.merlin.normalize_l1(output_merlin, self.mask_size)
                morgana_output_normalized = self.morgana.normalize_l1(output_morgana, self.mask_size)
                # Penalize outputs
                l1_penalty_merlin = (
                    self.merlin.l1_penalty_coefficient * MaskRegularizer.l1_norm(merlin_output_normalized)
                    if self.merlin.l1_penalty
                    else 0
                )
                tv_penalty_merlin = (
                    self.merlin.tv_penalty_coefficient * MaskRegularizer.tv_norm(merlin_output_normalized)
                    if self.merlin.tv_penalty
                    else 0
                )
                l1_penalty_morgana = (
                    self.morgana.l1_penalty_coefficient * MaskRegularizer.l1_norm(morgana_output_normalized)
                    if self.morgana.l1_penalty
                    else 0
                )
                tv_penalty_morgana = (
                    self.morgana.tv_penalty_coefficient * MaskRegularizer.tv_norm(morgana_output_normalized)
                    if self.morgana.tv_penalty
                    else 0
                )
                # Total penalty
                penalty_merlin = l1_penalty_merlin + tv_penalty_merlin
                penalty_morgana = l1_penalty_morgana + tv_penalty_morgana

            else:
                # Merlin optimization (separate backward pass)
                merlin_output_normalized = self._optimize_unet(
                    x_merlin, y_true_merlin, self.merlin, self.optimizer_merlin, steps=1
                )
                # Morgana optimization (separate backward pass)
                morgana_output_normalized = self._optimize_unet(
                    x_morgana, y_true_morgana, self.morgana, self.optimizer_morgana, steps=self.steps_morgana
                )
            # Apply segmentation
            merlin_output_segmented = self.merlin.segment(merlin_output_normalized, self.segmentation_method)
            morgana_output_segmented = self.morgana.segment(morgana_output_normalized, self.segmentation_method)

            # Apply continuous mask on input
            x_masked_merlin = self.merlin.apply_mask(x_merlin, merlin_output_segmented)
            x_masked_merlin_continuous = self.merlin.apply_mask(x_merlin, merlin_output_normalized)
            x_masked_morgana = self.morgana.apply_mask(x_morgana, morgana_output_segmented)
            x_masked_morgana_continuous = self.merlin.apply_mask(x_morgana, morgana_output_normalized)

            # Arthur's response
            self.arthur.eval()  # NOTE: Need to be in eval mode to prevent batchnorm from updating
            logits_merlin = self.arthur(x_masked_merlin)
            logits_merlin_continuous = self.arthur(x_masked_merlin_continuous)
            logits_morgana = self.arthur(x_masked_morgana)
            logits_morgana_continuous = self.arthur(x_masked_morgana_continuous)

            # Calculate loss
            if self.binary_classification is True:
                # Convert tensors to appropriate format if binary classification is used
                logits_merlin = logits_merlin.squeeze(1)
                logits_morgana = logits_morgana.squeeze(1)
                y_true_merlin = y_true_merlin.float()
                y_true_morgana = y_true_morgana.float()
                # Loss
                arthur_loss = (
                    1
                    - torch.mean(torch.sigmoid(logits_merlin))
                    - self.gamma * (1 - torch.mean(torch.sigmoid(logits_morgana)))
                )
            else:
                # Loss
                merlin_loss = self.merlin.criterion(logits_merlin, y_true_merlin)
                morgana_loss = self.morgana.criterion(logits_morgana, y_true_morgana)
                arthur_loss = merlin_loss + self.gamma * morgana_loss

            # Backpropagation
            arthur_loss.backward()
            # Optimization step for Arthur
            self.optimizer_arthur.step()
            if self.segmentation_method == "soft_topk":
                self.optimizer_merlin.step()
                self.optimizer_morgana.step()
            # Zero gradients (including Merlin and Morgana)
            self.optimizer_arthur.zero_grad()
            self.optimizer_merlin.zero_grad()
            self.optimizer_morgana.zero_grad()

            # Metrics
            completeness = get_accuracy(
                logits_merlin, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            completeness_continuous = get_accuracy(
                logits_merlin_continuous, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            soundness = get_accuracy(
                logits_morgana, y_true_morgana, mode="morgana", binary_classification=self.binary_classification
            )
            soundness_continuous = get_accuracy(
                logits_morgana_continuous,
                y_true_morgana,
                mode="morgana",
                binary_classification=self.binary_classification,
            )

            metrics_collector.add(
                loss=arthur_loss.item(),
                completeness=completeness,
                soundness=soundness,
                completeness_continuous=completeness_continuous,
                soundness_continuous=soundness_continuous,
            )
            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": metrics_collector.average("loss"),
                    "Compl.": metrics_collector.average("completeness"),
                    "Compl. cont.": metrics_collector.average("completeness_continuous"),
                    "Sound. cont.": metrics_collector.average("soundness_continuous"),
                    "Sound.": metrics_collector.average("soundness"),
                }
            )

        return metrics_collector

    @torch.no_grad()
    def _test_min_max_epoch(self) -> MetricsCollector:
        """Test epoch for Merlin-Arthur Classifier on MinMax setup.

        This is the same as the train epoch for Arthur on MinMax training.
        The only difference is that the Arthur model is in eval mode.
        Also, the gradients are not computed and the optimization step is not performed.
        Finally, the heatmaps are logged to Tensorboard.

        Returns:
            MetricsCollector: Metrics collector with the results of the epoch.
        """
        # Check if all necessary parameters are set
        assert self.arthur is not None, "Arthur model must be initialized."
        assert isinstance(self.merlin, FeatureSelector), "Merlin must be a FeatureSelector"
        assert isinstance(self.morgana, FeatureSelector), "Morgana must be a FeatureSelector"
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        assert isinstance(self.optimizer_merlin, torch.optim.Optimizer), "Merlin Optimizer not set."
        assert isinstance(self.optimizer_morgana, torch.optim.Optimizer), "Morgana Optimizer not set."
        assert self.segmentation_method is not None, "Segmentation method not set."
        assert self.gamma is not None, "Gamma not set."
        assert self.mask_size is not None, "Mask size not set."
        # Start eval mode
        self.arthur.eval()
        self.merlin.eval()
        self.morgana.eval()
        # Initialize unbound tensor
        mask_binarized = None
        # Initialize MetricsCollector object
        metrics_collector = MetricsCollector(
            "loss",
            "completeness",
            "soundness",
            "completeness_continuous",
            "soundness_continuous")
        pbar = tqdm(self.test_loader, desc=f"Test Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch in pbar:
            x_input = batch[0].to(self.device)
            y_true = batch[1].to(self.device)

            # Initialize inputs for Merlin and Morgana
            x_merlin = x_input
            y_true_merlin = y_true
            x_morgana = x_input
            y_true_morgana = y_true

            # If only on class is set, only use the samples of that class
            if self.merlin.only_on_class is True:
                # Merlin only for label 1
                x_merlin = x_input[y_true == 1]
                y_true_merlin = y_true[y_true == 1]
            if self.morgana.only_on_class is True:
                # Morgana only for label 0
                x_morgana = x_input[y_true == 0]
                y_true_morgana = y_true[y_true == 0]

            if self.calculate_iou is True:
                batch_left = batch[4].to(self.device)
                batch_top = batch[5].to(self.device)
                batch_width = batch[6].to(self.device)
                batch_height = batch[7].to(self.device)

                batch_left = batch_left[y_true == 1]
                batch_top = batch_top[y_true == 1]
                batch_width = batch_width[y_true == 1]
                batch_height = batch_height[y_true == 1]

            # Feature Extraction
            merlin_output = self.merlin(x_merlin, y_true_merlin)
            morgana_output = self.morgana(x_morgana, y_true_morgana)
            # Normalize
            merlin_output_normalized = self.merlin.normalize_l1(merlin_output, self.mask_size)
            morgana_output_normalized = self.morgana.normalize_l1(morgana_output, self.mask_size)
            # Segmentation
            merlin_output_segmented = self.merlin.segment(merlin_output_normalized, self.segmentation_method)
            morgana_output_segmented = self.morgana.segment(morgana_output_normalized, self.segmentation_method)
            # Apply mask
            x_masked_merlin = self.merlin.apply_mask(x_merlin, merlin_output_segmented)
            x_masked_morgana = self.morgana.apply_mask(x_morgana, morgana_output_segmented)
            x_masked_merlin_continuous = self.merlin.apply_mask(x_merlin, merlin_output_normalized)
            x_masked_morgana_continuous = self.morgana.apply_mask(x_morgana, morgana_output_normalized)
            # Arthur's response
            logits_merlin = self.arthur(x_masked_merlin)
            logits_morgana = self.arthur(x_masked_morgana)
            logits_merlin_continuous = self.arthur(x_masked_merlin_continuous)
            logits_morgana_continuous = self.arthur(x_masked_morgana_continuous)

            with torch.autocast(device_type=self.device.type, enabled=self.merlin.use_amp):  # type: ignore
                # Test Iou calculation
                if self.calculate_iou is True:
                    mask_binarized = merlin_output_segmented
                    if self.segmentation_method == "soft_topk":
                        # Binarize Mask for IoU calculation
                        mask_binarized = self.merlin.segment(merlin_output_segmented, "topk")
                    # Intersection over Union (IoU)
                    iou_batch = IoUCalculator.iou_batch(
                        mask_binarized, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    iou_batch_mean = torch.mean(iou_batch)  # type: ignore
                    # Pointing Game (PG)
                    pointing_game_batch = PointingGameCalculator.pointing_game_batch(
                        merlin_output_normalized, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    pointing_game_batch_mean = torch.mean(pointing_game_batch)  # type: ignore
                    # Relevant Mass Accuracy (RMA)
                    rma_batch = RMACalculator.rma_batch(
                        merlin_output_normalized, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rma_batch_mean = torch.mean(rma_batch)  # type: ignore
                    # Relevant Rank Accuracy (RRA)
                    rra_batch = RRACalculator.rra_batch(
                        merlin_output_normalized, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rra_batch_mean = torch.mean(rra_batch)  # type: ignore

                # Calculate loss
                if self.binary_classification is True:
                    # Convert tensors to appropriate format if binary classification is used
                    logits_merlin = logits_merlin.squeeze(1)
                    logits_morgana = logits_morgana.squeeze(1)
                    y_true_merlin = y_true_merlin.float()
                    y_true_morgana = y_true_morgana.float()
                    # Loss
                    arthur_loss = (
                        1
                        - torch.mean(torch.sigmoid(logits_merlin))
                        - self.gamma * (1 - torch.mean(torch.sigmoid(logits_morgana)))
                    )
                else:
                    # Loss
                    merlin_loss = self.merlin.criterion(logits_merlin, y_true_merlin)
                    morgana_loss = self.morgana.criterion(logits_morgana, y_true_morgana)
                    arthur_loss = merlin_loss + self.gamma * morgana_loss

                # Metrics
                completeness = get_accuracy(
                    logits_merlin, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
                )
                soundness = get_accuracy(
                    logits_morgana, y_true_morgana, mode="morgana", binary_classification=self.binary_classification
                )
                completeness_continuous = get_accuracy(
                    logits_merlin_continuous,
                    y_true_merlin,
                    mode="merlin",
                    binary_classification=self.binary_classification,
                )
                soundness_continuous = get_accuracy(
                    logits_morgana_continuous,
                    y_true_morgana,
                    mode="morgana",
                    binary_classification=self.binary_classification,
                )

                metrics_collector.add(
                    loss=arthur_loss.item(),
                    completeness=completeness,
                    soundness=soundness,
                    completeness_continuous=completeness_continuous,
                    soundness_continuous=soundness_continuous)

                # Monitoring
                pbar.set_postfix(
                    {
                        "Loss": metrics_collector.average("loss"),
                        "Compl.": metrics_collector.average("completeness"),
                        "Sound.": metrics_collector.average("soundness"),
                        "Compl. cont.": metrics_collector.average("completeness_continuous"),
                        "Sound. cont.": metrics_collector.average("soundness_continuous")}
                )

        # Logging
        if isinstance(self.logger, ModuleType) and self.data_type == "image":
            morgana_binarized = self.morgana.segment(morgana_output_segmented, "topk")  # type: ignore
            if self.dataset == "SVHN":
                # Add channel dimension to Feature Selector output in case of SVHN
                merlin_output = merlin_output.repeat(1, 3, 1, 1)  # type: ignore
                merlin_output_segmented = merlin_output_segmented.repeat(1, 3, 1, 1)  # type: ignore
                morgana_output = morgana_output.repeat(1, 3, 1, 1)  # type: ignore
                morgana_output_segmented = morgana_output_segmented.repeat(1, 3, 1, 1)  # type: ignore
            # Get batch size
            batch_size_merlin = x_merlin.size()[0]  # type: ignore
            batch_size_morgana = x_morgana.size()[0]  # type: ignore
            n = 30 if batch_size_merlin > 30 and batch_size_morgana > 30 else min(batch_size_merlin, batch_size_morgana)
            # Log Feature Selections
            image_grid = [x_merlin[:n], merlin_output[:n], merlin_output_segmented[:n], x_masked_merlin[:n, :3, :, :], x_morgana[:n], morgana_output[:n], morgana_output_segmented[:n], x_masked_morgana[:n, :3, :, :]]  # type: ignore
            if self.segmentation_method == "soft_topk":
                # Binarize Masks for logging
                assert mask_binarized is not None
                mask_binarized = mask_binarized.repeat(1, 3, 1, 1)
                morgana_binarized = morgana_binarized.repeat(1, 3, 1, 1)
                image_grid.insert(3, mask_binarized[:n])
                image_grid.insert(-1, morgana_binarized[:n])
            image_grid = torch.cat(image_grid, dim=0)
            image_grid = utils.make_grid(image_grid, nrow=n, normalize=True, scale_each=True)
            images = wandb.Image(
                image_grid,
                caption="From Top to bottom: Input, Merlin Output, Merlin Segmented, Merlin Masked, Morgana Output, Morgana Segmented, Morgana Masked",
            )
            wandb.log({"Feature Selections": images}, commit=False)

        return metrics_collector

    def _optimize_unet(
        self,
        x_input: torch.Tensor,
        y_true: torch.Tensor,
        feature_selector: FeatureSelector,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
    ) -> torch.Tensor:
        """Optimize a FeatureSelector model (Merlin or Morgana) for a given number of steps.

        Args:
            x_input (torch.Tensor): Input tensor.
            y_true (torch.Tensor): True labels.
            feature_selector (FeatureSelector): FeatureSelector model to optimize.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            steps (int, optional): Number of steps to optimize the model. Defaults to 1.
        Returns:
            torch.Tensor: Normalized output of the model.

        Raises:
            AssertionError: If the input is not a torch Tensor.
            AssertionError: If the true labels are not a torch Tensor.
            AssertionError: If the number of steps is not a positive integer.
            AssertionError: If the model is not a FeatureSelector.
            AssertionError: If the optimizer is not a torch.optim.Optimizer.
            AssertionError: If the criterion is not a torch.nn.CrossEntropyLoss or a MorganaCriterion.
            AssertionError: If Arthur's optimizer is not set.
        """
        # Assertions
        assert self.arthur is not None, "Arthur must be set"
        assert isinstance(x_input, torch.Tensor), "Input must be a torch Tensor"
        assert isinstance(y_true, torch.Tensor), "True labels must be a torch Tensor"
        assert steps >= 0, "Steps must be a positive integer"
        assert isinstance(feature_selector, FeatureSelector), "Model must be a FeatureSelector"
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer must be a torch.optim.Optimizer"
        assert isinstance(
            feature_selector.criterion, (torch.nn.CrossEntropyLoss, MorganaCriterion, torch.nn.BCEWithLogitsLoss)
        ), "Criterion must be a torch.nn.CrossEntropyLoss or a MorganaCriterion"
        assert self.optimizer_arthur is not None, "Arthur's optimizer must be set"
        # Get parameters
        l1_lambda = feature_selector.l1_penalty_coefficient
        l2_lambda = feature_selector.l2_penalty_coefficient
        tv_lambda = feature_selector.tv_penalty_coefficient
        # In case steps == 0, we return a tensor of zeros
        output = torch.zeros_like(x_input)

        saliency_loss = SaliencyLoss(
            num_classes=3, mode=feature_selector.mode, destroyer_loss_active=self.destroyer_loss_active
        )

        self.arthur.eval()  # NOTE: Attention!
        # Optimization loop
        for _ in range(steps):
            # Feature Extraction
            output = feature_selector(x_input, y_true)

            # Loss
            if self.other_loss is True:
                loss = saliency_loss.get(output, x_input, y_true, self.arthur)
            else:
                output = feature_selector.normalize_l1(output, self.mask_size)  # type: ignore

                # Regularization
                l1_penalty = l1_lambda * MaskRegularizer.l1_norm(output) if feature_selector.l1_penalty else 0
                l2_penalty = l2_lambda * MaskRegularizer.l2_norm(output) if feature_selector.l2_penalty else 0
                tv_penalty = tv_lambda * MaskRegularizer.tv_norm(output) if feature_selector.tv_penalty else 0

                # # Consider penalty for border pixels
                # if border_penalty > 0:
                # border = 0.5*torch.sum(output[:,:,-1,:]**2 + output[:,:,0,:]**2 + output[:,:,:,-1]**2 + output[:,:,:,0]**2)
                # else:
                #     border = 0.

                # Sum of regularization terms
                total_penalty = l1_penalty + l2_penalty + tv_penalty

                # Morgana maximizes objective but still tries to minimize the penalties
                if feature_selector.mode == "morgana" and self.binary_classification is not True:
                    total_penalty = -total_penalty

                # Apply continuous mask on input
                x_masked = feature_selector.apply_mask(x_input, output)  # type: ignore
                # Arthur's response
                # self.arthur.eval()  # Note: eval() is important here, due to running averages in BatchNorm
                logits = self.arthur(x_masked)

                # Loss
                if self.binary_classification is True:
                    logits = logits.squeeze(1)
                    y_true = y_true.float()
                    # apply sigmoid to logits and sum over all samples
                    loss = -torch.mean(torch.sigmoid(logits)) + total_penalty
                else:
                    loss = feature_selector.criterion(logits, y_true) + total_penalty

            # Separate Backpropagation of U-Nets
            loss.backward()
            # Optimization step
            optimizer.step()
            # Zero gradients (including Arthur)
            self.optimizer_arthur.zero_grad()
            optimizer.zero_grad()

        return output

    def train_min_max_with_mask_optimization(self, max_epochs: int, gamma: float) -> bool:
        """Train Arthur on MinMax training with SFW or unconstrained optimization.

        Calls training and testing routines for a given number of epochs.

        Args:
            max_epochs (int): Number of epochs in total.
            gamma (float): Morgana loss weight parameter.

        Raises:
            AssertionError: If number of epochs is not greater than 0.
            AssertionError: If gamma is not greater than 0.
            AssertionError: If Arthur's optimizer is not set.
        """
        # Asserts
        assert self.arthur is not None, "Arthur must be set"
        assert max_epochs > 0, "Number of epochs must be greater than 0."
        assert gamma >= 0, "Gamma must be greater than or equal to 0."
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        self.print_trainer_details()
        # Initialize variables
        self.max_epochs = max_epochs
        self.gamma = gamma
        # Initialize best model saver
        save_best_model = SaveBestModel() if self.save_model is True else None
        # Loop over epochs
        for self.epoch in range(self.max_epochs):
            train_metrics = self._train_mask_optimization_epoch()
            test_metrics = self._test_mask_optimization_epoch()
            if isinstance(self.scheduler_arthur, torch.optim.lr_scheduler._LRScheduler):
                self.scheduler_arthur.step()
            # Print results
            train_metrics.print_results("Train")
            test_metrics.print_results("Test")
            # Log epoch results
            if isinstance(self.logger, ModuleType):
                self.logger.log(
                    {
                        "train/loss": train_metrics.average("loss"),
                        "train/completeness": train_metrics.average("completeness"),
                        "completeness_continuous": train_metrics.average("completeness_continuous"),
                        "train/soundness": train_metrics.average("soundness"),
                        "test/loss": test_metrics.average("loss"),
                        "test/completeness": test_metrics.average("completeness"),
                        "test/soundness": test_metrics.average("soundness"),
                    },
                    step=self.epoch,
                )
            # Save model and optimizer states
            if self.save_model and isinstance(save_best_model, SaveBestModel):
                model_state = {"model_state_dict_arthur": self.arthur.state_dict()}
                optimizer_state = {"optimizer_state_dict_arthur": self.optimizer_arthur.state_dict()}
                save_best_model(
                    test_metrics.average("loss"),
                    self.epoch,
                    model_state,
                    optimizer_state,
                    log_dir=f"runs/min_max_train/merlin-arthur-framework/{self.dataset}/{self.approach}_approach/mask{self.mask_size}/state_dict/{self.arthur.__class__.__name__}_{self.dataset}_masksize_{self.mask_size}_Params_{self.num_param:1.2E}_gamma_{self.gamma}_lr_{self.lr:1.0E}.pth",
                )

        return True

    def _train_mask_optimization_epoch(self) -> MetricsCollector:
        """Train Arthur on MinMax training with SFW or unconstrained mask optimization for one epoch.

        Returns:
            MetricsCollector: MetricsCollector object with metrics for the epoch.

        Raises:
            AssertionError: If Arthur's optimizer is not set.
            AssertionError: If Merlin is not set.
            AssertionError: If Morgana is not set.
        """
        assert self.arthur is not None, "Arthur must be set"
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        assert self.merlin is not None, "Merlin must be set."
        assert self.morgana is not None, "Morgana must be set."
        assert self.mask_size is not None, "Mask size must be set."
        assert self.segmentation_method is not None, "Segmentation method must be set."
        # Start train mode
        self.arthur.eval()
        # Initialize Mask collection for Merlin's brute force serach
        grid = torch.Tensor()
        # Initialize mask collection for Morgana's brute force serach
        mask_collection = None
        if self.brute_force_merlin is True and self.brute_force_morgana is True:
            # Initialize mask collection for Merlin's brute force serach
            x_input, *_ = next(iter(self.train_loader))
            num_features = x_input.shape[1]
            array = np.array(list(itertools.combinations(range(num_features), self.mask_size)))
            grid = np.zeros((len(array), num_features), dtype="float32")
            grid[np.arange(len(array))[None].T, array] = 1
            grid = torch.tensor(grid)
        # Initialize MetricsCollector object
        metrics_collector = MetricsCollector("loss", "completeness", "completeness_continuous", "soundness")
        # Initialize dataloader before each epoch to include newest masks for Merlin and Morgana
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)  # type: ignore
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch in pbar:  # x_input, y_true, mask_merlin, mask_morgana, idx
            # Send to GPU
            x_input, y_true = batch[0].to(self.device), batch[1].to(self.device)
            mask_merlin, mask_morgana = batch[2].to(self.device), batch[3].to(self.device)
            idx = batch[-1]

            # Initialize inputs for Merlin and Morgana
            x_merlin = x_input
            y_true_merlin = y_true
            x_morgana = x_input
            y_true_morgana = y_true

            # If only on class is set, only use the samples of that class
            if self.merlin.only_on_class is True:
                # Merlin only for label 1
                x_merlin = x_input[y_true == 1]
                y_true_merlin = y_true[y_true == 1]
                mask_merlin = mask_merlin[y_true == 1]
            if self.morgana.only_on_class is True:
                # Morgana only for label 0
                x_morgana = x_input[y_true == 0]
                y_true_morgana = y_true[y_true == 0]
                mask_morgana = mask_morgana[y_true == 0]

            if self.original_image_ratio is not None:
                # Calculate the index up to which to keep the original images
                cutoff_idx_merlin = int(self.original_image_ratio * x_merlin.shape[0])
                cutoff_idx_morgana = int(self.original_image_ratio * x_morgana.shape[0])
                # Keep only the first 'cutoff_index' items along the first dimension (usually the batch dimension)
                x_merlin_original = x_merlin[:cutoff_idx_merlin]
                x_merlin = x_merlin[cutoff_idx_merlin:]
                x_morgana_orignal = x_morgana[:cutoff_idx_morgana]
                x_morgana = x_morgana[cutoff_idx_morgana:]
                # Labels
                y_true_merlin_original = y_true_merlin[:cutoff_idx_merlin]
                y_true_morgana_original = y_true_morgana[:cutoff_idx_morgana]
                y_true_merlin = y_true_merlin[cutoff_idx_merlin:]
                y_true_morgana = y_true_morgana[cutoff_idx_morgana:]
                # Masks
                mask_merlin = mask_merlin[cutoff_idx_merlin:]
                mask_morgana = mask_morgana[cutoff_idx_morgana:]

            if self.brute_force_merlin is True:
                # Merlin's Brute Force Search
                mask_merlin_sfw = self.merlin.brute_force_search(x_merlin, y_true_merlin, self.arthur, grid)
            else:
                # Merlin's SFW
                mask_merlin_sfw_continuous = self.merlin(x_merlin, y_true_merlin, self.arthur, mask_merlin)
                mask_merlin_sfw = self.merlin.segment(mask_merlin_sfw_continuous, method=self.segmentation_method)

            if self.brute_force_morgana is True:
                # Morgana's Brute Force Search
                mask_collection = (
                    mask_merlin_sfw if mask_collection is None else torch.cat((mask_collection, mask_merlin_sfw), dim=0)
                )
                mask_collection = torch.unique(mask_collection, dim=0)
                mask_morgana_sfw = self.morgana.brute_force_search(x_input, y_true, self.arthur, mask_collection)
            else:
                # Morgana's SFW
                mask_morgana_sfw = self.morgana(x_morgana, y_true_morgana, self.arthur, mask_morgana)  # type: ignore
                mask_morgana_sfw = self.morgana.segment(mask_morgana_sfw, method=self.segmentation_method)

            with torch.autocast(device_type=self.device.type, enabled=self.merlin.use_amp):  # type: ignore
                # Apply mask to input
                x_masked_merlin = self.merlin.apply_mask(x_merlin, mask_merlin_sfw)
                x_masked_merlin_continuous = self.merlin.apply_mask(x_merlin, mask_merlin_sfw_continuous)  # type: ignore
                x_masked_morgana = self.morgana.apply_mask(x_morgana, mask_morgana_sfw)
                # Blend in original images
                if self.original_image_ratio is not None:
                    x_masked_merlin = torch.cat((x_merlin_original, x_masked_merlin), dim=0)  # type: ignore
                    x_masked_morgana = torch.cat((x_morgana_orignal, x_masked_morgana), dim=0)  # type: ignore
                    y_true_merlin = torch.cat((y_true_merlin_original, y_true_merlin), dim=0)  # type: ignore
                    y_true_morgana = torch.cat((y_true_morgana_original, y_true_morgana), dim=0)  # type: ignore
                    # Continuous case
                    x_masked_merlin_continuous = torch.cat((x_merlin_original, x_masked_merlin_continuous), dim=0)  # type: ignore

                # Arthur's response to masked input
                logits_merlin = self.arthur(x_masked_merlin)
                logits_merlin_continuous = self.arthur(x_masked_merlin_continuous)
                logits_morgana = self.arthur(x_masked_morgana)

                # Convert tensors to appropriate format if binary classification is used
                if self.binary_classification is True:
                    logits_merlin = logits_merlin.squeeze(1)
                    y_true_merlin = y_true_merlin.float()
                    logits_morgana = logits_morgana.squeeze(1)
                    y_true_morgana = y_true_morgana.float()

                # Loss
                if self.binary_classification is True and self.merlin.optimize_probabilities is True:
                    arthur_loss = (
                        1
                        - torch.mean(torch.sigmoid(logits_merlin))
                        - self.gamma * (1 - torch.mean(torch.sigmoid(logits_morgana)))
                    )
                else:
                    merlin_loss = self.merlin.criterion(logits_merlin, y_true_merlin)
                    morgana_loss = self.morgana.criterion(logits_morgana, y_true_morgana)
                    arthur_loss = merlin_loss + self.gamma * morgana_loss
            # Backpropagation
            self.optimizer_arthur.zero_grad()
            arthur_loss.backward()
            self.optimizer_arthur.step()

            if self.original_image_ratio is not None:
                logits_merlin = logits_merlin[cutoff_idx_merlin:]  # type: ignore
                logits_morgana = logits_morgana[cutoff_idx_morgana:]  # type: ignore
                logits_merlin_continuous = logits_merlin_continuous[cutoff_idx_merlin:]  # type: ignore
                y_true_merlin = y_true_merlin[cutoff_idx_merlin:]  # type: ignore
                y_true_morgana = y_true_morgana[cutoff_idx_morgana:]  # type: ignore

            # Metrics
            completeness = get_accuracy(
                logits_merlin, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            completeness_continuous = get_accuracy(
                logits_merlin_continuous, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            soundness = get_accuracy(
                logits_morgana, y_true_morgana, mode="morgana", binary_classification=self.binary_classification
            )
            metrics_collector.add(
                loss=arthur_loss.item(),
                completeness=completeness,
                completeness_continuous=completeness_continuous,
                soundness=soundness,
            )
            # Monitoring
            pbar.set_postfix(
                {
                    "Loss": metrics_collector.average("loss"),
                    "Compl.": metrics_collector.average("completeness"),
                    "completeness_continuous": metrics_collector.average("completeness_continuous"),
                    "Sound.": metrics_collector.average("soundness"),
                }
            )
            # Error handling for CustomMNIST and CustomUCICensus
            if (
                not isinstance(self.train_data, CustomMNIST)
                and not isinstance(self.train_data, CustomUCICensus)
                and not isinstance(self.train_data, SVHNDataset)
                and not isinstance(self.train_data, ConcatDataset)
            ):
                raise TypeError(f"Expected type CustomMNIST or CustomUCICensus, got {type(self.train_data)}")

            # Check if train_data is CustomMNIST or CustomUCICensus
            if (
                isinstance(self.train_data, CustomMNIST)
                or isinstance(self.train_data, CustomUCICensus)
                or isinstance(self.train_data, SVHNDataset)
                and isinstance(self.train_data, ConcatDataset)
            ) and self.save_masks:
                # Update masks in train_data
                self.train_data.set_mask(mask_merlin_sfw.cpu(), mode="merlin", idx=idx)
                self.train_data.set_mask(mask_morgana_sfw.cpu(), mode="morgana", idx=idx)

        return metrics_collector

    def _test_mask_optimization_epoch(self) -> MetricsCollector:
        """Test SFW or unconstrained optimization min-max epoch.

        Returns:
            MetricsCollector: Metrics collector.
        """
        assert self.arthur is not None, "Arthur must be set"
        assert self.merlin is not None, "Merlin must be set."
        assert self.morgana is not None, "Morgana must be set."
        assert self.gamma is not None, "Gamma not set."
        assert self.mask_size is not None, "Mask size not set."
        assert self.segmentation_method is not None, "Segmentation method not set."
        metrics_collector = MetricsCollector(
            "loss", "completeness", "completeness_continuous", "soundness")
        self.arthur.eval()
        iou_batch_mean = 0
        # Initialize Mask collection for Merlin's brute force serach
        grid = torch.Tensor()
        # Initialize mask collection for Morgana's brute force serach
        mask_collection = None
        if self.brute_force_merlin is True and self.brute_force_morgana is True:
            # Initialize mask collection for Merlin's brute force serach
            x_input, *_ = next(iter(self.train_loader))
            num_features = x_input.shape[1]
            array = np.array(list(itertools.combinations(range(num_features), self.mask_size)))
            grid = np.zeros((len(array), num_features), dtype="float32")
            grid[np.arange(len(array))[None].T, array] = 1
            grid = torch.tensor(grid)
        pbar = tqdm(self.test_loader, desc=f"Test Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch in pbar:
            # Send to GPU
            x_input, y_true = batch[0].to(self.device), batch[1].to(self.device)
            mask_merlin, mask_morgana = batch[2].to(self.device), batch[3].to(self.device)

            if self.calculate_iou is True:
                batch_left = batch[4].to(self.device)
                batch_top = batch[5].to(self.device)
                batch_width = batch[6].to(self.device)
                batch_height = batch[7].to(self.device)

                batch_left = batch_left[y_true == 1]
                batch_top = batch_top[y_true == 1]
                batch_width = batch_width[y_true == 1]
                batch_height = batch_height[y_true == 1]

            # Initialize inputs for Merlin and Morgana
            x_merlin = x_input
            y_true_merlin = y_true
            x_morgana = x_input
            y_true_morgana = y_true

            # If only on class is set, only use the samples of that class
            if self.merlin.only_on_class is True:
                # Merlin only for label 1
                x_merlin = x_input[y_true == 1]
                y_true_merlin = y_true[y_true == 1]
                mask_merlin = mask_merlin[y_true == 1]
            if self.morgana.only_on_class is True:
                # Morgana only for label 0
                x_morgana = x_input[y_true == 0]
                y_true_morgana = y_true[y_true == 0]
                mask_morgana = mask_morgana[y_true == 0]

            if self.brute_force_merlin is True:
                # Merlin's Brute Force Search
                mask_merlin_sfw = self.merlin.brute_force_search(x_input, y_true, self.arthur, grid)
            else:
                # Merlin's SFW
                mask_merlin_sfw_continuous = self.merlin(x_merlin, y_true_merlin, self.arthur, mask_merlin)
                mask_merlin_sfw = self.merlin.segment(mask_merlin_sfw_continuous, method=self.segmentation_method)

            if self.brute_force_morgana is True:
                # Morgana's Brute Force Search
                mask_collection = (
                    mask_merlin_sfw if mask_collection is None else torch.cat((mask_collection, mask_merlin_sfw), dim=0)
                )
                mask_collection = torch.unique(mask_collection, dim=0)
                mask_morgana_sfw = self.morgana.brute_force_search(x_input, y_true, self.arthur, mask_collection)
            else:
                # Morgana's SFW
                mask_morgana_sfw = self.morgana(x_morgana, y_true_morgana, self.arthur, mask_morgana)  # type: ignore
                mask_morgana_sfw = self.morgana.segment(mask_morgana_sfw, method=self.segmentation_method)

            with torch.autocast(device_type=self.device.type, enabled=self.merlin.use_amp):  # type: ignore
                # Test Iou calculation for SVHN Dataset
                if self.calculate_iou is True:
                    # Intersection over Union (IoU)
                    iou_batch = IoUCalculator.iou_batch(
                        mask_merlin_sfw, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    iou_batch_mean = torch.mean(iou_batch)  # type: ignore
                    # Pointing Game (PG)
                    pointing_game_batch = PointingGameCalculator.pointing_game_batch(
                        mask_merlin_sfw_continuous, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    pointing_game_batch_mean = torch.mean(pointing_game_batch)  # type: ignore
                    # Relevant Mask Accuracy (RMA)
                    rma_batch = RMACalculator.rma_batch(
                        mask_merlin_sfw_continuous, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rma_batch_mean = torch.mean(rma_batch)  # type: ignore
                    # Relevant Rank Accuracy (RRA)
                    rra_batch = RRACalculator.rra_batch(
                        mask_merlin_sfw_continuous, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rra_batch_mean = torch.mean(rra_batch)  # type: ignore

                # Apply masks to input
                x_masked_merlin = self.merlin.apply_mask(x_merlin, mask_merlin_sfw)
                x_masked_merlin_continuous = self.merlin.apply_mask(x_merlin, mask_merlin_sfw_continuous)  # type: ignore
                x_masked_morgana = self.morgana.apply_mask(x_morgana, mask_morgana_sfw)
                # Arthur's predictions with masked input
                logits_merlin = self.arthur(x_masked_merlin)
                logits_merlin_continuous = self.arthur(x_masked_merlin_continuous)
                logits_morgana = self.arthur(x_masked_morgana)

                # Convert tensors to appropriate format if binary classification is used
                if self.binary_classification is True:
                    logits_merlin = logits_merlin.squeeze(1)
                    y_true_merlin = y_true_merlin.float()
                    logits_morgana = logits_morgana.squeeze(1)
                    y_true_morgana = y_true_morgana.float()

                # Loss
                if self.binary_classification is True and self.merlin.optimize_probabilities is True:
                    arthur_loss = (
                        1
                        - torch.mean(torch.sigmoid(logits_merlin))
                        - self.gamma * (1 - torch.mean(torch.sigmoid(logits_morgana)))
                    )
                else:
                    merlin_loss = self.merlin.criterion(logits_merlin, y_true_merlin)
                    morgana_loss = self.morgana.criterion(logits_morgana, y_true_morgana)
                    arthur_loss = merlin_loss + self.gamma * morgana_loss

                # Metrics
                completeness = get_accuracy(
                    logits_merlin, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
                )
                completeness_continuous = get_accuracy(
                    logits_merlin_continuous,
                    y_true_merlin,
                    mode="merlin",
                    binary_classification=self.binary_classification,
                )
                soundness = get_accuracy(
                    logits_morgana, y_true_morgana, mode="morgana", binary_classification=self.binary_classification
                )
                metrics_collector.add(
                    loss=arthur_loss.item(),
                    completeness=completeness,
                    completeness_continuous=completeness_continuous,
                    soundness=soundness)
                # Update progress bar
                pbar.set_postfix(
                    {
                        "Loss": metrics_collector.average("loss"),
                        "Compl.": metrics_collector.average("completeness"),
                        "Compl. (cont.)": metrics_collector.average("completeness_continuous"),
                        "Sound.": metrics_collector.average("soundness")
                    }
                )
        if isinstance(self.logger, ModuleType) and self.data_type == "image":
            if self.dataset == "SVHN":
                # Add channel dimension to Feature Selector output in case of SVHN
                mask_merlin_sfw = mask_merlin_sfw.repeat(1, 3, 1, 1)  # type: ignore
                mask_merlin_sfw_continuous = mask_merlin_sfw_continuous.repeat(1, 3, 1, 1)  # type: ignore
                mask_morgana_sfw = mask_morgana_sfw.repeat(1, 3, 1, 1)  # type: ignore
            # Log Feature Selections
            image_grid = [x_merlin[:30], mask_merlin_sfw_continuous[:30], mask_merlin_sfw[:30], x_masked_merlin[:30, :3, :, :], x_morgana[:30], mask_morgana_sfw[:30], x_masked_morgana[:30, :3, :, :]]  # type: ignore
            image_grid = torch.cat(image_grid, dim=0)
            image_grid = utils.make_grid(image_grid, nrow=30, normalize=False, scale_each=False)
            images = wandb.Image(
                image_grid,
                caption="From Top to bottom: Input, Merlin Output, Merlin Segmented, Merlin Masked, Morgana Output, Morgana Segmented, Morgana Masked",
            )
            wandb.log({"Feature Selections": images}, commit=False)

        return metrics_collector

    def train_min_max_with_posthoc(self, max_epochs: int, gamma: float) -> bool:
        """Train Arthur on MinMax training with posthoc method as Merlin.

        Calls training and testing routines for a given number of epochs.

        Args:
            max_epochs (int): Number of epochs in total.
            gamma (float): Morgana loss weight parameter.

        Raises:
            AssertionError: If number of epochs is not greater than 0.
            AssertionError: If gamma is not greater than 0.
            AssertionError: If Arthur's optimizer is not set.
        """
        # Asserts
        assert self.arthur is not None, "Arthur must be set"
        assert max_epochs > 0, "Number of epochs must be greater than 0."
        assert gamma >= 0, "Gamma must be greater than or equal to 0."
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        self.print_trainer_details()
        # Initialize variables
        self.max_epochs = max_epochs
        self.gamma = gamma
        # Start eval mode
        self.arthur.eval()
        # Instantiate xAI PostHoc Method (Merlin)
        self.ggc = GuidedGradCam(self.arthur, self.arthur.layer4[-1].conv3)  # type: ignore
        # Initialize best model saver
        save_best_model = SaveBestModel() if self.save_model is True else None
        # Loop over epochs
        for self.epoch in range(self.max_epochs):
            train_metrics = self._train_posthoc_epoch()
            test_metrics = self._test_posthoc_epoch()
            if isinstance(self.scheduler_arthur, torch.optim.lr_scheduler._LRScheduler):
                self.scheduler_arthur.step()
            # Print results
            train_metrics.print_results("Train")
            test_metrics.print_results("Test")
            # Log epoch results
            if isinstance(self.logger, ModuleType):
                self.logger.log(
                    {
                        "train/loss": train_metrics.average("loss"),
                        "train/completeness": train_metrics.average("completeness"),
                        "completeness_continuous": train_metrics.average("completeness_continuous"),
                        "train/soundness": train_metrics.average("soundness"),
                        "test/loss": test_metrics.average("loss"),
                        "test/completeness": test_metrics.average("completeness"),
                        "test/soundness": test_metrics.average("soundness"),
                        "test/iou": test_metrics.average("iou"),
                    },
                    step=self.epoch,
                )
            # Save model and optimizer states
            if self.save_model and isinstance(save_best_model, SaveBestModel):
                model_state = {"model_state_dict_arthur": self.arthur.state_dict()}
                optimizer_state = {"optimizer_state_dict_arthur": self.optimizer_arthur.state_dict()}
                save_best_model(
                    test_metrics.average("loss"),
                    self.epoch,
                    model_state,
                    optimizer_state,
                    log_dir=f"runs/min_max_train/merlin-arthur-framework/{self.dataset}/{self.approach}_approach/mask{self.mask_size}/state_dict/{self.arthur.__class__.__name__}_{self.dataset}_masksize_{self.mask_size}_Params_{self.num_param:1.2E}_gamma_{self.gamma}_lr_{self.lr:1E}.pth",
                )

        return True

    def _train_posthoc_epoch(self) -> MetricsCollector:
        """Train Arthur on MinMax training with PostHoc as Merlin and Morgana as Mask Optimization.

        Returns:
            MetricsCollector: MetricsCollector object with metrics for the epoch.

        Raises:
            AssertionError: If Arthur's optimizer is not set.
            AssertionError: If Merlin is not set.
            AssertionError: If Morgana is not set.
        """
        assert self.arthur is not None, "Arthur must be set"
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        assert self.merlin is not None, "Merlin must be set."
        assert self.morgana is not None, "Morgana must be set."
        assert self.mask_size is not None, "Mask size must be set."
        assert self.segmentation_method is not None, "Segmentation method must be set."
        # Initialize MetricsCollector object
        metrics_collector = MetricsCollector("loss", "completeness", "completeness_continuous", "soundness")
        # Initialize dataloader before each epoch to include newest masks for Merlin and Morgana
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)  # type: ignore
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch in pbar:  # x_input, y_true, mask_merlin, mask_morgana, idx
            # Send to GPU
            x_input, y_true = batch[0].to(self.device), batch[1].to(self.device)
            mask_merlin, mask_morgana = batch[2].to(self.device), batch[3].to(self.device)
            idx = batch[-1]

            # Initialize inputs for Merlin and Morgana
            x_merlin = x_input
            y_true_merlin = y_true
            x_morgana = x_input
            y_true_morgana = y_true

            # If only on class is set, only use the samples of that class
            if self.morgana.only_on_class is True:
                # Merlin only for label 1
                x_merlin = x_input[y_true == 1]
                y_true_merlin = y_true[y_true == 1]
                mask_merlin = mask_merlin[y_true == 1]
            if self.morgana.only_on_class is True:
                # Morgana only for label 0
                x_morgana = x_input[y_true == 0]
                y_true_morgana = y_true[y_true == 0]
                mask_morgana = mask_morgana[y_true == 0]

            if self.original_image_ratio is not None:
                # Calculate the index up to which to keep the original images
                cutoff_idx_merlin = int(self.original_image_ratio * x_merlin.shape[0])
                cutoff_idx_morgana = int(self.original_image_ratio * x_morgana.shape[0])
                # Keep only the first 'cutoff_index' items along the first dimension (usually the batch dimension)
                x_merlin_original = x_merlin[:cutoff_idx_merlin]
                x_merlin = x_merlin[cutoff_idx_merlin:]
                x_morgana_orignal = x_morgana[:cutoff_idx_morgana]
                x_morgana = x_morgana[cutoff_idx_morgana:]
                # Labels
                y_true_merlin_original = y_true_merlin[:cutoff_idx_merlin]
                y_true_morgana_original = y_true_morgana[:cutoff_idx_morgana]
                y_true_merlin = y_true_merlin[cutoff_idx_merlin:]
                y_true_morgana = y_true_morgana[cutoff_idx_morgana:]
                # Masks
                mask_merlin = mask_merlin[cutoff_idx_merlin:]
                mask_morgana = mask_morgana[cutoff_idx_morgana:]

            # Merlin's (Continuous) Posthoc Explanation
            mask_ggc_continuous = self.ggc.attribute(x_merlin).sum(dim=1, keepdim=True)
            mask_ggc = self.merlin.segment(mask_ggc_continuous, method=self.segmentation_method)

            # Morgana's still uses Mask Optimization Approach
            mask_morgana = self.morgana(x_morgana, y_true_morgana, self.arthur, mask_morgana)  # type: ignore
            mask_morgana = self.morgana.segment(mask_morgana, method=self.segmentation_method)

            with torch.autocast(device_type=self.device.type, enabled=self.merlin.use_amp):  # type: ignore
                # Apply mask to input
                x_masked_merlin = self.merlin.apply_mask(x_merlin, mask_ggc)
                x_masked_merlin_continuous = self.merlin.apply_mask(x_merlin, mask_ggc_continuous)  # type: ignore
                x_masked_morgana = self.morgana.apply_mask(x_morgana, mask_morgana)
                # Blend in original images
                if self.original_image_ratio is not None:
                    x_masked_merlin = torch.cat((x_merlin_original, x_masked_merlin), dim=0)  # type: ignore
                    x_masked_morgana = torch.cat((x_morgana_orignal, x_masked_morgana), dim=0)  # type: ignore
                    y_true_merlin = torch.cat((y_true_merlin_original, y_true_merlin), dim=0)  # type: ignore
                    y_true_morgana = torch.cat((y_true_morgana_original, y_true_morgana), dim=0)  # type: ignore
                    # Continuous case
                    x_masked_merlin_continuous = torch.cat((x_merlin_original, x_masked_merlin_continuous), dim=0)  # type: ignore

                # Arthur's response to masked input
                logits_merlin = self.arthur(x_masked_merlin)
                logits_merlin_continuous = self.arthur(x_masked_merlin_continuous)
                logits_morgana = self.arthur(x_masked_morgana)

                # Convert tensors to appropriate format if binary classification is used
                if self.binary_classification is True:
                    logits_merlin = logits_merlin.squeeze(1)
                    y_true_merlin = y_true_merlin.float()
                    logits_morgana = logits_morgana.squeeze(1)
                    y_true_morgana = y_true_morgana.float()

                # Loss
                if self.binary_classification is True and self.merlin.optimize_probabilities is True:
                    arthur_loss = (
                        1
                        - torch.mean(torch.sigmoid(logits_merlin))
                        - self.gamma * (1 - torch.mean(torch.sigmoid(logits_morgana)))
                    )
                else:
                    merlin_loss = self.merlin.criterion(logits_merlin, y_true_merlin)
                    morgana_loss = self.morgana.criterion(logits_morgana, y_true_morgana)
                    arthur_loss = merlin_loss + self.gamma * morgana_loss
            # Backpropagation
            self.optimizer_arthur.zero_grad()
            arthur_loss.backward()
            self.optimizer_arthur.step()

            if self.original_image_ratio is not None:
                logits_merlin = logits_merlin[cutoff_idx_merlin:]  # type: ignore
                logits_morgana = logits_morgana[cutoff_idx_morgana:]  # type: ignore
                logits_merlin_continuous = logits_merlin_continuous[cutoff_idx_merlin:]  # type: ignore
                y_true_merlin = y_true_merlin[cutoff_idx_merlin:]  # type: ignore
                y_true_morgana = y_true_morgana[cutoff_idx_morgana:]  # type: ignore

            # Metrics
            completeness = get_accuracy(
                logits_merlin, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            completeness_continuous = get_accuracy(
                logits_merlin_continuous, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            soundness = get_accuracy(
                logits_morgana, y_true_morgana, mode="morgana", binary_classification=self.binary_classification
            )
            metrics_collector.add(
                loss=arthur_loss.item(),
                completeness=completeness,
                completeness_continuous=completeness_continuous,
                soundness=soundness,
            )
            # Monitoring
            pbar.set_postfix(
                {
                    "Loss": metrics_collector.average("loss"),
                    "Compl.": metrics_collector.average("completeness"),
                    "completeness_continuous": metrics_collector.average("completeness_continuous"),
                    "Sound.": metrics_collector.average("soundness"),
                }
            )

        return metrics_collector

    def _test_posthoc_epoch(self) -> MetricsCollector:
        assert self.arthur is not None, "Arthur must be set"
        assert isinstance(self.optimizer_arthur, torch.optim.Optimizer), "Arthur Optimizer not set."
        assert self.merlin is not None, "Merlin must be set."
        assert self.morgana is not None, "Morgana must be set."
        assert self.mask_size is not None, "Mask size must be set."
        assert self.segmentation_method is not None, "Segmentation method must be set."
        self.optimizer_arthur.zero_grad()
        # Initialize MetricsCollector object
        metrics_collector = MetricsCollector(
            "loss", "completeness", "completeness_continuous", "soundness", "iou", "pg", "rma", "rra"
        )
        # Initialize dataloader before each epoch to include newest masks for Merlin and Morgana
        pbar = tqdm(self.test_loader, desc=f"Test Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch in pbar:  # x_input, y_true, mask_merlin, mask_morgana, idx
            # Send to GPU
            x_input, y_true = batch[0].to(self.device), batch[1].to(self.device)
            mask_merlin, mask_morgana = batch[2].to(self.device), batch[3].to(self.device)
            idx = batch[-1]

            # Initialize inputs for Merlin and Morgana
            x_merlin = x_input
            y_true_merlin = y_true
            x_morgana = x_input
            y_true_morgana = y_true

            if self.calculate_iou is True:
                batch_left = batch[4].to(self.device)
                batch_top = batch[5].to(self.device)
                batch_width = batch[6].to(self.device)
                batch_height = batch[7].to(self.device)

                batch_left = batch_left[y_true == 1]
                batch_top = batch_top[y_true == 1]
                batch_width = batch_width[y_true == 1]
                batch_height = batch_height[y_true == 1]

            # If only on class is set, only use the samples of that class
            if self.morgana.only_on_class is True:
                # Merlin only for label 1
                x_merlin = x_input[y_true == 1]
                y_true_merlin = y_true[y_true == 1]
                mask_merlin = mask_merlin[y_true == 1]
            if self.morgana.only_on_class is True:
                # Morgana only for label 0
                x_morgana = x_input[y_true == 0]
                y_true_morgana = y_true[y_true == 0]
                mask_morgana = mask_morgana[y_true == 0]

            if self.original_image_ratio is not None:
                # Calculate the index up to which to keep the original images
                cutoff_idx_merlin = int(self.original_image_ratio * x_merlin.shape[0])
                cutoff_idx_morgana = int(self.original_image_ratio * x_morgana.shape[0])
                # Keep only the first 'cutoff_index' items along the first dimension (usually the batch dimension)
                x_merlin_original = x_merlin[:cutoff_idx_merlin]
                x_merlin = x_merlin[cutoff_idx_merlin:]
                x_morgana_orignal = x_morgana[:cutoff_idx_morgana]
                x_morgana = x_morgana[cutoff_idx_morgana:]
                # Labels
                y_true_merlin_original = y_true_merlin[:cutoff_idx_merlin]
                y_true_morgana_original = y_true_morgana[:cutoff_idx_morgana]
                y_true_merlin = y_true_merlin[cutoff_idx_merlin:]
                y_true_morgana = y_true_morgana[cutoff_idx_morgana:]
                # Masks
                mask_merlin = mask_merlin[cutoff_idx_merlin:]
                mask_morgana = mask_morgana[cutoff_idx_morgana:]

            # Merlin's (Continuous) Posthoc Explanation
            mask_ggc_continuous = self.ggc.attribute(x_merlin).sum(dim=1, keepdim=True)
            mask_ggc = self.merlin.segment(mask_ggc_continuous, method=self.segmentation_method)

            # Morgana's still uses Mask Optimization Approach
            mask_morgana = self.morgana(x_morgana, y_true_morgana, self.arthur, mask_morgana)  # type: ignore
            mask_morgana = self.morgana.segment(mask_morgana, method=self.segmentation_method)

            with torch.autocast(device_type=self.device.type, enabled=self.merlin.use_amp):  # type: ignore
                # Test Iou calculation
                if self.calculate_iou is True:
                    # Intersection over Union (IoU)
                    iou_batch = IoUCalculator.iou_batch(
                        mask_ggc, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    iou_batch_mean = torch.mean(iou_batch)  # type: ignore
                    # Pointing Game (PG)
                    pointing_game_batch = PointingGameCalculator.pointing_game_batch(
                        mask_ggc_continuous, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    pointing_game_batch_mean = torch.mean(pointing_game_batch)  # type: ignore
                    # Relevant Mask Accuracy (RMA)
                    rma_batch = RMACalculator.rma_batch(
                        mask_ggc_continuous, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rma_batch_mean = torch.mean(rma_batch)  # type: ignore
                    # Relevant Rank Accuracy (RRA)
                    rra_batch = RRACalculator.rra_batch(
                        mask_ggc_continuous, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rra_batch_mean = torch.mean(rra_batch)  # type: ignore

                # Apply mask to input
                x_masked_merlin = self.merlin.apply_mask(x_merlin, mask_ggc)
                x_masked_merlin_continuous = self.merlin.apply_mask(x_merlin, mask_ggc_continuous)  # type: ignore
                x_masked_morgana = self.morgana.apply_mask(x_morgana, mask_morgana)
                # Blend in original images
                if self.original_image_ratio is not None:
                    x_masked_merlin = torch.cat((x_merlin_original, x_masked_merlin), dim=0)  # type: ignore
                    x_masked_morgana = torch.cat((x_morgana_orignal, x_masked_morgana), dim=0)  # type: ignore
                    y_true_merlin = torch.cat((y_true_merlin_original, y_true_merlin), dim=0)  # type: ignore
                    y_true_morgana = torch.cat((y_true_morgana_original, y_true_morgana), dim=0)  # type: ignore
                    # Continuous case
                    x_masked_merlin_continuous = torch.cat((x_merlin_original, x_masked_merlin_continuous), dim=0)  # type: ignore

                # Arthur's response to masked input
                logits_merlin = self.arthur(x_masked_merlin)
                logits_merlin_continuous = self.arthur(x_masked_merlin_continuous)
                logits_morgana = self.arthur(x_masked_morgana)

                # Convert tensors to appropriate format if binary classification is used
                if self.binary_classification is True:
                    logits_merlin = logits_merlin.squeeze(1)
                    y_true_merlin = y_true_merlin.float()
                    logits_morgana = logits_morgana.squeeze(1)
                    y_true_morgana = y_true_morgana.float()

                # Loss
                if self.binary_classification is True and self.merlin.optimize_probabilities is True:
                    arthur_loss = (
                        1
                        - torch.mean(torch.sigmoid(logits_merlin))
                        - self.gamma * (1 - torch.mean(torch.sigmoid(logits_morgana)))
                    )
                else:
                    merlin_loss = self.merlin.criterion(logits_merlin, y_true_merlin)
                    morgana_loss = self.morgana.criterion(logits_morgana, y_true_morgana)
                    arthur_loss = merlin_loss + self.gamma * morgana_loss

            if self.original_image_ratio is not None:
                logits_merlin = logits_merlin[cutoff_idx_merlin:]  # type: ignore
                logits_morgana = logits_morgana[cutoff_idx_morgana:]  # type: ignore
                logits_merlin_continuous = logits_merlin_continuous[cutoff_idx_merlin:]  # type: ignore
                y_true_merlin = y_true_merlin[cutoff_idx_merlin:]  # type: ignore
                y_true_morgana = y_true_morgana[cutoff_idx_morgana:]  # type: ignore

            # Metrics
            completeness = get_accuracy(
                logits_merlin, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            completeness_continuous = get_accuracy(
                logits_merlin_continuous, y_true_merlin, mode="merlin", binary_classification=self.binary_classification
            )
            soundness = get_accuracy(
                logits_morgana, y_true_morgana, mode="morgana", binary_classification=self.binary_classification
            )

            metrics_collector.add(
                loss=arthur_loss.item(),
                completeness=completeness,
                completeness_continuous=completeness_continuous,
                soundness=soundness,
                iou=iou_batch_mean.item() if self.calculate_iou is True else 0,  # type: ignore
                pg=pointing_game_batch_mean.item() if self.calculate_iou is True else 0,  # type: ignore
                rma=rma_batch_mean.item() if self.calculate_iou is True else 0,  # type: ignore
                rra=rra_batch_mean.item() if self.calculate_iou is True else 0,  # type: ignore
            )
            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": metrics_collector.average("loss"),
                    "Compl.": metrics_collector.average("completeness"),
                    "Compl. (cont.)": metrics_collector.average("completeness_continuous"),
                    "Sound.": metrics_collector.average("soundness"),
                    "iou": metrics_collector.average("iou"),
                    "pg": metrics_collector.average("pg"),
                    "rma": metrics_collector.average("rma"),
                    "rra": metrics_collector.average("rra"),
                }
            )
        if isinstance(self.logger, ModuleType) and self.data_type == "image":
            if self.dataset == "SVHN":
                # Add channel dimension to Feature Selector output in case of SVHN
                mask_ggc = mask_ggc.repeat(1, 3, 1, 1)  # type: ignore
                mask_ggc_continuous = mask_ggc_continuous.repeat(1, 3, 1, 1)  # type: ignore
                mask_morgana = mask_morgana.repeat(1, 3, 1, 1)  # type: ignore
            batch_size_merlin = x_merlin.size(0)  # type: ignore
            # Log Feature Selections
            image_grid = [x_merlin[:batch_size_merlin], mask_ggc_continuous[:batch_size_merlin], mask_ggc[:batch_size_merlin], x_masked_merlin[:30, :3, :, :], x_morgana[:30], mask_morgana[:30], x_masked_morgana[:30, :3, :, :]]  # type: ignore
            image_grid = torch.cat(image_grid, dim=0)
            image_grid = utils.make_grid(image_grid, nrow=batch_size_merlin, normalize=True, scale_each=True)
            images = wandb.Image(
                image_grid,
                caption="From Top to bottom: Input, Merlin Output, Merlin Segmented, Merlin Masked, Morgana Output, Morgana Segmented, Morgana Masked",
            )
            wandb.log({"Feature Selections": images}, commit=False)

            # Monitoring
            pbar.set_postfix(
                {
                    "Loss": metrics_collector.average("loss"),
                    "Compl.": metrics_collector.average("completeness"),
                    "completeness_continuous": metrics_collector.average("completeness_continuous"),
                    "Sound.": metrics_collector.average("soundness"),
                }
            )

        return metrics_collector

    def train(self):
        if self.approach == "regular":
            self.regular_train(max_epochs=self.epochs)
        elif self.approach in ("sfw", "mask_optimization"):
            assert self.gamma is not None, "Gamma must be specified for SFW and Mask Optimization"
            self.train_min_max_with_mask_optimization(max_epochs=self.epochs, gamma=self.gamma)
        elif self.approach == "posthoc":
            assert self.gamma is not None, "Gamma must be specified for SFW and Mask Optimization"
            self.train_min_max_with_posthoc(max_epochs=self.epochs, gamma=self.gamma)
        elif self.approach == "unet":
            assert self.gamma is not None, "Gamma must be specified for SFW and Mask Optimization"
            assert isinstance(self.steps_morgana, int), "Steps Morgana must be specified for SFW and Mask Optimization"
            self.train_min_max(max_epochs=self.epochs, gamma=self.gamma, steps_morgana=self.steps_morgana)

    def compare_xai_tool(self):
        """Compares explainations obtained with post-hoc xai methods."""
        assert self.arthur is not None, "Arthur model must be initialized."
        assert self.optimizer_arthur is not None, "Optimizer for Arthur must be initialized."
        self.print_trainer_details()

        self.retrained_arthur = False
        if self.retrained_arthur is True:
            state_dict = torch.load(
                "runs/min_max_train/merlin-arthur-framework/SVHN/unet_approach/mask512/state_dict/minmax_targets01_512_Params_6.68E+07_gamma1.0.pth",
                map_location=self.device,
            )
            self.arthur.load_state_dict(state_dict["model_state_dict_arthur"])
        self.arthur.eval()

        s = Saliency(self.arthur)
        ig = IntegratedGradients(self.arthur)
        gs = GradientShap(self.arthur)
        gb = GuidedBackprop(self.arthur)
        ggc = GuidedGradCam(self.arthur, self.arthur.layer4[-1].conv3)  # type: ignore

        # NOTE: Some additional methods we could use
        # deconvultion = Deconvolution(self.arthur)
        # lime = Lime(self.arthur)
        # lrp = LRP(self.arthur)
        # occlusion = Occlusion(self.arthur)
        # deep_lift_shap = DeepLiftShap(self.arthur)
        # deep_lift = DeepLift(self.arthur)

        self.max_epochs = 1
        loss_criterion = (
            torch.nn.BCEWithLogitsLoss() if self.binary_classification is True else torch.nn.CrossEntropyLoss()
        )

        metrics_collector = MetricsCollector("test_loss_avg", "test_acc_avg")
        xai_methods = [gb, ggc, s, ig, gs]
        iou_metric_names = ["iou_" + method.__class__.__name__ for method in xai_methods]
        pg_metric_names = ["pg_" + method.__class__.__name__ for method in xai_methods]
        rma_metric_names = ["rma_" + method.__class__.__name__ for method in xai_methods]
        rra_metric_names = ["rra_" + method.__class__.__name__ for method in xai_methods]

        metrics_collector.add_metric(*iou_metric_names, *pg_metric_names, *rma_metric_names, *rra_metric_names)

        for self.epoch in range(self.max_epochs):
            assert self.arthur is not None, "Arthur model must be initialized."

            # Set model to eval mode
            self.arthur.eval()

            # Initialize metrics
            test_loss_avg = Averager()
            test_acc_avg = Averager()

            continuous_explanations_history = []
            topk_explanations_history = []
            x_masked_history = []
            x_input_history = []

            pbar = tqdm(self.test_loader, desc=f"Test Epoch {self.epoch + 1}/{self.max_epochs}")
            for i, batch in enumerate(pbar):  # Test loop
                x_input = batch[0].to(self.device)
                y_true = batch[1].to(self.device)
                y_true_all = y_true

                # Only for label 1
                x_input = x_input[y_true == 1]
                y_true = y_true[y_true == 1]

                # Skip batches where the filtered tensor has one or no entries (due to low batch size)
                if len(x_input) <= 1:
                    print("Skipping batch due to insufficient entries.")
                    continue  # Skip to the next iteration of your loop

                batch_left = batch[4].to(self.device)
                batch_top = batch[5].to(self.device)
                batch_width = batch[6].to(self.device)
                batch_height = batch[7].to(self.device)

                batch_left = batch_left[y_true_all == 1]
                batch_top = batch_top[y_true_all == 1]
                batch_width = batch_width[y_true_all == 1]
                batch_height = batch_height[y_true_all == 1]

                # Inference
                output = self.arthur(x_input)

                if self.binary_classification is True:
                    # Adapt tensors for BCEWithLogitsLoss
                    output = output.squeeze()
                    y_true = y_true.float()

                # Metrics
                loss = loss_criterion(output, y_true)
                accuracy = categorical_accuracy(output, y_true, self.binary_classification)

                # Log metrics
                test_acc_avg.add(accuracy)
                test_loss_avg.add(loss.item())
                metrics_collector.add(test_loss_avg=test_loss_avg.result(), test_acc_avg=test_acc_avg.result())

                self.arthur.zero_grad()

                # # Saliency Map
                # saliency_grads = saliency.attribute(x_input)
                # # Integrated Gradients
                # attr_ig, delta = integrated_gradients.attribute(
                #     x_input, baselines=x_input * 0, return_convergence_delta=True, internal_batch_size=1
                # )
                # # Gradient Shap
                # attr_gs = gradient_shap.attribute(x_input, baselines=x_input * 0)
                if i == 0:
                    # clone x_input
                    x_input_cloned = x_input.clone()
                    x_input_history.append(x_input_cloned.detach().cpu())
                for xai_method in xai_methods:
                    if xai_method == gs:
                        # Gradient Shap needs baselines
                        continuous_explanation = xai_method.attribute(x_input, baselines=x_input * 0).sum(
                            dim=1, keepdim=True
                        )
                    elif xai_method == ig:
                        # IG needs smaller batch size
                        continuous_explanation = xai_method.attribute(x_input, internal_batch_size=1).sum(
                            dim=1, keepdim=True
                        )
                    else:
                        # All other methods do not need modifications
                        continuous_explanation = xai_method.attribute(x_input).sum(dim=1, keepdim=True)

                    xai_method_name = xai_method.__class__.__name__

                    # Apply top-k filter
                    topk = self.merlin.segment(continuous_explanation, method="topk")
                    x_masked = self.merlin.apply_mask(x_input, topk).detach().cpu()

                    # Free GPU
                    continuous_explanation = continuous_explanation.detach().cpu()
                    topk = topk.detach().cpu()

                    # Calculate Metrics
                    # Intersection over Union (IoU)
                    iou_batch = IoUCalculator.iou_batch(
                        topk, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    iou_batch_mean = torch.mean(iou_batch)  # type: ignore
                    # Pointing Game (PG)
                    pointing_game_batch = PointingGameCalculator.pointing_game_batch(
                        continuous_explanation, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    pointing_game_batch_mean = torch.mean(pointing_game_batch)  # type: ignore
                    # Relevant Mask Accuracy (RMA)
                    rma_batch = RMACalculator.rma_batch(
                        continuous_explanation, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rma_batch_mean = torch.mean(rma_batch)  # type: ignore
                    # Relevant Rank Accuracy (RRA)
                    rra_batch = RRACalculator.rra_batch(
                        continuous_explanation, batch_left, batch_top, batch_width, batch_height  # type: ignore
                    )
                    rra_batch_mean = torch.mean(rra_batch)  # type: ignore

                    metrics = {
                        f"iou_{xai_method_name}": iou_batch_mean.item(),
                        f"pg_{xai_method_name}": pointing_game_batch_mean.item(),
                        f"rma_{xai_method_name}": rma_batch_mean.item(),
                        f"rra_{xai_method_name}": rra_batch_mean.item(),
                    }

                    metrics_collector.add(**metrics)

                    topk = topk.detach().cpu()

                    # save explanations only for first batch
                    if i == 0:
                        x_masked_history.append(x_masked)
                        # Repeat dimensions for visualization
                        continuous_explanation = continuous_explanation.repeat(1, 3, 1, 1)
                        topk = topk.repeat(1, 3, 1, 1)
                        # Save explanations
                        continuous_explanations_history.append(continuous_explanation)
                        topk_explanations_history.append(topk)

                    # Free up memory
                    del continuous_explanation, topk, x_masked
                    del iou_batch, iou_batch_mean
                    del pointing_game_batch, pointing_game_batch_mean
                    del rma_batch, rma_batch_mean
                    del rra_batch, rra_batch_mean

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                wandb.log(
                    {
                        "Test Loss Avg": metrics_collector.average("test_loss_avg"),
                        "Test Acc Avg": metrics_collector.average("test_acc_avg"),
                        "IoU (GB)": metrics_collector.average("iou_GuidedBackprop"),
                        "IoU (GGC)": metrics_collector.average("iou_GuidedGradCam"),
                        "IoU (S)": metrics_collector.average("iou_Saliency"),
                        "IoU (IG)": metrics_collector.average("iou_IntegratedGradients"),
                        "IoU (GS)": metrics_collector.average("iou_GradientShap"),
                        "Pointing Game (GB)": metrics_collector.average("pg_GuidedBackprop"),
                        "Pointing Game (GGC)": metrics_collector.average("pg_GuidedGradCam"),
                        "Pointing Game (S)": metrics_collector.average("pg_Saliency"),
                        "Pointing Game (IG)": metrics_collector.average("pg_IntegratedGradients"),
                        "Pointing Game (GS)": metrics_collector.average("pg_GradientShap"),
                        "RMA (GB)": metrics_collector.average("rma_GuidedBackprop"),
                        "RMA (GGC)": metrics_collector.average("rma_GuidedGradCam"),
                        "RMA (S)": metrics_collector.average("rma_Saliency"),
                        "RMA (IG)": metrics_collector.average("rma_IntegratedGradients"),
                        "RMA (GS)": metrics_collector.average("rma_GradientShap"),
                        "RRA (GB)": metrics_collector.average("rra_GuidedBackprop"),
                        "RRA (GGC)": metrics_collector.average("rra_GuidedGradCam"),
                        "RRA (S)": metrics_collector.average("rra_Saliency"),
                        "RRA (IG)": metrics_collector.average("rra_IntegratedGradients"),
                        "RRA (GS)": metrics_collector.average("rra_GradientShap"),
                    }
                )

                pbar.set_postfix(
                    {
                        "Test Loss Avg": metrics_collector.average("test_loss_avg"),
                        "Test Acc Avg": metrics_collector.average("test_acc_avg"),
                        "IoU (GB)": metrics_collector.average("iou_GuidedBackprop"),
                        "IoU (GGC)": metrics_collector.average("iou_GuidedGradCam"),
                        "IoU (S)": metrics_collector.average("iou_Saliency"),
                        "IoU (IG)": metrics_collector.average("iou_IntegratedGradients"),
                        "IoU (GS)": metrics_collector.average("iou_GradientShap"),
                        "Pointing Game (GB)": metrics_collector.average("pg_GuidedBackprop"),
                        "Pointing Game (GGC)": metrics_collector.average("pg_GuidedGradCam"),
                        "Pointing Game (S)": metrics_collector.average("pg_Saliency"),
                        "Pointing Game (IG)": metrics_collector.average("pg_IntegratedGradients"),
                        "Pointing Game (GS)": metrics_collector.average("pg_GradientShap"),
                        "RMA (GB)": metrics_collector.average("rma_GuidedBackprop"),
                        "RMA (GGC)": metrics_collector.average("rma_GuidedGradCam"),
                        "RMA (S)": metrics_collector.average("rma_Saliency"),
                        "RMA (IG)": metrics_collector.average("rma_IntegratedGradients"),
                        "RMA (GS)": metrics_collector.average("rma_GradientShap"),
                        "RRA (GB)": metrics_collector.average("rra_GuidedBackprop"),
                        "RRA (GGC)": metrics_collector.average("rra_GuidedGradCam"),
                        "RRA (S)": metrics_collector.average("rra_Saliency"),
                        "RRA (IG)": metrics_collector.average("rra_IntegratedGradients"),
                        "RRA (GS)": metrics_collector.average("rra_GradientShap"),
                    }
                )

            # Convert lists to tensors
            continuous_explanations_history = torch.cat(continuous_explanations_history, dim=0)
            topk_explanations_history = torch.cat(topk_explanations_history, dim=0)
            x_masked_history = torch.cat(x_masked_history, dim=0)
            x_input_history = torch.cat(x_input_history, dim=0)
            # Create the image grid using the modified tensors
            image_grid = [x_input_history, continuous_explanations_history, topk_explanations_history, x_masked_history]
            image_grid = torch.cat(image_grid, dim=0)
            image_grid = utils.make_grid(image_grid, nrow=len(x_input_history), normalize=True, scale_each=True)

            # Create a wandb.Image and log it
            images = wandb.Image(image_grid, caption="Attributions from Saliency")
            wandb.log({"Attributions": images})
            self._print_epoch_results(test_acc_avg, test_loss_avg, test_acc_avg, test_loss_avg)

        return True
