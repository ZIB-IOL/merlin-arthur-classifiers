import itertools
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader
from tqdm import tqdm

import feature_selection
from models import SimpleNet


class MetricsCollector:
    def __init__(self, *metrics) -> None:
        assert all(
            isinstance(metric, str) for metric in metrics
        ), f"Expected all metrics to be of type `str`, got {type(metrics)}"
        # Tuple of strings
        self.metrics = list(metrics)
        self.metric_dict = {}
        # Initializes the averager for each passed metric
        for metric_name in self.metrics:
            self.metric_dict[metric_name] = Averager()

    @torch.no_grad()
    def add(self, **kwargs) -> None:
        """Adds values to Averager for each metric corresponding to the keywords"""
        for metric_name, val in kwargs.items():
            self.metric_dict[metric_name].add(val)

    def add_metric(self, *new_metrics: str) -> None:
        """Adds a new metric to be collected."""
        for new_metric in new_metrics:
            if new_metric not in self.metrics:
                self.metrics.append(new_metric)
                self.metric_dict[new_metric] = Averager()
            else:
                print(f"Metric {new_metric} already exists.")

    def print_results(self, prefix) -> None:
        """Prints the current average of each metric"""
        print(10 * "---")
        for metric_name in self.metrics:
            print(f"{prefix} {metric_name}: {self.metric_dict[metric_name].result():1.2f}")

    @torch.no_grad()
    def average(self, metric: str) -> float:
        """Returns current average of metric"""
        assert (
            metric in self.metric_dict.keys()
        ), f"Passed argument is not a key of the corresponding dictionary, got `{metric}`"
        return self.metric_dict[metric].result()

    def __repr__(self):
        return f"The metric that are collected are {self.metrics}"


class EntropyCalculator:
    """Class to calculate conditional entropy with respect to the features of a feature selector.

    Depending on the feature selector, the conditional entropy is calculated as follows:
        - Merlin: H(y|x_masked) = - sum_x p(y|x_masked) * log(p(y|x_masked)).

    Args:
        feature_selector (FeatureSelector): Feature selector to calculate entropy with.
        dataloader (DataLoader): Dataloader to use for entropy calculation.
        mask_size (int): Size of the mask to use for entropy calculation.
        num_targets (int): Number of targets in the dataset.

    Returns:
        None
    """

    def __init__(
        self,
        arthur: torch.nn.Module,
        feature_selector,
        morgana,
        segmentation_method: str,
        inner_dataloader,
        outer_dataloader: DataLoader,
        mask_size: int,
        num_targets: int,
        **kwargs,
    ) -> None:
        # Attributes
        self.arthur = arthur.eval()
        self.feature_selector = feature_selector.eval()
        self.morgana = morgana.eval()
        self.segmentation_method = segmentation_method
        self.inner_dataloader = inner_dataloader
        self.outer_dataloader = outer_dataloader
        self.mask_size = mask_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_targets = num_targets
        self.brute_force_merlin = kwargs.get("brute_force_merlin", None)
        self.brute_force_morgana = kwargs.get("brute_force_morgana", None)
        # Assertions
        assert isinstance(
            self.feature_selector, feature_selection.FeatureSelector
        ), f"FeatureSelector expected, got {type(self.feature_selector)}"
        assert isinstance(self.segmentation_method, str), f"str expected, got {type(self.segmentation_method)}"
        assert isinstance(self.outer_dataloader, DataLoader), f"Dataloader expected, got {type(self.outer_dataloader)}"
        assert isinstance(self.mask_size, int), f"int expected, got {type(self.mask_size)}"
        assert isinstance(self.device, torch.device), f"torch.device expected, got {type(self.device)}"
        assert isinstance(self.num_targets, int), f"int expected, got {type(self.num_targets)}"

    def calc_conditional_entropy(
        self,
        tol: float = 1e-7,
    ) -> MetricsCollector:
        """Returns the average precision, conditional entropy and mean images per feature of the dataset.

        The conditional entropy is calculated with respect to the features of the feature selector.
        The conditional entropy is calculated as follows: H(y|x_masked) = - sum_x p(y|x_masked) * log(p(y|x_masked)).
        It is is calculated for each feature chosen by Merlin and averaged over all features.

        Args:
            tol (float, optional): Tolerance for feature comparison. Defaults to 1e-7.

        Returns:
            MetricsCollector: MetricsCollector object containing the
                (1) conditional entropy,
                (2) average precision,
                (3) mean images per feature.

        """
        # Assertions
        assert isinstance(tol, float), f"float expected, got {type(tol)}"
        # Initialize MetricsCollector
        metrics_collector = MetricsCollector(
            "conditional_entropy",
            "average_precision",
            "mean_datapoints_per_feature",
            "class_completeness_0",
            "class_completeness_1",
            "class_soundness_0",
            "class_soundness_1",
        )
        grid = torch.Tensor()
        mask_collection = None
        if self.brute_force_merlin and self.brute_force_morgana:
            x_input, *_ = next(iter(self.outer_dataloader))
            num_features = x_input.shape[1]
            array = np.array(list(itertools.combinations(range(num_features), self.mask_size)))
            grid = np.zeros((len(array), num_features), dtype="float32")
            grid[np.arange(len(array))[None].T, array] = 1
            grid = torch.tensor(grid)

        # Loop over outer dataloader, but ignore loading masks if CustomMNIST is used
        pbar = tqdm(self.outer_dataloader, desc="Calculating conditional entropy")
        for batch in pbar:
            x_input = batch[0].to(self.device)
            y_true = batch[1].to(self.device)

            # Condition on a single class, e.g. 2 vs 4
            target_class = 0
            morgana_target_class = 1

            # Condition on a single class
            x_merlin = x_input[y_true == target_class]
            y_true_merlin = y_true[y_true == target_class]
            x_morgana = x_input[y_true == morgana_target_class]
            y_true_morgana = y_true[y_true == morgana_target_class]
            if self.feature_selector.model == "sfw":
                init_mask = batch[2].to(self.device)
                init_mask_merlin = init_mask[y_true == target_class]
                init_mask_morgana = init_mask[y_true == morgana_target_class]

            # Get mask from Merlin
            if isinstance(self.feature_selector.model, SimpleNet):
                output = self.feature_selector(x_merlin)
                output_morgana = self.morgana(x_morgana)
                continuous_mask = self.feature_selector.normalize_l1(output, mask_size=self.mask_size)
                continuous_mask_morgana = self.feature_selector.normalize_l1(output_morgana, mask_size=self.mask_size)
                topk_mask = self.feature_selector.segment(continuous_mask, self.segmentation_method)
                topk_mask_morgana = self.morgana.segment(continuous_mask_morgana, self.segmentation_method)

            elif self.feature_selector.model == "sfw":
                if self.brute_force_merlin:
                    topk_mask = self.feature_selector.brute_force_search(x_input, y_true, self.arthur, grid)
                    mask_collection = (
                        topk_mask if mask_collection is None else torch.cat((mask_collection, topk_mask), dim=0)
                    )
                    mask_collection = torch.unique(mask_collection, dim=0)
                    topk_mask_morgana = self.morgana.brute_force_search(x_input, y_true, self.arthur, mask_collection)
                else:
                    continuous_mask = self.feature_selector(x_merlin, y_true_merlin, self.arthur, init_mask_merlin)  # type: ignore
                    continuous_mask_morgana = self.morgana(x_morgana, y_true_morgana, self.arthur, init_mask_morgana)  # type: ignore
                    topk_mask = self.feature_selector.segment(continuous_mask, self.segmentation_method)
                    topk_mask_morgana = self.morgana.segment(continuous_mask_morgana, self.segmentation_method)

            else:
                raise ValueError(f"Model {self.feature_selector.model} is not supported.")
            x_masked = self.feature_selector.apply_mask(x_merlin, topk_mask)  # type: ignore
            x_masked_morgana = self.morgana.apply_mask(x_morgana, topk_mask_morgana)  # type: ignore
            # # Test new occurence count
            # test = self.get_class_count(x_merlin, topk_mask, self.inner_dataloader)
            # Count number of samples with same feature as x_input for each class in the entire test dataset
            occurrence_per_class = self.count_occurrence(x_masked, topk_mask, tol=tol)  # type: ignore
            occurence = torch.sum(occurrence_per_class, dim=1, keepdim=True)  # shape: (batch_size, 1)

            # import matplotlib.pyplot as plt
            # # save plot of last th ree images, masks and masked images of the batch
            # plt.figure(figsize=(15, 12))
            # plt.subplot(3, 3, 1)
            # plt.imshow(x_merlin[-3].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 2)
            # plt.imshow(topk_mask[-3].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 3)
            # plt.imshow(x_masked[-3].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 4)
            # plt.imshow(x_merlin[-2].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 5)
            # plt.imshow(topk_mask[-2].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 6)
            # plt.imshow(x_masked[-2].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 7)
            # plt.imshow(x_merlin[-1].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 8)
            # plt.imshow(topk_mask[-1].cpu().squeeze().numpy())
            # plt.subplot(3, 3, 9)
            # plt.imshow(x_masked[-1].cpu().squeeze().numpy())
            # plt.savefig(f"test.pdf", transparent=True, bbox_inches="tight", pad_inches=0)

            ###### Very few instances have zero occurence, so we replace them with 1.0 to avoid division by zero.
            # Find where occurence is zero
            zero_idx = (occurence == 0).squeeze()
            # Replace zero occurrences with some default or minimal value to avoid division by zero.
            # You have to decide what default value makes sense in your context.
            occurence[zero_idx] = 1.0
            ######

            # Calculate class probabilities
            class_probs = occurrence_per_class / occurence
            # Metrics
            average_precision = self.precision_from_probs(class_probs, y_true_merlin).mean()
            conditional_entropy = self.entropy_from_probs(class_probs).mean()
            datapoints_per_feature = occurence.mean()
            # complteness and soundness
            per_image_completeness = self.get_per_image_accuracy(x_masked, y_true_merlin, mode="merlin")
            per_image_soundness = self.get_per_image_accuracy(x_masked_morgana, y_true_morgana, mode="morgana")
            if len(per_image_completeness[y_true_merlin == target_class]) != 0:
                class_completeness_0 = per_image_completeness[y_true_merlin == target_class].sum() / len(
                    per_image_completeness[y_true_merlin == target_class]
                )
                metrics_collector.add(class_completeness_0=class_completeness_0)
            # if len(per_image_completeness[y_true_merlin == 1]) != 0:
            #     class_completeness_1 = per_image_completeness[y_true == 1].sum() / len(
            #         per_image_completeness[y_true == 1]
            #     )
            #     metrics_collector.add(class_completeness_1=class_completeness_1)
            # if len(per_image_soundness[y_true_merlin == 1]) != 0:
            #     class_soundness_0 = per_image_soundness[y_true == 0].sum() / len(per_image_soundness[y_true == 0])
            #     metrics_collector.add(class_soundness_0=class_soundness_0)
            if len(per_image_soundness[y_true_morgana == morgana_target_class]) != 0:
                class_soundness_1 = per_image_soundness[y_true_morgana == morgana_target_class].sum() / len(
                    per_image_soundness[y_true_morgana == morgana_target_class]
                )
                metrics_collector.add(class_soundness_1=class_soundness_1)
            # Add metrics to MetricsCollector
            metrics_collector.add(
                average_precision=average_precision,
                conditional_entropy=conditional_entropy,
                mean_datapoints_per_feature=datapoints_per_feature,
            )

        average_precision = metrics_collector.average("average_precision")
        conditional_entropy = metrics_collector.average("conditional_entropy")
        mean_datapoints_per_feature = metrics_collector.average("mean_datapoints_per_feature")
        class_completeness_0 = metrics_collector.average("class_completeness_0")
        # class_completeness_1 = metrics_collector.average("class_completeness_1")
        # class_soundness_0 = metrics_collector.average("class_soundness_0")
        class_soundness_1 = metrics_collector.average("class_soundness_1")

        print(f"Average precision: {average_precision:.4f}")
        print(f"Conditional entropy: {conditional_entropy:.4f}")
        print(f"Mean Data Points per Feature: {mean_datapoints_per_feature:.4f}")
        print(f"Class Completeness {target_class}: {class_completeness_0:.4f}")
        # print(f"Class Completeness 1: {class_completeness_1:.4f}")
        # print(f"Class Soundness 0: {class_soundness_0:.4f}")
        print(f"Class Soundness {morgana_target_class}: {class_soundness_1:.4f}")

        return metrics_collector

    @torch.no_grad()
    def count_occurrence(
        self,
        x_masked: torch.Tensor,
        mask: torch.Tensor,
        tol: float = 1e-7,
    ) -> torch.Tensor:
        """For a given input, mask and dataloader, this function counts the number of elements in each class which share the same highlighted feature (up to a tolerance of `tol`).

        The highlighted feature is defined by the mask. The mask is applied to the input and the resulting masked input is compared to the masked inputs of the elements in the inner dataloader.
        If the norm of the difference between the masked inputs is smaller than `tol`, the element is counted as a sample with the same highlighted feature.

        Args:
            x_masked (torch.Tensor): Input image.
            mask (torch.Tensor): Mask.
            tol (float, optional): Tolerance. Defaults to 1e-7.

        Returns:
            torch.Tensor: Tensor with shape (batch_size, num_targets) containing the number of elements in each class which share the same highlighted feature.
        """
        # Assertions
        assert isinstance(
            self.feature_selector, feature_selection.FeatureSelector
        ), "Feature selector must be of type `FeatureSelector`."
        assert tol > 0.0, "Tolerance must be greater than zero."
        assert isinstance(x_masked, torch.Tensor), f"torch.Tensor expected, got {type(x_masked)}"
        assert isinstance(mask, torch.Tensor), f"torch.Tensor expected, got {type(mask)}"
        assert x_masked.shape[0] == mask.shape[0], "Batch size of input and mask must be equal."
        # Add channel dimension for UCI Census Data to match shape of MNIST
        if self.feature_selector.data_type == "categorical":
            x_masked = x_masked.unsqueeze(1)
            mask = mask.unsqueeze(1)
        # Initialize class count
        incidence_per_class = torch.zeros(mask.shape[0], self.num_targets).to(self.device)
        # Loop over dataloader
        for batch in self.inner_dataloader:
            x_batch = batch[0].to(self.device)
            y_true = batch[1].to(self.device)
            if self.feature_selector.data_type == "image":
                # Prepare broadcast for MNIST subtraction
                x_batch = x_batch.reshape(
                    1, x_batch.shape[0], x_batch.shape[-2], x_batch.shape[-1]
                )  # output shape for MNIST: (1, batch_size, 28, 28)
            elif self.feature_selector.data_type == "categorical":
                # Prepare broadcast for UCI Census Data | Add channel dimension
                x_batch = x_batch.unsqueeze(0)
            # Apply mask to batch from inner loop
            x_batch_masked = self.feature_selector.apply_mask(x_batch, mask)
            # Calculate norm of difference | Broadcasted subtraction | shape: (batch_size, batch_size)
            norm_diff = torch.linalg.norm(x_masked - x_batch_masked, dim=(2, 3)).squeeze()
            # Boolean mask where condition is fulfilled
            feature_bool_mask = norm_diff < tol
            for label in range(self.num_targets):
                # Boolean mask for current label
                label_bool_mask = (y_true == label).unsqueeze(0)  # shape: (1, batch_size)
                # Merge masks
                merged_bool_mask = torch.logical_and(
                    feature_bool_mask, label_bool_mask
                )  # shape: (batch_size, batch_size)
                # Count number of samples with same feature as x_input for current label
                incidence_per_class[:, label] += merged_bool_mask.sum(dim=1)

        return incidence_per_class

    def get_class_count(self, x_image, mask, fixedLoader):
        # Mask and make space for comparison batch dimension
        # mask = mask.unsqueeze(1)
        # x_image = x_image.unsqueeze(1)
        x_masked = mask * x_image
        self.n_classes = 2

        class_count = torch.zeros(mask.size(0), self.n_classes).to(self.device)  #

        for x_comp_image, y_comp in fixedLoader:
            x_comp_image, y_comp = x_comp_image.to(self.device), y_comp.to(self.device)

            # x_comp_image = x_image
            # y_comp = y_comp[:mask.shape[0]]
            # Mask and make space for feature batch dimension
            x_comp_image = x_comp_image.unsqueeze(dim=0)
            x_comp_image = x_comp_image.squeeze(dim=2)
            x_comp_masked = mask * x_comp_image

            # print(x_masked.shape, x_comp_masked.shape)

            # Calculate difference
            feature_deviation = torch.norm(x_masked - x_comp_masked, p=2, dim=(2, 3))
            feature_boolmask = feature_deviation < 1e-7

            for label in range(self.n_classes):
                # create label mask to compare to feature
                label_boolmask = (y_comp == label).unsqueeze(dim=0)

                feature_label_boolmask = torch.logical_and(feature_boolmask, label_boolmask)
                del label_boolmask

                # counts incidence of label for every feature
                incidence_count = feature_label_boolmask.sum(dim=1)
                del feature_label_boolmask

                class_count[:, label] += incidence_count
        return class_count

    @torch.no_grad()
    def self_information_from_prob(self, prob: torch.Tensor) -> torch.Tensor:
        """Returns the self-information of a given input."""
        return -torch.log(prob)

    @torch.no_grad()
    def entropy_from_probs(self, probs) -> torch.Tensor:
        """Returns the differential entropy of a given input batch."""
        return -torch.xlogy(probs, probs).sum(dim=1)

    @torch.no_grad()
    def precision_from_probs(self, probs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Returns the precision of the model for a given batch of predictions and targets."""
        return torch.gather(probs, 1, y_true.unsqueeze(1))

    def get_per_image_accuracy(self, x_masked, y, mode):
        if mode == "merlin":
            prediction = torch.argmax(self.arthur.eval()(x_masked), dim=1)
            accuracy = prediction.eq(y.squeeze())
        elif mode == "morgana":
            prediction = torch.argmax(self.arthur.eval()(x_masked), dim=1)
            accuracy = torch.logical_or(prediction.eq(y.squeeze()), prediction.eq(2))
        return accuracy  # type: ignore


class MorganaCriterion(torch.nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns the loss that is minimized by Arthur and maximized by Morgana.

        Args:
            logits (torch.Tensor): Arthurs output.
            target (torch.Tensor): True targets.

        Raises:
            ValueError: Reduction assertion, possible values are `mean`, `sum` and `none`.

        Returns:
            torch.Tensor: Outputs loss minimized by Arthur and maximized by Morgana.
        """
        logits_wrt_true_class = torch.gather(logits, dim=1, index=target.unsqueeze(1))
        logits_idk = logits[:, 2].unsqueeze(1)
        logits_concatenated = torch.cat((logits_wrt_true_class, logits_idk), 1)

        diff = -torch.abs(logits_wrt_true_class - logits_idk)

        target_cloned = torch.clone(target)
        target_cloned[torch.argmax(logits_concatenated, dim=1) == 1] = 2
        criterion = torch.nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction)

        if self.reduction == "mean":
            correction_term = -torch.log(1 + torch.exp(diff)).mean()
        elif self.reduction == "sum":
            correction_term = -torch.log(1 + torch.exp(diff)).sum()
        elif self.reduction == "none":
            correction_term = -torch.log(1 + torch.exp(diff)).squeeze()
        else:
            raise ValueError(f"unexpected value for reduction, got `{self.reduction}`")

        loss = criterion(logits, target_cloned) + correction_term

        return loss


def morgana_criterion_unstable(output, target):
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(output)
    loss1 = -(torch.gather(probs, 1, target.unsqueeze(1)) + probs[:, 2]).log().mean()
    return loss1


@torch.no_grad()
def categorical_accuracy(
    output: torch.Tensor, y_true: torch.Tensor, binary_classification: Optional[bool] = None
) -> float:
    if binary_classification is True:
        prediction = torch.round(torch.sigmoid(output)).squeeze()  # convert probabilities to binary predictions
    else:
        prediction = torch.argmax(output, dim=1)

    return prediction.eq(y_true).sum().item() / float(len(y_true))


@torch.no_grad()
def get_accuracy(
    logits: torch.Tensor, y_true: torch.Tensor, mode: str, binary_classification: Optional[bool] = None
) -> float:
    if binary_classification is True:
        prediction = torch.round(torch.sigmoid(logits)).squeeze()  # convert probabilities to binary predictions
    else:
        prediction = torch.argmax(logits, dim=1)
    if mode == "merlin":
        accuracy = prediction.eq(y_true.squeeze()).sum().item() / float(len(y_true))
    elif mode == "morgana" and binary_classification is not True:
        accuracy = torch.logical_or(prediction.eq(y_true.squeeze()), prediction.eq(2)).sum().item() / float(len(y_true))
    elif mode == "morgana" and binary_classification is True:
        accuracy = prediction.eq(y_true.squeeze()).sum().item() / float(len(y_true))
    else:
        raise ValueError(f"Unexpected value for mode, got `{mode}`")

    return accuracy


class Averager:
    def __init__(self):
        """Averager class for calculating the average of a given value.

        Example:
            >>> avg = Averager()
            >>> avg.add(1)
            >>> avg.add(2)
            >>> avg.result()
            1.5

        """
        self.reset()

    def reset(self):
        """Resets the averager."""
        self.sum = 0
        self.count = 0

    def result(self):
        """Returns the average of the added values."""
        assert self.count >= 1, f"You have not added a value to the averager so far, Counter = {self.count}"

        return self.sum / self.count

    def add(self, val, n=1):
        """Adds a value to the averager."""
        self.sum += val
        self.count += n


class EarlyStopping:
    """Sets Early Stopping Criteria.

    If loss/rde increases or stagnates several times, early stop is triggered.

    Args:
        tolerance (int): Number of repeated times until early stopping is triggered.
        min_delta (float): Threshold value for increasing the counter.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.continue_loop = True

    def __call__(self, val1: float, val2: float):
        """Checks if early stop is required.

        Args:
            val1 (float): Current loss.
            val2 (float): Best loss achieved so far.

        Returns True if early stop is required and False otherwise.
        """
        if (val1 - val2) > self.min_delta:
            # If new loss is bigger than the best loss so far achieved, increase counter.
            self.counter += 1
            if self.counter >= self.patience:
                # If the loss is higher than the best loss for multiple times, an early stop is required.
                self.continue_loop = False
        elif abs(val1 - val2) < self.min_delta:
            # If the loss does not improve significantly for multiple times, an early stop is required.
            self.counter += 1
            if self.counter >= self.patience:
                self.continue_loop = False
        else:
            # If the loss is smaller than the best loss so far achieved, reset counter.
            self.counter = 0
        return self.continue_loop


class MaskRegularizer:
    """Mask Regularizor class for calculating the norm of the mask."""

    @staticmethod
    def l1_norm(mask: torch.Tensor) -> torch.Tensor:
        """Calculates the L1 norm of the mask.

        Args:
            mask (torch.Tensor): Mask.

        Returns:
            torch.Tensor: L1 norm of the mask.
        """
        return torch.mean(torch.abs(mask))

    @staticmethod
    def l2_norm(mask: torch.Tensor) -> torch.Tensor:
        """Calculates the L2 norm of the mask.

        Args:
            mask (torch.Tensor): Mask.

        Returns:
            torch.Tensor: L2 norm of the mask.
        """
        return torch.mean(torch.square(mask))

    @staticmethod
    def tv_norm(mask: torch.Tensor, power=2) -> torch.Tensor:
        """Calculates the TV norm of the mask.

        Same as in https://arxiv.org/abs/1705.07857 with power 2.

        Args:
            mask (torch.Tensor): Mask.

        Returns:
            torch.Tensor: TV norm of the mask.
        """

        tv_norm = torch.sum(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:]) ** power) + torch.sum(
            torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]) ** power
        )

        return tv_norm / (mask.shape[0])  # normalize by batch size


class IoUCalculator:
    """IoU Calculator class for calculating the IoU of a given mask and bounding box.

    For each bounding box, the IoU is calculated with the mask. The maximum IoU is returned.
    """

    @staticmethod
    def iou(
        binary_mask: torch.Tensor,
        left: torch.Tensor,
        top: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
        pad_value: int = -1,
    ):
        iou_list = []
        for l, t, w, h in zip(left, top, width, height):
            if l == pad_value:  # Ignore padded bounding boxes
                continue

            bbox_mask = torch.zeros_like(binary_mask)
            bbox_mask[0][t : t + h, l : l + w] = 1

            intersection = torch.logical_and(binary_mask, bbox_mask)
            union = torch.logical_or(binary_mask, bbox_mask)
            iou = torch.sum(intersection) / torch.sum(union)

            iou_list.append(iou)

        max_iou = max(iou_list) if iou_list else 0
        return max_iou

    @staticmethod
    def iou_batch(
        binary_masks: torch.Tensor,
        lefts: torch.Tensor,
        tops: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
    ) -> List[float]:
        """Calculates the IoU of a batch of masks and bounding boxes. The batch size is the number of masks.

        For each bounding box, the IoU is calculated with the mask. The maximum IoU is returned and a list of IoUs is
        returned.
        """
        # TODO: This is not the most efficient way to calculate the IoU. It can be improved by vectorizing the code.
        # However, this is not a bottleneck in the code, so it is not a priority.
        iou_values = torch.tensor([])
        for binary_mask, left, top, width, height in zip(binary_masks, lefts, tops, widths, heights):
            iou = IoUCalculator.iou(binary_mask, left, top, width, height)
            iou_values = torch.cat((iou_values, torch.tensor([iou])))

        return iou_values  # type: ignore


class SaliencyLoss:
    def __init__(self, num_classes, mode=None, **kwargs):
        self.num_classes = num_classes
        self.area_loss_coef = 8 if mode == "merlin" else -8
        self.smoothness_loss_coef = 0.5 if mode == "merlin" else -0.5
        self.preserver_loss_coef = 0.3
        self.area_loss_power = 0.3 if mode == "merlin" else 0.3
        self.destroy_loss_active = kwargs.get("destroyer_loss_active", False)

    def get(self, masks, images, targets, black_box_func):
        one_hot_targets = self.one_hot(targets)

        area_loss = self.area_loss(masks)
        smoothness_loss = self.smoothness_loss(masks)
        destroyer_loss = (
            self.destroyer_loss(images, masks, one_hot_targets, black_box_func)
            if self.destroy_loss_active is True
            else 0
        )
        preserver_loss = self.preserver_loss(images, masks, one_hot_targets, black_box_func)

        return (
            destroyer_loss
            + self.area_loss_coef * area_loss
            + self.smoothness_loss_coef * smoothness_loss
            + self.preserver_loss_coef * preserver_loss  # type: ignore
        )

    def one_hot(self, targets):
        depth = self.num_classes
        if targets.is_cuda:
            return Variable(torch.zeros(targets.size(0), depth).cuda().scatter_(1, targets.long().view(-1, 1).data, 1))
        else:
            return Variable(torch.zeros(targets.size(0), depth).scatter_(1, targets.long().view(-1, 1).data, 1))

    def tensor_like(self, x):
        if x.is_cuda:
            return torch.Tensor(*x.size()).cuda()
        else:
            return torch.Tensor(*x.size())

    def area_loss(self, masks):
        if self.area_loss_power != 1:
            masks = (masks + 0.0005) ** self.area_loss_power  # prevent nan (derivative of sqrt at 0 is inf)

        return torch.mean(masks)

    def smoothness_loss(self, masks, power=2, border_penalty=0.0):
        x_loss = torch.sum((torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])) ** power)
        y_loss = torch.sum((torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])) ** power)
        if border_penalty > 0:
            border = float(border_penalty) * torch.sum(
                masks[:, :, -1, :] ** power
                + masks[:, :, 0, :] ** power
                + masks[:, :, :, -1] ** power
                + masks[:, :, :, 0] ** power
            )
        else:
            border = 0.0
        return (x_loss + y_loss + border) / float(power * masks.size(0))  # watch out, normalised by the batch size!

    def destroyer_loss(self, images, masks, targets, black_box_func):
        destroyed_images = self.apply_mask(images, 1 - masks)
        out = black_box_func(destroyed_images)

        return self.cw_loss(out, targets, targeted=False, t_conf=1, nt_conf=5)

    def preserver_loss(self, images, masks, targets, black_box_func):
        preserved_images = self.apply_mask(images, masks)
        out = black_box_func(preserved_images)

        return self.cw_loss(out, targets, targeted=True, t_conf=1, nt_conf=1)

    def apply_mask(
        self,
        images,
        mask,
        noise=True,
        random_colors=True,
        blurred_version_prob=0.5,
        noise_std=0.11,
        color_range=0.66,
        blur_kernel_size=55,
        blur_sigma=11,
        bypass=0.0,
        boolean=False,
        preserved_imgs_noise_std=0.03,
    ):
        images = images.clone()
        cuda = images.is_cuda

        if boolean:
            # remember its just for validation!
            return (mask > 0.5).float() * images

        assert 0.0 <= bypass < 0.9
        n, c, _, _ = images.size()

        if preserved_imgs_noise_std > 0:
            images = images + Variable(
                self.tensor_like(images).normal_(std=preserved_imgs_noise_std), requires_grad=False
            )
        if bypass > 0:
            mask = (1.0 - bypass) * mask + bypass
        if noise and noise_std:
            alt = self.tensor_like(images).normal_(std=noise_std)
        else:
            alt = self.tensor_like(images).zero_()
        if random_colors:
            if cuda:
                alt += torch.Tensor(n, c, 1, 1).cuda().uniform_(-color_range / 2.0, color_range / 2.0)
            else:
                alt += torch.Tensor(n, c, 1, 1).uniform_(-color_range / 2.0, color_range / 2.0)

        alt = Variable(alt, requires_grad=False)

        if blurred_version_prob > 0.0:  # <- it can be a scalar between 0 and 1
            cand = self.gaussian_blur(images, kernel_size=blur_kernel_size, sigma=blur_sigma)
            if cuda:
                when = Variable(
                    (torch.Tensor(n, 1, 1, 1).cuda().uniform_(0.0, 1.0) < blurred_version_prob).float(),
                    requires_grad=False,
                )
            else:
                when = Variable(
                    (torch.Tensor(n, 1, 1, 1).uniform_(0.0, 1.0) < blurred_version_prob).float(), requires_grad=False
                )
            alt = alt * (1.0 - when) + cand * when

        return (mask * images.detach()) + (1.0 - mask) * alt.detach()

    def cw_loss(self, logits, one_hot_labels, targeted=True, t_conf=2, nt_conf=5):
        this = torch.sum(logits * one_hot_labels, 1)
        other_best, _ = torch.max(
            logits * (1.0 - one_hot_labels) - 12111 * one_hot_labels, 1
        )  # subtracting 12111 from selected labels to make sure that they dont end up a maximum
        t = F.relu(other_best - this + t_conf)
        nt = F.relu(this - other_best + nt_conf)
        if isinstance(targeted, (bool, int)):
            return torch.mean(t) if targeted else torch.mean(nt)

    def gaussian_blur(self, _images, kernel_size=55, sigma=11):
        """Very fast, linear time gaussian blur, using separable convolution. Operates on batch of images [N, C, H, W].
        Returns blurred images of the same size. Kernel size must be odd.
        Increasing kernel size over 4*simga yields little improvement in quality. So kernel size = 4*sigma is a good choice.
        """

        kernel_a, kernel_b = self._gaussian_kernels(kernel_size=kernel_size, sigma=sigma, chans=_images.size(1))
        kernel_a = torch.Tensor(kernel_a)
        kernel_b = torch.Tensor(kernel_b)
        if _images.is_cuda:
            kernel_a = kernel_a.cuda()
            kernel_b = kernel_b.cuda()
        _rows = conv2d(
            _images, Variable(kernel_a, requires_grad=False), groups=_images.size(1), padding=(int(kernel_size / 2), 0)
        )
        return conv2d(
            _rows, Variable(kernel_b, requires_grad=False), groups=_images.size(1), padding=(0, int(kernel_size / 2))
        )

    def _gaussian_kernels(self, kernel_size, sigma, chans):
        assert kernel_size % 2, "Kernel size of the gaussian blur must be odd!"
        x = np.expand_dims(np.array(range(int(-kernel_size / 2), int(-kernel_size / 2) + kernel_size, 1)), 0)
        vals = np.exp(-np.square(x) / (2.0 * sigma**2))
        _kernel = np.reshape(vals / np.sum(vals), (1, 1, kernel_size, 1))
        kernel = np.zeros((chans, 1, kernel_size, 1), dtype=np.float32) + _kernel
        return kernel, np.transpose(kernel, [0, 1, 3, 2])


class PointingGameCalculator:
    """Pointing Game Calculator class for evaluating the pointing game metric.

    For each bounding box, it checks if the max point in the mask lies within the bounding box.
    """

    @staticmethod
    def is_point_inside_bbox(
        continuous_mask: torch.Tensor,
        left: torch.Tensor,
        top: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
        pad_value: int = -1,
    ) -> bool:
        if left == pad_value:  # Ignore padded bounding boxes
            return False

        y_max, x_max = torch.where(continuous_mask == torch.max(continuous_mask))

        # Check if max point is inside the bounding box
        return left <= x_max[0] < left + width and top <= y_max[0] < top + height

    @staticmethod
    def pointing_game(
        continuous_mask: torch.Tensor,
        lefts: torch.Tensor,
        tops: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
    ) -> float:
        """Evaluates the pointing game for a given mask and multiple bounding boxes.

        Returns the score for the pointing game: 1 if max point is inside any bounding box, 0 otherwise.
        """
        for left, top, width, height in zip(lefts, tops, widths, heights):
            if PointingGameCalculator.is_point_inside_bbox(continuous_mask.squeeze(), left, top, width, height):
                return 1.0
        return 0.0

    @staticmethod
    def pointing_game_batch(
        continuous_masks: torch.Tensor,
        lefts_batch: torch.Tensor,
        tops_batch: torch.Tensor,
        widths_batch: torch.Tensor,
        heights_batch: torch.Tensor,
    ) -> List[float]:
        """Calculates the pointing game for a batch of masks and bounding boxes.

        The batch size is the number of masks. Returns a list of pointing game scores.
        """
        scores = torch.tensor([])
        for continuous_mask, lefts, tops, widths, heights in zip(
            continuous_masks, lefts_batch, tops_batch, widths_batch, heights_batch
        ):
            score = PointingGameCalculator.pointing_game(continuous_mask, lefts, tops, widths, heights)
            scores = torch.cat((scores, torch.tensor([score])))

        return scores  # type: ignore


class RMACalculator:
    """RMACalculator class for calculating the RMA of a given heatmap and bounding box.

    For each heatmap, the RMA is calculated with the bounding box. The maximum RMA is returned.
    """

    @staticmethod
    def rma(
        heatmap: torch.Tensor,
        left: torch.Tensor,
        top: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
        pad_value: int = -1,
    ):
        rma_list = []

        for l, t, w, h in zip(left, top, width, height):
            if l == pad_value:  # Ignore padded bounding boxes
                continue

            # Convert bounding box coordinates to a mask (1s inside the box, 0s outside)
            mask = torch.zeros_like(heatmap)
            mask[0][t : t + h, l : l + w] = 1

            correct_relevance = torch.abs(heatmap[mask != 0]).sum().item()
            total_relevance = torch.abs(heatmap).sum().item()

            overlap = correct_relevance / total_relevance if total_relevance != 0 else 0
            rma_list.append(overlap)

        max_rma = float(max(rma_list)) if rma_list else 0.0
        return max_rma

    @staticmethod
    def rma_batch(
        heatmaps: torch.Tensor,
        lefts: torch.Tensor,
        tops: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
    ) -> List[float]:
        """Calculates the RMA of a batch of heatmaps and bounding boxes. The batch size is the number of heatmaps."""

        rma_values = []
        for heatmap, left, top, width, height in zip(heatmaps, lefts, tops, widths, heights):
            if heatmap.dtype != torch.float32:
                heatmap = heatmap.to(torch.float32)
            heatmap_tensor = heatmap.clone().detach()
            rma = RMACalculator.rma(heatmap_tensor, left, top, width, height)
            rma_values.append(rma)

        return torch.tensor(rma_values)  # type: ignore


class RRACalculator:
    """
    For each bounding box, a mask of the same shape as the heatmap is created, with ones inside the bounding box and zeros outside.
    The size of the ground truth mask, K, is the sum of the values inside the mask.
    The top K highest relevance values in the heatmap are obtained.
    For these top K values, the number of values that lie within the bounding box is counted.
    This count is divided by K to get the accuracy for this bounding box.
    The average RRA over all bounding boxes is returned.
    """

    @staticmethod
    def rra(
        heatmap: torch.Tensor,
        left: torch.Tensor,
        top: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
        pad_value: int = -1,
    ):
        rra_list = []

        for l, t, w, h in zip(left, top, width, height):
            if l == pad_value:  # Ignore padded bounding boxes
                continue

            # Convert bounding box coordinates to a mask (1s inside the box, 0s outside)
            mask = torch.zeros_like(heatmap)
            mask[0][t : t + h, l : l + w] = 1
            # print("RMACalc mask shape:", mask.shape)

            # Get K highest relevance values
            K = int(mask.sum().item())  # K is the size of the ground truth mask

            heatmap_1d = heatmap.view(-1)  # Flatten the heatmap to 1D
            top_k_values = torch.topk(heatmap_1d, k=K, largest=True).indices

            # Check how many of the top K values lie within the ground truth
            relevant_count = mask.flatten()[top_k_values].sum().item()

            accuracy = relevant_count / K if K != 0 else 0
            rra_list.append(accuracy)

        # average_rra = sum(rra_list) / len(rra_list) if rra_list else 0
        max_rra = float(max(rra_list)) if rra_list else 0.0

        return max_rra

    @staticmethod
    def rra_batch(
        heatmaps: torch.Tensor,
        lefts: torch.Tensor,
        tops: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
    ) -> List[float]:
        """Calculates the RRA of a batch of heatmaps and bounding boxes. The batch size is the number of heatmaps."""

        rra_values = []
        for heatmap, left, top, width, height in zip(heatmaps, lefts, tops, widths, heights):
            if heatmap.dtype != torch.float32:
                heatmap = heatmap.to(torch.float32)
            heatmap_tensor = heatmap.clone().detach()
            rra = RRACalculator.rra(heatmap_tensor, left, top, width, height)
            rra_values.append(rra)

        return torch.tensor(rra_values)
