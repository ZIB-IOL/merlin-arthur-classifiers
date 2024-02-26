import random
from typing import Optional, Union

import numpy as np
import torch

import utilities.metrics as metrics
from config_files.config_dataclass import *
from models import SaliencyModel, SimpleNet
from stochastic_frank_wolfe import SFW, PositiveKSparsePolytope


class FeatureSelector(torch.nn.Module):
    def __init__(
        self,
        fs_config: FeatureSelectorConfig,
        mask_optim_cfg: Optional[MaskOptimizationConfig],
        unet_cfg: Optional[UnetConfig],
        metrics_and_penalties_cfg: MetricsPenaltiesConfig,
        **kwargs,
    ) -> None:
        """Class to define feature selector and its properties. Can be used for Merlin or Morgana.

        Args in configs include:
            model (Optional[Union[SimpleNet, str]]): If SimpleNet, then U-Nets are used as feature selector.
                If ``sfw``, then Stochastic-Frank-Wolfe optimizer is used as feature selector. Defaults to None.
            mode (Optional[str]): If "merlin", then Merlin is used. If "morgana", then Morgana is used. Defaults to None.
            mask_size (Optional[int]): Size of the mask. Defaults to None.
            gaussian (Optional[bool]): If True, then Gaussian noise is added to the input. Defaults to None.
            only_on_class (Optional[bool]): If True, then only the class of interest is considered. Defaults to None.
            add_mask_channel (Optional[bool]): If True, then the mask is added as a channel to the input. Defaults to None.
            binary_classification (Optional[bool]): If True, then binary classification is used. Defaults to False.
            optimize_probabilities (Optional[bool]): If True, then probabilities are optimized. Defaults to None.
            use_amp (Optional[bool]): If True, then autocast is used. Defaults to None.
            l1_penalty (Optional[bool]): If True, then L1 penalty is used. Defaults to None.
            l2_penalty (Optional[bool]): If True, then L2 penalty is used. Defaults to None.
            tv_penalty (Optional[bool]): If True, then TV penalty is used. Defaults to None.
            l1_penalty_coefficient (Optional[float]): Coefficient of L1 penalty. Defaults to None.
            l2_penalty_coefficient (Optional[float]): Coefficient of L2 penalty. Defaults to None.
            tv_penalty_coefficient (Optional[float]): Coefficient of TV penalty. Defaults to None.
            entropy_penalty_coefficient (Optional[float]): Coefficient of entropy penalty. Defaults to None.
            tv_power (Optional[float]): Power of TV penalty. Defaults to None.

        Raises:
            ValueError: If model is not SimpleNet or str.
            ValueError: If mode is not str.
            ValueError: If mask_size is not int.
            ValueError: If mask_size is not greater than 0.
            ValueError: If mode is not "merlin" or "morgana".
        """
        super().__init__()
        self.fs_config = fs_config
        self.mask_optim_cfg = mask_optim_cfg
        self.unet_cfg = unet_cfg
        self.metrics_and_penalties_cfg = metrics_and_penalties_cfg
        self.mask_size = fs_config.mask_size

        self.model = kwargs.get("model", None)
        self.mode = kwargs.get("mode", None)

        # Assertions of input parameters
        assert isinstance(self.model, (SimpleNet, SaliencyModel, str)), "Model must be SimpleNet, SaliencyModel or str."
        assert isinstance(self.mode, str), "Mode must be str."
        assert isinstance(self.mask_size, int), "Mask size must be int."
        assert self.mask_size > 0, "Mask size must be greater than 0."
        assert self.mode in ["merlin", "morgana"], "Mode must be ``merlin`` or ``morgana``."

        # Assign parameters
        self.gaussian = bool(self.fs_config.gaussian)
        self.only_on_class = bool(self.fs_config.only_on_class)
        self.add_mask_channel = bool(kwargs.get("add_mask_channel", False))
        self.binary_classification = bool(kwargs.get("binary_classification", False))
        self.optimize_probabilities = bool(self.metrics_and_penalties_cfg.optimize_probabilities)
        self.use_amp = kwargs.get("use_amp", False)
        self.l1_penalty = bool(self.metrics_and_penalties_cfg.l1_penalty)
        self.l2_penalty = bool(self.metrics_and_penalties_cfg.l2_penalty)
        self.l1_penalty_coefficient = self.metrics_and_penalties_cfg.l1_penalty_coefficient
        self.tv_penalty = bool(self.metrics_and_penalties_cfg.tv_penalty)
        self.l2_penalty_coefficient = self.metrics_and_penalties_cfg.l2_penalty_coefficient
        self.tv_penalty_coefficient = self.metrics_and_penalties_cfg.tv_penalty_coefficient
        self.entropy_penalty_coefficient = self.metrics_and_penalties_cfg.entropy_penalty_coefficient
        self.tv_penalty_power = self.metrics_and_penalties_cfg.tv_penalty_power
        self._data_type = None
        self.sfw_configuration = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure Loss Function
        if self.mode == "merlin":
            self.criterion = (
                torch.nn.BCEWithLogitsLoss() if self.binary_classification is True else torch.nn.CrossEntropyLoss()
            )
        elif self.mode == "morgana":
            self.criterion = torch.nn.BCEWithLogitsLoss() if self.binary_classification is True else metrics.MorganaCriterion()  # type: ignore
            # self.criterion =  metrics.morgana_criterion_unstable # Unstable version of Morgana criterion

    def configure_sfw_optimizer(
        self, learning_rate: float = 0.1, momentum: float = 0.9, max_iterations=350, stoptol=1e-5
    ):
        """Configures optimizer for SFW."""
        # Assertions of input parameters
        assert self.model in (
            "sfw",
            "mask_optimization",
            "posthoc",
        ), f"Model must be ``sfw`` or  ``msak_optimization``, got {self.model}."
        assert learning_rate > 0, "Learning rate must be greater than 0."
        assert momentum > 0, "Momentum must be greater than 0."
        assert max_iterations > 0, "Max iterations must be greater than 0."
        assert stoptol > 0, "Stoptol must be greater than 0."
        # Assign parameters
        self.sfw_learning_rate = learning_rate
        self.sfw_momentum = momentum
        self.sfw_max_iterations = max_iterations
        self.sfw_stoptol = stoptol
        self.sfw_configuration = True

    def forward(
        self,
        x,
        y: Optional[torch.Tensor] = None,
        arthur_classifier: Optional[torch.nn.Module] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Merlin or Morgana."""
        # a = SegmentationMethod.soft_topk(1, 2)
        if isinstance(self.model, SimpleNet):
            # Forward pass through U-Nets -> returns non-binary mask
            output = self.model(x)
        elif isinstance(self.model, SaliencyModel):
            # Forward pass through other U-net architecture
            output = self.model(x, y)
        elif self.model == "sfw":
            # Assertions
            self.__assert_inputs_forward_pass(x, y, arthur_classifier, mask)
            # Forward pass through SFW optimization
            output = self.optimize_mask_with_sfw(x, y, arthur_classifier, mask)  # type: ignore
        elif self.model in ("mask_optimization", "posthoc"):
            self.__assert_inputs_forward_pass(x, y, arthur_classifier, mask)
            # Random initialization of mask
            mask = torch.rand_like(mask, requires_grad=True)  # type: ignore
            # Forward pass through Adam/SGD optimization
            output = self.optimize_mask_unconstraint(x, y, arthur_classifier, mask)  # type: ignore
        else:
            raise ValueError(f"Model must be SimpleNet or ``sfw``, got {self.model}")

        return output

    def __assert_inputs_forward_pass(self, x, y, arthur_classifier, mask):
        """Assertions of optional input parameters for SFW optimization"""
        # assert self.sfw_configuration is True, "SFW optimizer must be configured."
        assert arthur_classifier is not None, "Arthur classifier must be provided."
        assert mask is not None, "Mask must be provided."
        assert isinstance(mask, torch.Tensor), "Mask must be torch.Tensor."
        assert isinstance(y, torch.Tensor), "True label must be torch.Tensor."
        assert y.shape[0] == x.shape[0], "True label must have same batch size as input."

    def optimize_mask_with_sfw(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        arthur_classifier: torch.nn.Module,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Optimize mask for Merlin or Morgana with SFW.

        Note: use torch.nn.utils.clip_grad_norm_(mask, 1) for clipping the gradients to a maximum norm of 1

        Args:
            x (torch.Tensor): Input image.
            y (torch.Tensor): True label.
            arthur_classifier (torch.nn.Module): Arthur classifier model.
            mask (torch.Tensor): Mask.

        Returns:
            torch.Tensor: Optimized continuous mask which needs to be projected to binary mask later on.

        Raises:
            ValueError: If mode is not "merlin" or "morgana".
        """
        batch_size = y.shape[0]
        assert x.numel() % batch_size == 0  # check if remainder is zero (no remainder)
        assert self.mask_size is not None, "Mask size must be provided."
        # Set constraint for optimization
        constraint = PositiveKSparsePolytope(n=x.numel() / batch_size, bs=batch_size, k=self.mask_size)
        # NOT ACTIVATED: Will only do something if necessary
        mask = constraint.shift_inside(mask)
        # Do not push nn.Parameter() to cuda, since this makes s non-leaf
        mask = torch.nn.Parameter(mask.to(self.device), requires_grad=True)
        mask.requires_grad = True

        # Set SFW optimizer
        rde_optim = SFW([mask], learning_rate=self.sfw_learning_rate, momentum=self.sfw_momentum, rescale=None)  # type: ignore

        # Disable the Arthur's autograd
        for param in arthur_classifier.parameters():
            param.requires_grad = False

        opt_loss = float("inf")
        early_stopping_criterion = metrics.EarlyStopping(patience=3, min_delta=1e-5)
        arthur_classifier.eval()  # Note: eval() is important here, due to running averages in BatchNorm
        # Loop over SFW iterations
        for _ in range(self.sfw_max_iterations):
            # Zero gradients
            rde_optim.zero_grad()
            # Forward pass
            x_masked = self.apply_mask(x, mask)
            logits = arthur_classifier(x_masked)
            # Convert tensors to appropriate format if binary classification is used
            if self.binary_classification is True:
                logits = logits.squeeze(1)
                y = y.float()

            # Calculate loss
            if self.mode == "merlin" and self.optimize_probabilities is not True:
                # Note: We want to minimize the distortion, so we use self.criterion
                distortion = self.criterion(logits, y)
            elif self.mode == "morgana" and self.optimize_probabilities is not True:
                # Note: Morgana want to maximize the distortion, so we use -self.criterion
                distortion = -self.criterion(logits, y)
            elif self.binary_classification is True and self.optimize_probabilities is True:
                distortion = 1 - torch.mean(torch.sigmoid(logits))
            else:
                raise ValueError(f"Mode must be ``merlin`` or ``morgana``, got {self.mode}")
            # Get parameters
            l1_penalty = self.l1_penalty_coefficient * metrics.MaskRegularizer.l1_norm(mask) if self.l1_penalty else 0
            tv_penalty = self.tv_penalty_coefficient * metrics.MaskRegularizer.tv_norm(mask) if self.tv_penalty else 0
            # Calculate loss
            loss = distortion + l1_penalty + tv_penalty
            # Backpropagation
            loss.backward()
            # Update mask
            rde_optim.step(constraints=[constraint])

            # Check if loss has improved, if not: stop
            if early_stopping_criterion(loss.item(), opt_loss):
                opt_loss = loss.item()
            else:
                break  # End of optimization

        # Enable the network's autograd
        for param in arthur_classifier.parameters():
            param.requires_grad = True

        # arthur_classifier.train()  # Note: train() is important here, due to running averages in BatchNorm
        # Detach mask from backward graph
        optimized_mask = mask.detach().clone()

        return optimized_mask

    def optimize_mask_unconstraint(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        arthur_classifier: torch.nn.Module,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Optimize mask for Merlin or Morgana without any constraints (e.g., SGD or ADAM).

        After an optimization step the mask is clipped to the interval [0, 1] and the L1 norm is normalized to the desired mask size.
        Finally, the mask is projected to a binary mask.

        Args:
            x (torch.Tensor): Input image.
            y (torch.Tensor): True label.
            arthur_classifier (torch.nn.Module): Arthur classifier model.
            mask (torch.Tensor): Mask.

        Returns:
            torch.Tensor: Optimized continuous mask which needs to be projected to binary mask later on.

        Raises:
            ValueError: If mode is not "merlin" or "morgana".
        """
        batch_size = y.shape[0]
        assert x.numel() % batch_size == 0  # check if remainder is zero (no remainder)

        # Do not push nn.Parameter() to cuda, since this makes s non-leaf
        mask = torch.nn.Parameter(mask.to(self.device), requires_grad=True)
        mask.requires_grad = True

        rde_optim = torch.optim.Adam([mask], lr=self.sfw_learning_rate)

        # Disable the Arthur's autograd
        for param in arthur_classifier.parameters():
            param.requires_grad = False

        opt_loss = float("inf")
        early_stopping_criterion = metrics.EarlyStopping(patience=5, min_delta=1e-5)
        arthur_classifier.eval()  # Note: eval() is important here, due to running averages in BatchNorm
        # Loop over iterations
        for _ in range(self.sfw_max_iterations):
            # Zero gradients
            rde_optim.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):  # type: ignore
                # Forward pass
                x_masked = self.apply_mask(x, mask)
                logits = arthur_classifier(x_masked)
                # Convert tensors to appropriate format if binary classification is used
                if self.binary_classification is True:
                    logits = logits.squeeze(1)
                    y = y.float()

                # Calculate loss
                if self.mode == "merlin" and self.optimize_probabilities is not True:
                    # Note: We want to minimize the distortion, so we use self.criterion
                    distortion = self.criterion(logits, y)
                elif self.mode == "morgana" and self.optimize_probabilities is not True:
                    # Note: Morgana want to maximize the distortion, so we use -self.criterion
                    distortion = -self.criterion(logits, y)
                elif self.binary_classification is True and self.optimize_probabilities is True:
                    distortion = 1 - torch.mean(torch.sigmoid(logits))
                else:
                    raise ValueError(f"Mode must be ``merlin`` or ``morgana``, got {self.mode}")

                # Get penalties
                l1_penalty = self.l1_penalty_coefficient * metrics.MaskRegularizer.l1_norm(mask) if self.l1_penalty else 0
                tv_penalty = 0
                if self.tv_penalty:
                    tv_penalty = (
                        self.tv_penalty_coefficient / (4 * np.sqrt(self.mask_size))
                    ) * metrics.MaskRegularizer.tv_norm(
                        mask, power=self.tv_penalty_power if self.tv_penalty_power is not None else 1
                    )  # type: ignore
                entropy_penalty = self.entropy_penalty_coefficient * torch.mean(mask * (1 - mask))
                # l1_penalty = self.l1_penalty_coefficient * ((torch.sum(torch.abs(mask), dim=[2, 3]) - self.mask_size) ** 2).sum()
                penalties = tv_penalty + entropy_penalty + l1_penalty
                # Calculate loss
                loss = distortion + penalties 
            # Backpropagation
            loss.backward()
            # Update mask
            # rde_optim.step(constraints=[constraint])
            rde_optim.step()

            with torch.no_grad():
                mask -= mask.grad * 0.01  # type: ignore
                mask.grad.zero_()  # type: ignore
                mask.data = torch.clip(mask, 0, 1)
                mask.data = self.normalize_l1(mask.data, self.mask_size)  # type: ignore

            # Check if loss has improved, if not: stop
            if early_stopping_criterion(loss.item(), opt_loss):
                opt_loss = loss.item()
            else:
                break  # End of optimization

        # Enable the network's autograd
        for param in arthur_classifier.parameters():
            param.requires_grad = True

        # Detach mask from backward graph
        optimized_mask = mask.detach().clone()

        return optimized_mask

    @torch.no_grad()
    def brute_force_search(
        self, x_input: torch.Tensor, y_true: torch.Tensor, arthur: torch.nn.Module, mask_collection: torch.Tensor
    ) -> torch.Tensor:
        """Brute force search for Merlin or Morgana.

        Args:
            x (torch.Tensor): Input.
            y (torch.Tensor): True label.
            arthur (torch.nn.Module): Arthur classifier model.

        Returns:
            torch.Tensor: Binary mask.

        Raises:
            ValueError: If mode is not "merlin" or "morgana".
        """
        batch_size = x_input.shape[0]
        arthur.eval()
        # Initialize first k-sparse mask
        mask = mask_collection[0]
        mask = mask.repeat(batch_size, 1).to(self.device)

        x_masked = self.apply_mask(x_input, mask)
        logits = arthur(x_masked)
        if self.mode == "merlin":
            criterion = torch.nn.CrossEntropyLoss(reduction="none")
            loss = criterion(logits, y_true)
        elif self.mode == "morgana":
            criterion = metrics.MorganaCriterion(reduction="none")
            loss = -criterion(logits, y_true)
        else:
            raise ValueError(f"Mode must be ``merlin`` or ``morgana``, got {self.mode}")

        # Loop over all masks
        for new_mask in mask_collection[1:]:
            # Iterate all combinations with brute force
            new_mask = new_mask.repeat(batch_size, 1).to(self.device)
            x_masked = self.apply_mask(x_input, new_mask)
            logits = arthur(x_masked)
            if self.mode == "merlin":
                new_loss = criterion(logits, y_true)
            elif self.mode == "morgana":
                new_loss = -criterion(logits, y_true)
            else:
                raise ValueError(f"Mode must be ``merlin`` or ``morgana``, got {self.mode}")
            loss_diff = loss - new_loss
            updated_loss = torch.where(loss_diff > 0, new_loss, loss)
            # Update mask entries, where necessary
            mask[~updated_loss.eq(loss)] = new_mask[~updated_loss.eq(loss)]
            # Update loss, where necessary
            loss = updated_loss

        return mask

    def normalize_l1(self, input: torch.Tensor, mask_size: int) -> torch.Tensor:
        factor = torch.clamp(mask_size / (1e-7 + torch.norm(input, p=1, dim=(2, 3), keepdim=True)), max=1)  # type: ignore
        return factor * input

    def apply_mask(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self._data_type == "categorical":
            return self._apply_mask_for_tabular_data(input, mask)
        elif self._data_type == "image":
            return self._apply_mask_for_image_data(input, mask, expected_noise=0.0)
        else:
            raise ValueError(f"Data type must be ``image`` or ``categorical``, got {self._data_type}")

    def _apply_mask_for_tabular_data(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # TODO: Check if correct
        mask = mask.unsqueeze(-1)
        return mask * input

    def _apply_mask_for_image_data(
        self, input: torch.Tensor, mask: torch.Tensor, expected_noise: float
    ) -> torch.Tensor:
        if self.gaussian:
            # Add gaussian noise to masked input
            x_masked = mask * input + (1 - mask) * torch.rand_like(input)
        else:
            x_masked = mask * input + expected_noise * (1 - mask) * torch.ones_like(input)

        if self.add_mask_channel:
            # add mask to channel of x_masked if flag is set
            x_masked = torch.cat([x_masked, mask], dim=1)

        return x_masked

    def segment(self, input: torch.Tensor, method: str) -> torch.Tensor:
        """Returns binary mask from tensor with continuous values.

        Args:
            continuous_mask (torch.Tensor): Continuous mask to apply segmentation.
            method (str): Method to use for segmentation.
            mask_size (Optional[int]): Size of mask if specified.

        Returns:
            torch.Tensor: Binary mask obtained by specified segmentation method.
        """
        if method == "topk":
            assert self.mask_size is not None, "Mask size must be specified for topk segmentation."
            mask = SegmentationMethod.topk(input, self.mask_size)
        elif method == "soft_topk":
            assert self.mask_size is not None, "Mask size must be specified for soft topk segmentation."
            mask = SegmentationMethod.soft_topk(input, self.mask_size)
        elif method == "otsu":
            mask = SegmentationMethod.otsus_method(input)
        elif method == "thresholding":
            # binary_mask = SegmentationMethod.threshold_method(input, threshold=1)
            raise NotImplementedError("Thresholding not implemented yet.")
        elif method is None:
            mask = input
        else:
            raise ValueError(f"Segmentation method {method} not supported.")

        return mask

    @property
    def data_type(self) -> str:
        if self._data_type is None:
            raise ValueError("Data type not set. Prepare Dataset in MerlinArthurTrainer to set the data type.")
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: str):
        if data_type not in ["image", "categorical"]:
            raise ValueError(f"Data type must be ``image`` or ``categorical``, got {data_type}")
        self._data_type = data_type


class SegmentationMethod:
    """Class to hold segmentation methods.

    Please check https://en.wikipedia.org/wiki/Image_segmentation for more information.

    Note: All methods must be static.
    """

    @staticmethod
    def topk(input: torch.Tensor, mask_size: int) -> torch.Tensor:
        """Returns saliency map from tensor with continuous values.

        Args:
            continuous_mask (torch.Tensor): Continuous, (potentially) normalized mask.
            mask_size (int): Size of mask.

        Returns:
            torch.Tensor: Binary mask with top k selected pixels.
        """
        v = torch.zeros_like(input).flatten(start_dim=1)
        max_indices = torch.topk(torch.abs(input.flatten(start_dim=1)), k=mask_size).indices.to(input.device)
        v.scatter_(1, max_indices, 1.0)

        return v.reshape(input.shape)

    @staticmethod
    def soft_topk(
        input: torch.Tensor, mask_size: int, num_samples: int = 5000, temperature: float = 1.0
    ) -> torch.Tensor:
        """Returns saliency map from tensor with continuous values.

        Args:
            continuous_mask (torch.Tensor): Continuous, (potentially) normalized mask.
            mask_size (int): Size of mask.

        Returns:
            torch.Tensor: Mask with top k selected pixels.
        """
        a = input.flatten(start_dim=1, end_dim=-1)
        _, length = a.size()  # Replace with the desired number of entries

        # Generate a random tensor of size (n, length)
        tensor = torch.randn(num_samples, length).to(a.device)

        # Get the values and indices of the top 10 entries along the last dimension
        _, top_indices = torch.topk(tensor, mask_size, dim=1)

        # Create a tensor of zeros with the same size as the original tensor
        mask = torch.zeros_like(tensor).to(a.device)

        # Use scatter to set the entries in the zeros tensor to 1 for each row
        mask.scatter_(1, top_indices, 1)

        a_m = torch.einsum("ik,jk->ij", a, mask)
        max_entries = a_m.max(dim=1).values
        z_i = torch.exp((a_m - max_entries.unsqueeze(1)) / temperature)

        soft_topk = torch.matmul(z_i, mask) / z_i.sum(dim=1).unsqueeze(1)

        return soft_topk.reshape(input.shape)

    @staticmethod
    def soft_topk_with_patches(
        input: torch.Tensor, mask_size: int, num_samples: int = 5000, temperature: float = 1.0
    ) -> torch.Tensor:
        """Returns saliency map from tensor with continuous values.

        Args:
            continuous_mask (torch.Tensor): Continuous, (potentially) normalized mask.
            mask_size (int): Size of mask.

        Returns:
            torch.Tensor: Mask with top k selected pixels.
        """
        mask_size = 512
        a = input.flatten(start_dim=1, end_dim=-1)
        _, length = a.size()  # Replace with the desired number of entries

        rand_int = torch.randint(48, 96, (num_samples,))

        # # Get the values and indices of the top 10 entries along the last dimension
        # _, top_indices = torch.topk(tensor, mask_size, dim=1)

        # # Create a tensor of zeros with the same size as the original tensor
        # mask = torch.zeros_like(tensor).to(a.device)

        # # Use scatter to set the entries in the zeros tensor to 1 for each row
        # mask.scatter_(1, top_indices, 1)

        # a_m = torch.einsum("ik,jk->ij", a, mask)
        # max_entries = a_m.max(dim=1).values
        # z_i = torch.exp((a_m - max_entries.unsqueeze(1)) / temperature)

        # soft_topk = torch.matmul(z_i, mask) / z_i.sum(dim=1).unsqueeze(1)

        # return soft_topk.reshape(input.shape)

    @staticmethod
    def otsus_method(input: torch.Tensor) -> torch.Tensor:
        """Returns binary mask via Otsu's method from tensor with continuous values.

        See: https://en.wikipedia.org/wiki/Otsu%27s_method
        or https://learnopencv.com/otsu-thresholding-with-opencv/.

        Args:
            continuous_mask (torch.Tensor): Continuous mask to apply segmentation.

        Returns:
            torch.Tensor: Binary mask obtained by Otsu's method.
        """
        raise NotImplementedError("Otsu's method not implemented yet.")

    @staticmethod
    def kmeans_method(input: torch.Tensor) -> torch.Tensor:
        """Returns binary mask from tensor with continuous values.

        Args:
            continuous_mask (torch.Tensor): Continuous mask to apply segmentation.

        Returns:
            torch.Tensor: Binary mask obtained by specified segmentation method.
        """
        raise NotImplementedError("K-means not implemented yet.")

    @staticmethod
    def threshold_method(input: torch.Tensor, threshold: float) -> torch.Tensor:
        """Returns binary mask from tensor with continuous values.

        Args:
            continuous_mask (torch.Tensor): Continuous mask to apply segmentation.

        Returns:
            torch.Tensor: Binary mask obtained by simple thresholding.
        """

        raise NotImplementedError("Thresholding not implemented yet.")

        # Note: Not implemented properly at the moment
        return torch.where(input >= threshold, torch.ones_like(input), torch.zeros_like(input))

    @staticmethod
    def growing_method(input: torch.Tensor) -> torch.Tensor:
        """Returns binary mask from tensor with continuous values.

        Args:
            continuous_mask (torch.Tensor): Continuous mask to apply segmentation.

        Returns:
            torch.Tensor: Binary mask obtained by specified segmentation method.
        """
        raise NotImplementedError("Growing method not implemented yet.")
