from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard.writer as logging

import wandb


class SaveBestModel:
    """Class to save the best model while training.

    If the current epoch's validation loss is less than the previous least less, the model state is saved.
    """

    def __init__(self, best_valid_loss=float("inf")):  # type: ignore
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model_state, optimizer_state, log_dir):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Overall best validation loss: {self.best_valid_loss:1.4f}")
            print(f"Saving best model for epoch: {epoch+1} to ``{log_dir}``")
            print(10 * "---")
            epoch_state = {"epoch": epoch + 1}
            merged_state = {**model_state, **optimizer_state, **epoch_state}
            torch.save(
                merged_state,
                log_dir,
            )


@torch.no_grad()
def log_heatmap(
    img: torch.Tensor,
    output_unet: torch.Tensor,
    binarized_output_unet: torch.Tensor,
    masked_image: torch.Tensor,
    mode: str,
    lr: float,
    gamma: float,
    mask_size: int,
    logger: logging.SummaryWriter,
    global_step: int,
    epoch: int,
):
    """Plots input image, normalized Unet output, binarized Unet output and masked input image.

    Args:
        img (torch.Tensor): Original image.
        output_unet (torch.Tensor): Non-normalized output of Merlin or Morgana.
        binarized_output_unet (torch.Tensor): Binary Mask by Merlin or Morgana.
        masked_image (torch.Tensor): Masked image by Merlin or Morgana.
        mode (str): `merlin` or `morgana`.
        lr (float): Learning rate of optimizer of U-Net.
        gamma (float): Parameter of Morgana's loss function.
        logger (SummaryWriter): Tensorboard logger.
        global_step (int): Index from batch.
        epoch (int): Current epoch.
    """
    # Adjust dimensions of images
    if img.shape[0] == 1:
        img = img.cpu().squeeze()  # (1, H, W)
        masked_image = masked_image.cpu().squeeze()  # (C, H, W)
    elif img.shape[0] == 3:
        img = img.cpu().permute(1, 2, 0)  # (H, W, 3)
        masked_image = masked_image.cpu().permute(1, 2, 0)  # (C, H, W)
    output_unet = output_unet.cpu().squeeze()  # (C, H, W)
    binarized_output_unet = binarized_output_unet.cpu().squeeze()  # (C, H, W)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    matplotlib.use("Agg")

    # Plot images
    ax1.imshow(img)  # type: ignore # plot image
    ax2.imshow(output_unet, cmap="viridis", interpolation="nearest")  # plot normalized heatmap
    ax3.imshow(binarized_output_unet)  # plot binarized UNet output
    ax4.imshow(masked_image)  # plot masked image

    fig.tight_layout()
    logger.add_figure(
        tag=f"Test Saliency Maps | Epoch: {epoch} | {mode.upper()} | lr={lr:1.1E} | gamma={gamma} | mask_size={mask_size}",
        figure=fig,
        global_step=global_step,
    )
    logger.flush()
    plt.close(fig)


@torch.no_grad()
def log_heatmap_sfw(
    img: torch.Tensor,
    mask: torch.Tensor,
    masked_image: torch.Tensor,
    mode: str,
    sfw_lr: float,
    gamma: float,
    mask_size: int,
    logger: logging.SummaryWriter,
    global_step: int,
    epoch: int,
):
    """Plots input image, normalized Unet output, binarized Unet output and masked input image.

    Args:
        img (torch.Tensor): Original image.
        mask: (torch.Tensor): Mask by Merlin or Morgana.
        masked_image (torch.Tensor): Masked image by Merlin or Morgana.
        mode (str): `merlin` or `morgana`.
        sfw_lr (float): Learning rate of optimizer of U-Net.
        gamma (float): Parameter of Morgana's loss function.
        logger (SummaryWriter): Tensorboard logger.
        global_step (int): Index from batch.
        epoch (int): Current epoch.
    """
    # Adjust dimensions of images
    if img.shape[0] == 1:
        img = img.cpu().squeeze()  # (1, H, W)
        masked_image = masked_image.cpu().squeeze()  # (C, H, W)
    elif img.shape[0] == 3:
        img = img.cpu().permute(1, 2, 0)  # (H, W, 3)
        masked_image = masked_image.cpu().permute(1, 2, 0)  # (C, H, W)
    binary_mask = mask.cpu().squeeze()  # (H, W)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    matplotlib.use("Agg")

    # Plot images

    ax1.imshow(img)  # type: ignore # plot image
    ax2.imshow(binary_mask)  # type: ignore # plot binary mask
    ax3.imshow(masked_image)  # type: ignore # plot masked image

    fig.tight_layout()
    logger.add_figure(
        tag=f"Test Saliency Maps | Epoch: {epoch} | {mode.upper()}-SFW | lr={sfw_lr:1.1E} | gamma={gamma} | mask_size={mask_size}",
        figure=fig,
        global_step=global_step,
    )
    logger.flush()
    plt.close(fig)


# save pytorch mnist tensor as pdf
def save_tensor_as_pdf(tensor, tensor2, filename):
    """Saves a pytorch tensor as pdf.

    Args:
        tensor (torch.Tensor): Tensor to save.
        filename (str): Name of the file.
    """
    fig, axs = plt.subplots(2)
    axs[0].imshow(tensor, cmap="gray")
    axs[1].imshow(tensor2, cmap="gray")
    plt.axis("off")
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def dataclass_to_dict(dataclass_instance):
    return dataclass_instance.__dict__


def initialize_logger(config: Dict[str, Any]):
    """Initializes the logger.

    Args:
        config (Dict[str, Any]): Dictionary containing all hyperparameters and configurations.
    Returns:
        logger (wandb or None): Weights and Biases Logger.
    """
    if config["trainer_config"].wandb is True:
        config_dict_converted = {k: dataclass_to_dict(v) for k, v in config.items()}
        wandb.init(project="Merlin-Arthur", config=config_dict_converted)  # type: ignore
        logger = wandb  # type: ignore
    return logger  # type: ignore


def plot_with_bbox(x_merlin, x_masked_merlin, batch_left, batch_top, batch_width, batch_height):

    import random

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    def unnormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    x_merlin = x_merlin.cpu()
    x_masked_merlin = x_masked_merlin.cpu()
    batch_left = batch_left.cpu()
    batch_top = batch_top.cpu()
    batch_width = batch_width.cpu()
    batch_height = batch_height.cpu()

    num_pairs = 5
    random_index = random.randint(0, x_merlin.shape[0] - 4)

    fig, axs = plt.subplots(num_pairs, 2, figsize=(3, 2 * num_pairs))

    for i in range(num_pairs):
        # Unnormalize the image and saliency map
        unnormalized_image = unnormalize(x_merlin[i + random_index].clone(), mean, std).permute(1, 2, 0)
        unnormalized_saliency_map = unnormalize(x_masked_merlin[i + random_index].clone(), mean, std).permute(1, 2, 0)

        # Display the unnormalized image
        axs[i, 0].imshow(unnormalized_image)
        axs[i, 0].axis("off")

        # Display the unnormalized saliency map
        axs[i, 1].imshow(unnormalized_saliency_map)
        axs[i, 1].axis("off")

        # Add bounding boxes from given tensors left, top, width, height in correspondance

        # Remove tick marks and labels
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])

        # Iterate through the bounding boxes and add them to the image axes
        for j in range(len(batch_left[i + random_index])):
            if batch_left[i + random_index][j] != -1:  # Skip the padded values
                rect_img = patches.Rectangle(
                    (batch_left[i + random_index][j], batch_top[i + random_index][j]),
                    batch_width[i + random_index][j],
                    batch_height[i + random_index][j],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                rect_mask = patches.Rectangle(
                    (batch_left[i + random_index][j], batch_top[i + random_index][j]),
                    batch_width[i + random_index][j],
                    batch_height[i + random_index][j],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )

                axs[i, 0].add_patch(rect_img)
                axs[i, 1].add_patch(rect_mask)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=-0.77)
    plt.savefig("svhn_saliency_pairs.pdf", format="pdf")
