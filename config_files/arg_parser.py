import argparse


# fmt: off
def add_trainer_args(parser):
    trainer_args = parser.add_argument_group('Trainer args')
    trainer_args.add_argument("-e", "--epochs", type=int, help="Number of Epochs", required=True)
    trainer_args.add_argument("--approach", type=str, help="Approach (e.g., `regular`, `unet`, `sfw`, `mask_optimization`, `posthoc`)", required=True)
    trainer_args.add_argument("--add_mask_channel", action=argparse.BooleanOptionalAction, help="Add mask channel")
    trainer_args.add_argument("-d", "--debug", action=argparse.BooleanOptionalAction, help="Debug mode for single batch")
    trainer_args.add_argument("--save_masks", action=argparse.BooleanOptionalAction, help="Save Masks of Optimization Approach for next Epoch")

    trainer_args.add_argument("--seed", type=int, help="Seed for Reproducibility", default=42)
    trainer_args.add_argument("--wandb", action=argparse.BooleanOptionalAction, help="Log to wandb")

def add_dataset_args(parser):
    dataset_args = parser.add_argument_group('Dataset args')
    dataset_args.add_argument("--dataset", type=str, help="Dataset", required=True)
    dataset_args.add_argument("--batch_size", type=int, help="Batch Size", required=True)
    dataset_args.add_argument("--targets", type=int, nargs="+", help="Targets", default=[0, 1])
    dataset_args.add_argument("--add_normalization", action=argparse.BooleanOptionalAction, help="Normalizes the input images")
    dataset_args.add_argument("--original_image_ratio", type=float, help="Ratio of original image size to resized image size", default=None)
                              
def add_svhn_args(parser):
    svhn_args = parser.add_argument_group('SVHN specific args')
    svhn_args.add_argument("--svhn_target", type=int, metavar="", help="SVHN Target", default=1)
    svhn_args.add_argument("--resized_height", type=int, help="Defines the resized height of the SVHN image", default=32)
    svhn_args.add_argument("--resized_width", type=int, help="Defines the resized height of the SVHN image", default=32)
    svhn_args.add_argument("--extra_data_max_samples", type=int, help="Max samples for extra data")
    svhn_args.add_argument("--svhn_autoaugment", action=argparse.BooleanOptionalAction, help="SVHN Auto Augmentation, see https://arxiv.org/abs/1805.09501")
    svhn_args.add_argument("--add_extra_data", action=argparse.BooleanOptionalAction, help="Add extra data to SVHN")
    svhn_args.add_argument("--use_grayscale", action=argparse.BooleanOptionalAction, help="Use grayscale images")
    svhn_args.add_argument("--one_vs_two", action=argparse.BooleanOptionalAction, help="1 vs. 2 (SVHN)")

def add_metrics_and_penalties_args(parser):
    metrics_and_penalties_args = parser.add_argument_group('Metrics and Penalties args')
    # Decision of Loss Function
    metrics_and_penalties_args.add_argument("--optimize_probabilities", action=argparse.BooleanOptionalAction, help="Optimize probabilities")
    metrics_and_penalties_args.add_argument("--other_loss", action=argparse.BooleanOptionalAction, help="Use loss function from Dabkowski et al. (2017)")
    # Metrics
    metrics_and_penalties_args.add_argument("--calculate_iou", action=argparse.BooleanOptionalAction, help="Calculate IoU")
    # Penalties
    metrics_and_penalties_args.add_argument("--l1_penalty", action=argparse.BooleanOptionalAction, help="Add L1 penalty to the loss")
    metrics_and_penalties_args.add_argument("--l2_penalty", action=argparse.BooleanOptionalAction, help="Add L2 penalty to the loss")
    metrics_and_penalties_args.add_argument("--tv_penalty", action=argparse.BooleanOptionalAction, help="Add Total Variation penalty to the loss")
    metrics_and_penalties_args.add_argument("--l1_penalty_coefficient", type=float, help="l1 Penalty coefficient", default=0.0)
    metrics_and_penalties_args.add_argument("--l2_penalty_coefficient", type=float, help="l2 penalty coefficient", default=0.0)
    metrics_and_penalties_args.add_argument("--tv_penalty_coefficient", type=float, help="TV penalty coefficient for U-Net", default=0.0)
    metrics_and_penalties_args.add_argument("--tv_penalty_power", type=int, help="TV power for SFW", default=1)
    metrics_and_penalties_args.add_argument("--entropy_penalty_coefficient", type=float, help="Entropy penalty coefficient", default=0.4)

def add_boolean_args(parser):
    boolean_args = parser.add_argument_group('BooleanOptionalAction args')
    boolean_args.add_argument("--remove_batch_norm", action=argparse.BooleanOptionalAction, help="Removes batch normalization from the WideResNet model (not working atm)")
    boolean_args.add_argument("--destroyer_loss_active", action=argparse.BooleanOptionalAction, help="Use destroyer loss in loss function as adversarial loss")
    boolean_args.add_argument("--save_model", action=argparse.BooleanOptionalAction, help="Save model")
    boolean_args.add_argument("--use_amp", action=argparse.BooleanOptionalAction, help="Use automatic mixed precision")
    boolean_args.add_argument("--cuda_benchmark", action=argparse.BooleanOptionalAction, help="Use cuda benchmark (not recommended)")

def add_arthur_args(parser):
    arthur_args = parser.add_argument_group('Arthur specific args')
    arthur_args.add_argument("--model_arthur", type=str, help="Model for Arthur, e.g., SimpleCNN, DeeperCNN, ResNet18 etc.")
    arthur_args.add_argument("--imagenet_pretrained", action=argparse.BooleanOptionalAction, help="Use pretrained model for Arthur's regular training")
    arthur_args.add_argument("--reduce_kernel_size", action=argparse.BooleanOptionalAction, help="Reduce kernel size of Arthur's First Convolutional Layer to fit resized SVHN images")
    arthur_args.add_argument("--lr", type=float, help="Learning Rate of Arthur", default=1e-4)
    arthur_args.add_argument("--weight_decay", type=float, help="Weight Decay for Arthur", default=0)
    arthur_args.add_argument("--pretrained_arthur", action=argparse.BooleanOptionalAction, help="Use pretrained model")
    arthur_args.add_argument("--pretrained_path", type=str, help="Path to pretrained model")
    arthur_args.add_argument("--hidden_size", type=int, help="Hidden Size of Classifier of MLP for tabular data (deprecated)")
    arthur_args.add_argument("--num_layers", type=int, help="Number of Layers in Classifier (MLP) for tabular data (deprecated)")
    arthur_args.add_argument("--scheduler_arthur_step_size", type=int, help="Step size for scheduler of Arthur")
    arthur_args.add_argument("--scheduler_arthur", action=argparse.BooleanOptionalAction, help="Use scheduler for Arthur")
    arthur_args.add_argument("--binary_classification", action=argparse.BooleanOptionalAction, help="Binary Classification")

def add_feature_selector_args(parser):
    feature_selector_args = parser.add_argument_group('General Feature Selector (SFW or U-Net) args')
    feature_selector_args.add_argument("--segmentation_method", type=str, help="Segmentation method for Merlin and Morgana (only topk atm)")
    feature_selector_args.add_argument("--mask_size", type=int, help="Size of Mask")
    feature_selector_args.add_argument("--lr_merlin", type=float, help="Learning Rate of Merlin either as U-Net or SFW Optimizer")
    feature_selector_args.add_argument("--lr_morgana", type=float, help="Learning Rate of Morgana either as U-Net or SFW Optimizer")
    feature_selector_args.add_argument("--gamma", type=float, help="Gamma for weighting the loss between Merlin and Morgana")
    feature_selector_args.add_argument("-g", "--gaussian", action=argparse.BooleanOptionalAction, help="Add Gaussian (now uniform!) noise for background to the masked images")
    feature_selector_args.add_argument("--only_on_class", action=argparse.BooleanOptionalAction, help="Only optimize on class")

def add_mask_optimization_args(parser):
    mask_optimization_args = parser.add_argument_group('Mask Optimization specific args')
    mask_optimization_args.add_argument("--max_iterations", type=int, help="Maximum number of iterations for Merlin and Morgana", default=300)
    mask_optimization_args.add_argument("--stoptol", type=float, help="Stopping tolerance for Merlin and Morgana", default=1e-5)

def add_unet_args(parser):
    unet_args = parser.add_argument_group('U-Net specific args')
    unet_args.add_argument("--saliency_mapper", action=argparse.BooleanOptionalAction, help="Initializes U-Net architecture for Dabkowski et al. (2017)")
    unet_args.add_argument("--parameter_sharing", action=argparse.BooleanOptionalAction, help="Use solo U-Net for Merlin and Morgana")
    unet_args.add_argument("--steps_morgana", type=int, help="Number of optimization steps for Morgana", default=1)
    unet_args.add_argument("--weight_decay_merlin", type=float, help="Weight Decay for Merlin", default=0)
    unet_args.add_argument("--weight_decay_morgana", type=float, help="Weight Decay for Morgana", default=0)
    unet_args.add_argument("--scheduler_merlin", action=argparse.BooleanOptionalAction, help="Use scheduler for Merlin")
    unet_args.add_argument("--scheduler_morgana", action=argparse.BooleanOptionalAction, help="Use scheduler for Morgana")
    unet_args.add_argument("--scheduler_merlin_step_size", type=int, help="Step size for scheduler of Merlin")
    unet_args.add_argument("--scheduler_morgana_step_size", type=int, help="Step size for scheduler of Morgana")
