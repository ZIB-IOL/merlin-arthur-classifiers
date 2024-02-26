from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainerConfig:
    epochs: int
    approach: str
    add_mask_channel: bool = False
    debug: bool = False
    save_masks: bool = False
    seed: int = 42
    wandb: bool = False

@dataclass
class DatasetConfig:
    dataset: str
    batch_size: int
    targets: List[str]
    add_normalization: bool = False
    original_image_ratio: Optional[float] = None

@dataclass
class SvhnConfig:
    svhn_target: int = 1
    resized_height: int = 32
    resized_width: int = 32
    extra_data_max_samples: Optional[int] = None
    svhn_autoaugment: bool = False
    add_extra_data: bool = False
    use_grayscale: bool = False
    one_vs_two: bool = False

@dataclass
class MetricsPenaltiesConfig:
    optimize_probabilities: bool = False
    other_loss: bool = False
    calculate_iou: bool = False
    l1_penalty: bool = False
    l2_penalty: bool = False
    tv_penalty: bool = False
    l1_penalty_coefficient: Optional[float] = None
    l2_penalty_coefficient: Optional[float] = None
    tv_penalty_coefficient: Optional[float] = None
    tv_penalty_power: int = 1
    entropy_penalty_coefficient: float = 0.4
    destroyer_loss_active: bool = False


@dataclass
class BooleanConfig:
    save_model: bool = False
    use_amp: bool = False
    cuda_benchmark: bool = False

@dataclass
class ArthurConfig:
    model_arthur: str
    imagenet_pretrained: bool = False
    reduce_kernel_size: bool = False
    binary_classification: bool = False
    lr: float = 1e-4
    weight_decay: float = 0
    pretrained_arthur: bool = False
    pretrained_path: Optional[str] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    scheduler_arthur_step_size: Optional[int] = None
    scheduler_arthur: bool = False

@dataclass
class FeatureSelectorConfig:
    segmentation_method: Optional[str] = None
    mask_size: Optional[int] = None
    lr_merlin: Optional[float] = None
    lr_morgana: Optional[float] = None
    gamma: Optional[float] = None
    gaussian: bool = False
    only_on_class: bool = False

@dataclass
class MaskOptimizationConfig:
    max_iterations: int = 300
    stoptol: float = 1e-5

@dataclass
class UnetConfig:
    saliency_mapper: bool = False
    parameter_sharing: bool = False
    steps_morgana: int = 1
    weight_decay_merlin: float = 0
    weight_decay_morgana: float = 0
    scheduler_merlin: bool = False
    scheduler_morgana: bool = False
    scheduler_merlin_step_size: Optional[int] = None
    scheduler_morgana_step_size: Optional[int] = None
