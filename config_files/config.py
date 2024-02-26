import argparse

import config_files.config_dataclass as cfg_dataclass
from config_files import arg_parser


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merlin-Arthur Training")
    arg_parser.add_trainer_args(parser)
    arg_parser.add_dataset_args(parser)
    arg_parser.add_svhn_args(parser)
    arg_parser.add_metrics_and_penalties_args(parser)
    arg_parser.add_boolean_args(parser)
    arg_parser.add_arthur_args(parser)
    arg_parser.add_feature_selector_args(parser)
    arg_parser.add_mask_optimization_args(parser)
    arg_parser.add_unet_args(parser)

    return parser.parse_args()


def create_config_instances(args):
    """Create instances of the configuration classes and add them to a dictionary."""
    config_dict = {}
    # Create an instance of each configuration class and add it to the dictionary.
    config_dict["trainer_config"] = cfg_dataclass.TrainerConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.TrainerConfig.__annotations__}
    )
    config_dict["dataset_config"] = cfg_dataclass.DatasetConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.DatasetConfig.__annotations__}
    )
    config_dict["svhn_config"] = cfg_dataclass.SvhnConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.SvhnConfig.__annotations__}
    )
    config_dict["metrics_penalties_config"] = cfg_dataclass.MetricsPenaltiesConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.MetricsPenaltiesConfig.__annotations__}
    )
    config_dict["boolean_config"] = cfg_dataclass.BooleanConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.BooleanConfig.__annotations__}
    )
    config_dict["arthur_config"] = cfg_dataclass.ArthurConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.ArthurConfig.__annotations__}
    )
    config_dict["feature_selector_config"] = cfg_dataclass.FeatureSelectorConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.FeatureSelectorConfig.__annotations__}
    )
    config_dict["mask_optimization_config"] = cfg_dataclass.MaskOptimizationConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.MaskOptimizationConfig.__annotations__}
    )
    config_dict["unet_config"] = cfg_dataclass.UnetConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.UnetConfig.__annotations__}
    )

    return config_dict


def parse_arguments_and_create_config():
    args = parse_arguments()
    return create_config_instances(args)
