import torch

from config_files.config import parse_arguments_and_create_config
from merlin_arthur_framework import MerlinArthurTrainer
from utilities.util import initialize_logger


def main():
    config = parse_arguments_and_create_config()
    torch.manual_seed(config["trainer_config"].seed)
    approach = config["trainer_config"].approach
    logger = initialize_logger(config)

    # Trainer
    trainer = MerlinArthurTrainer(config["trainer_config"], config["boolean_config"], logger)
    trainer.prepare_dataset(config["dataset_config"], config["svhn_config"])

    # Arthur
    trainer.initialize_arthur(config["arthur_config"])
    trainer.configure_arthur_optimizer()

    # Feature Selectors
    if approach in ("sfw", "unet", "mask_optimization", "posthoc"):
        trainer.initialize_feature_selectors(
            config["feature_selector_config"],
            config["mask_optimization_config"],
            config["unet_config"],
            config["metrics_penalties_config"],
        )

        if approach in ("sfw", "mask_optimization", "posthoc"):
            # Configure mask optimization approach (this includes sfw and adam)
            if trainer.merlin is not None and trainer.morgana is not None:
                trainer.merlin.configure_sfw_optimizer(
                    learning_rate=config["feature_selector_config"].lr_merlin,
                    momentum=0.9,
                    max_iterations=config["mask_optimization_config"].max_iterations,
                    stoptol=config["mask_optimization_config"].stoptol,
                )
                trainer.morgana.configure_sfw_optimizer(
                    learning_rate=config["feature_selector_config"].lr_morgana,
                    momentum=0.9,
                    max_iterations=config["mask_optimization_config"].max_iterations,
                    stoptol=config["mask_optimization_config"].stoptol,
                )
        elif approach == "unet":
            # Configure unet approach according to unet_config
            trainer.configure_unet_optimizers()
            # TODO: load pretrained U-Nets

    # Start training
    trainer.calculate_conditional_entropy()


if __name__ == "__main__":
    main()
