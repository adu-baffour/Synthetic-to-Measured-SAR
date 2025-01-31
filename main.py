import os
import sys
from pathlib import Path
import torch
from torch import nn, optim
from dataset import DataAugmentation, DatasetManager
from torch.utils.data import DataLoader
from model import Model
from trainer import ModelTrainer
from util import evaluate_model_performance, set_seed
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)

set_seed()


@dataclass
class Config:
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 0.0001
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    base_dir: Path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
    results_dir: Path = base_dir / "results"
    plots_dir: Path = results_dir / "plots"
    checkpoint_path: Path = results_dir / "saved_model" / "best_model.pth"

    # Data paths
    train_dir: Path = base_dir / "dataset/synth"
    test_dir: Path = base_dir / "dataset/real"

    # Model configuration
    model_name: str = "convnext_base"
    num_classes: int = 10
    class_names: Tuple[str, ...] = (
        "2s1",
        "bmp2",
        "btr70",
        "m1",
        "m2",
        "m35",
        "m60",
        "m548",
        "t72",
        "zsu23",
    )

    # Data augmentation configuration
    image_size: Tuple[int, int] = (128, 128)

    color_jitter: bool = True
    color_jitter_params = {
        "brightness": 0.1,
        "hue": 0.5,
    }

    gaussian_noise: bool = True
    gaussian_noise_params: float = 0.4

    random_erasing: bool = True
    random_erasing_params = {
        "p": 0.3,
        "scale": (0.02, 0.33),
        "ratio": (0.3, 3.3),
        "value": 0,
    }

    val_split: float = 0.2

    def setup_directories(self) -> None:
        """Create necessary directories."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration dictionary."""
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "checkpoint_path": str(self.checkpoint_path),
            "initial_lr": self.learning_rate,
        }

    def setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data augmentation and dataset manager."""
        set_seed()

        augmentation = DataAugmentation(
            color_jitter=self.color_jitter,
            jitter_params=self.color_jitter_params,
            gaussian_noise=self.gaussian_noise,
            noise_sigma=self.gaussian_noise_params,
            random_erasing=self.random_erasing,
            erasing_params=self.random_erasing_params,
        )

        dataset_manager = DatasetManager(
            train_dir=str(self.train_dir),
            test_dir=str(self.test_dir),
            batch_size=self.batch_size,
            val_split=self.val_split,
            augmentation=augmentation,
        )

        return dataset_manager.setup()

    def setup_model(self) -> Tuple[Model, nn.Module, optim.Optimizer]:
        """Initialize model, loss function, and optimizer."""
        set_seed()

        model = Model(model_name=self.model_name, num_classes=self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, criterion, optimizer


def main() -> None:
    """Main function to execute the training and evaluation pipeline."""
    set_seed()
    config = Config()
    config.setup_directories()

    train_loader, test_loader, valid_loader = config.setup_data()
    model, criterion, optimizer = config.setup_model()

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        config=config.get_training_config(),
    )

    results = trainer.train_and_evaluate()

    evaluate_model_performance(
        y_true=results.true_labels,
        y_pred=results.predictions,
        embeddings=results.embeddings,
        history=results.history,
        class_names=config.class_names,
        save_dir=str(config.plots_dir),
        show_plots=False,
    )


if __name__ == "__main__":
    main()