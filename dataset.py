from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from typing import Tuple, List, Optional, Dict


class DataAugmentation:
    """Handles data augmentation for SAR image datasets."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        color_jitter: bool = False,
        jitter_params: Optional[Dict[str, float]] = None,
        gaussian_noise: bool = False,
        noise_sigma: float = 0.1,
        random_erasing: bool = False,
        erasing_params: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize data augmentation settings.

        Args:
            image_size: Size of the output images.
            normalize_mean: Mean for normalization.
            normalize_std: Standard deviation for normalization.
            color_jitter: Whether to apply color jitter.
            jitter_params: Parameters for color jitter.
            gaussian_noise: Whether to add Gaussian noise.
            noise_sigma: Standard deviation of Gaussian noise.
            random_erasing: Whether to apply random erasing.
            erasing_params: Parameters for random erasing.
        """
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        # Default parameters
        self.jitter_params = jitter_params or {"brightness": 0.5, "hue": 0.3}
        self.noise_sigma = noise_sigma
        self.erasing_params = erasing_params or {
            "p": 0.5,
            "scale": (0.02, 0.33),
            "ratio": (0.3, 3.3),
            "value": 0,
        }

        self.train_transform = self._build_transform(
            color_jitter=color_jitter,
            gaussian_noise=gaussian_noise,
            random_erasing=random_erasing,
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def _add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to a tensor."""
        noise = torch.randn_like(tensor) * self.noise_sigma
        return tensor + noise

    def _build_transform(
        self, color_jitter: bool, gaussian_noise: bool, random_erasing: bool
    ) -> transforms.Compose:
        """Build the transformation pipeline."""
        transform_list: List[transforms.Transform] = [
            transforms.Resize(self.image_size),
        ]

        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=self.jitter_params["brightness"],
                    hue=self.jitter_params["hue"],
                )
            )

        transform_list.append(transforms.ToTensor())

        if gaussian_noise:
            transform_list.append(self._add_gaussian_noise)

        if random_erasing:
            transform_list.append(
                transforms.RandomErasing(
                    p=self.erasing_params["p"],
                    scale=self.erasing_params["scale"],
                    ratio=self.erasing_params["ratio"],
                    value=self.erasing_params["value"],
                )
            )

        transform_list.append(
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        )

        return transforms.Compose(transform_list)


class DatasetManager:
    """Manages dataset loading, splitting, and creation of data loaders."""

    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        batch_size: int,
        val_split: float = 0.2,
        num_workers: int = 0,
        augmentation: Optional[DataAugmentation] = None,
    ):
        """
        Initialize dataset manager.

        Args:
            train_dir: Path to training data directory.
            test_dir: Path to test data directory.
            batch_size: Number of samples per batch.
            val_split: Fraction of test data to use for validation.
            num_workers: Number of workers for data loading.
            augmentation: Data augmentation configuration.

        Raises:
            ValueError: If directories don't exist or val_split is invalid.
        """
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self._validate_inputs()

        self.augmentation = augmentation or DataAugmentation(
            color_jitter=True, gaussian_noise=True, random_erasing=True
        )

        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.train_dir.exists():
            raise ValueError(f"Training directory not found: {self.train_dir}")
        if not self.test_dir.exists():
            raise ValueError(f"Test directory not found: {self.test_dir}")
        if not 0 < self.val_split < 1:
            raise ValueError(f"Invalid validation split: {self.val_split}")

    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Split a dataset into test and validation sets.

        Args:
            dataset: Dataset to split.

        Returns:
            Test and validation datasets.
        """
        test_size = int(len(dataset) * (1 - self.val_split))
        val_size = len(dataset) - test_size
        return random_split(dataset, [test_size, val_size])

    def setup(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Set up datasets and create data loaders.

        Returns:
            Train, test, and validation data loaders.

        Raises:
            RuntimeError: If dataset loading fails.
        """
        def worker_init_fn(worker_id: int) -> None:
            np.random.seed(42 + worker_id)

        try:
            self.train_dataset = datasets.ImageFolder(
                root=str(self.train_dir), transform=self.augmentation.train_transform
            )

            test_dataset = datasets.ImageFolder(
                root=str(self.test_dir), transform=self.augmentation.test_transform
            )

            self.test_dataset, self.val_dataset = self._split_dataset(test_dataset)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )

            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )

            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )

            return train_loader, test_loader, val_loader

        except Exception as e:
            raise RuntimeError(f"Failed to set up datasets: {str(e)}")

    @property
    def class_names(self) -> List[str]:
        """Get list of class names from the training dataset."""
        if self.train_dataset is None:
            raise RuntimeError("Datasets not yet initialized. Call setup() first.")
        return self.train_dataset.classes

    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(self.class_names)