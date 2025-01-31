import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Optional


class Model(nn.Module):
    """
    A wrapper class for various pretrained models from torchvision.

    Attributes:
        model_name (str): Name of the pretrained model to use.
        num_classes (int): Number of output classes.
        model (nn.Module): The underlying PyTorch model.
    """

    # Model configuration dictionary
    MODEL_CONFIGS: Dict[str, Dict] = {
        "convnext_base": {
            "creator": models.convnext_base,
            "feature_layer": "classifier",
            "feature_index": 2,
        },
        "convnext_large": {
            "creator": models.convnext_large,
            "feature_layer": "classifier",
            "feature_index": 2,
        },
        "resnet50": {"creator": models.resnet50, "feature_layer": "fc"},
        "resnet101": {"creator": models.resnet101, "feature_layer": "fc"},
        "resnet152": {"creator": models.resnet152, "feature_layer": "fc"},
        "efficientnet_b7": {
            "creator": models.efficientnet_b7,
            "feature_layer": "classifier",
            "feature_index": 1,
        },
        "mobilenet_v2": {
            "creator": models.mobilenet_v2,
            "feature_layer": "classifier",
            "feature_index": 1,
        },
        "vit16": {
            "creator": models.vit_b_16,
            "feature_layer": "heads",
            "fixed_features": 768,
        },
        "vit32": {
            "creator": models.vit_b_32,
            "feature_layer": "heads",
            "fixed_features": 768,
        },
    }

    def __init__(self, model_name: str = "convnext_large", num_classes: int = 10):
        """
        Initialize the model.

        Args:
            model_name: Name of the pretrained model to use.
            num_classes: Number of output classes.

        Raises:
            ValueError: If model_name is not supported.
        """
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Choose from {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._initialize_model()

    def _initialize_model(self) -> nn.Module:
        """
        Initialize the specified model with a modified classification head.

        Returns:
            Initialized model with modified classification head.
        """
        config = self.MODEL_CONFIGS[self.model_name]
        model = config["creator"](weights="IMAGENET1K_V1")

        # Handle different model architectures
        if "fixed_features" in config:
            # ViT models have fixed feature size
            setattr(
                model,
                config["feature_layer"],
                nn.Linear(config["fixed_features"], self.num_classes),
            )
        else:
            # Get the target layer
            target_layer = getattr(model, config["feature_layer"])

            # Handle models with indexed layers (like classifier[1])
            if "feature_index" in config:
                num_features = target_layer[config["feature_index"]].in_features
                target_layer[config["feature_index"]] = nn.Linear(
                    num_features, self.num_classes
                )
            else:
                # Handle models with direct fc layer
                num_features = target_layer.in_features
                setattr(
                    model,
                    config["feature_layer"],
                    nn.Linear(num_features, self.num_classes),
                )

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Model output tensor.
        """
        return self.model(x)

    @classmethod
    def available_models(cls) -> List[str]:
        """
        Get a list of available model architectures.

        Returns:
            List of supported model names.
        """
        return list(cls.MODEL_CONFIGS.keys())