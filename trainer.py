from typing import Tuple, List, Dict, Any, NamedTuple
from pathlib import Path
import copy
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time: float


class EvaluationResults(NamedTuple):
    """Container for model evaluation results."""
    embeddings: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    history: Dict[str, List[float]]


class ModelTrainer:
    """Handles model training, validation, and testing with model checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader.
            criterion: Loss function.
            optimizer: Optimization algorithm.
            device: Device to run on (CPU/GPU).
            config: Training configuration dictionary.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # Initialize best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.checkpoint_path = Path(config.get("checkpoint_path", "best_model.pth"))

    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (epoch loss, epoch accuracy).
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)

        return epoch_loss, epoch_acc.item()

    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Returns:
            Tuple of (epoch loss, epoch accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_loader.dataset)

        return epoch_loss, epoch_acc.item()

    def _save_checkpoint(self) -> None:
        """Save the current model state as the best checkpoint."""
        torch.save(self.model.state_dict(), self.checkpoint_path)
        self.best_model_state = copy.deepcopy(self.model.state_dict())

    def train_and_evaluate(self) -> EvaluationResults:
        """
        Train the model and return evaluation results.

        Returns:
            EvaluationResults containing embeddings, predictions, true labels, and training history.
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": [],
        }

        for epoch in range(self.config["num_epochs"]):
            epoch_start = time.time()

            # Training phase
            train_loss, train_acc = self._train_epoch()

            # Validation phase
            val_loss, val_acc = self._validate_epoch()

            # Check if this is the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint()
                print(f"\nNew best model saved! Validation acc: {val_acc:.4f}")

            epoch_time = time.time() - epoch_start

            # Update history
            metrics = TrainingMetrics(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_time=epoch_time,
            )

            self._update_history(history, metrics)
            self._log_progress(epoch, metrics)

        # Load the best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with validation accuracy: {self.best_val_acc:.4f}")

        # Get test results
        embeddings, predictions, true_labels = self.test()

        return EvaluationResults(
            embeddings=embeddings,
            predictions=predictions,
            true_labels=true_labels,
            history=history,
        )

    def _update_history(
        self, history: Dict[str, List[float]], metrics: TrainingMetrics
    ) -> None:
        """Update training history with new metrics."""
        history["train_loss"].append(metrics.train_loss)
        history["train_acc"].append(metrics.train_acc)
        history["val_loss"].append(metrics.val_loss)
        history["val_acc"].append(metrics.val_acc)
        history["epoch_times"].append(metrics.epoch_time)

    def _log_progress(self, epoch: int, metrics: TrainingMetrics) -> None:
        """Log training progress."""
        print(
            f"\nEpoch {epoch + 1}/{self.config['num_epochs']} "
            f"({metrics.epoch_time:.2f}s)"
        )
        print(
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"Train Acc: {metrics.train_acc:.4f}"
        )
        print(
            f"Val Loss: {metrics.val_loss:.4f} | "
            f"Val Acc: {metrics.val_acc:.4f}"
        )

    def test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model on test set.

        Returns:
            Tuple of (embeddings, predictions, labels).
        """
        self.model.eval()
        embeddings = []
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                inputs, batch_labels = batch
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                embeddings.append(outputs.cpu().numpy())
                predictions.append(preds.cpu().numpy())
                labels.append(batch_labels.numpy())

        return (
            np.concatenate(embeddings),
            np.concatenate(predictions),
            np.concatenate(labels),
        )