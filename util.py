from typing import Dict, List, Optional
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training history metrics.

    Args:
        history: Dictionary containing training metrics.
        save_dir: Directory to save plots.
        show: Whether to display plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot Loss
    axes[0].plot(history["train_loss"], label="Training Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracy
    axes[1].plot(history["train_acc"], label="Training Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/training_history.png")
    if show:
        plt.show()
    plt.close()


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot t-SNE visualization of embeddings.

    Args:
        embeddings: Feature embeddings from model.
        labels: True labels.
        class_names: List of class names.
        save_dir: Directory to save plot.
        show: Whether to display plot.
    """
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", alpha=0.6
    )

    plt.title("t-SNE Visualization of Feature Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if class_names:
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=class_names,
            title="Classes",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/tsne_embeddings.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_dir: Directory to save plot.
        show: Whether to display plot.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="pred")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/confusion_matrix.png")
    if show:
        plt.show()
    plt.close()


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot classification report as heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_dir: Directory to save plot.
        show: Whether to display plot.
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names if class_names else None, output_dict=True
    )

    # Convert to DataFrame and drop support column
    df_report = pd.DataFrame(report).T
    df_report = df_report.drop("support", axis=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_report, annot=True, cmap="RdYlGn", fmt=".4f")

    plt.title("Classification Report")
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/classification_report.png")
    if show:
        plt.show()
    plt.close()


def plot_training_time(
    history: Dict[str, List[float]],
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training time per epoch.

    Args:
        history: Dictionary containing training metrics.
        save_dir: Directory to save plot.
        show: Whether to display plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch_times"], marker="o")
    plt.title("Training Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/training_time.png")
    if show:
        plt.show()
    plt.close()


def plot_prediction_distribution(
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot distribution of predictions across classes.

    Args:
        y_pred: Predicted labels.
        class_names: List of class names.
        save_dir: Directory to save plot.
        show: Whether to display plot.
    """
    plt.figure(figsize=(12, 6))

    # Count predictions per class
    unique, counts = np.unique(y_pred, return_counts=True)
    labels = class_names if class_names else [f"Class {i}" for i in unique]

    # Create bar plot
    plt.bar(labels, counts)
    plt.title("Distribution of Predictions Across Classes")
    plt.xlabel("Class")
    plt.ylabel("Number of Predictions")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/prediction_distribution.png")
    if show:
        plt.show()
    plt.close()


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    history: Dict[str, List[float]],
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show_plots: bool = True,
) -> None:
    """
    Comprehensive model evaluation with all plots.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        embeddings: Model embeddings.
        history: Training history.
        class_names: List of class names.
        save_dir: Directory to save plots.
        show_plots: Whether to display plots.
    """
    # Create all visualizations
    plot_training_history(history, save_dir, show_plots)
    plot_tsne_embeddings(embeddings, y_true, class_names, save_dir, show_plots)
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir, show_plots)
    plot_classification_report(y_true, y_pred, class_names, save_dir, show_plots)
    plot_training_time(history, save_dir, show_plots)
    plot_prediction_distribution(y_pred, class_names, save_dir, show_plots)