import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")

from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.nn import Module
from typing import Tuple, List
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


def evaluate(
    model: Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> Tuple[float, float, List[int], List[int], List[float]]:
    """
    Evaluates the model on validation data.

    Args:
        model (Module): The neural network model to be evaluated.
        data_loader (DataLoader): DataLoader providing the validation dataset in batches.
        loss_fn (nn.Module): The loss function used to compute the loss.
        device (torch.device): The device (CPU or GPU) on which the model and data are located.

    Returns:
        tuple: (Average loss, accuracy, true labels, predicted labels, confidence scores)
            - Average loss: The average loss across all batches.
            - Accuracy: The accuracy of the model on the validation dataset.
            - true labels: The true labels for the validation dataset.
            - predicted labels: The predicted labels from the model.
            - confidence scores: The maximum confidence for each prediction (the probability for the predicted class).
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    all_y_true = []
    all_y_pred = []
    all_confidences = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            predictions = torch.argmax(probs, dim=1)

            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            # Store values for evaluation plots
            all_y_true.extend(targets.cpu().numpy())
            all_y_pred.extend(predictions.cpu().numpy())
            all_confidences.extend(
                probs.max(dim=1)[0].cpu().numpy()
            )  # Max confidence per prediction

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy, all_y_true, all_y_pred, all_confidences


# 📊 **Confusion Matrix Plot**
def plot_confusion_matrix(
    y_true: List[int], y_pred: List[int], class_names: List[str], save_path: str
) -> None:
    """
    Plots and saves a confusion matrix for the predictions.

    Args:
        y_true (List[int]): True labels for the dataset.
        y_pred (List[int]): Predicted labels for the dataset.
        class_names (List[str]): List of class labels (e.g., ['class_0', 'class_1', 'class_2']).
        save_path (str): Path to save the confusion matrix plot.

    Notes:
        The confusion matrix is saved as a heatmap. Each class is represented on both axes.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    print(f"📊 Saved Confusion Matrix: {save_path}\n")
    plt.close()


# 📊 **Precision, Recall, F1-score Plot**
def plot_classification_report(
    y_true: List[int], y_pred: List[int], class_names: List[str], save_path: str
) -> None:
    """
    Plots and saves a bar chart for precision, recall, and F1-score per class.

    Args:
        y_true (List[int]): True labels for the dataset.
        y_pred (List[int]): Predicted labels for the dataset.
        class_names (List[str]): List of class labels (e.g., ['class_0', 'class_1', 'class_2']).
        save_path (str): Path to save the classification report plot.

    Notes:
        - Uses `classification_report` from `sklearn` to calculate precision, recall, and F1-score.
        - `zero_division=0` is used to avoid division by zero errors (replaces undefined values with 0).
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).T

    df.plot(kind="bar", figsize=(10, 6), colormap="coolwarm")
    plt.title("Precision, Recall & F1-score per Class")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    print(f"📊 Saved Classification Report: {save_path}\n")
    plt.close()


# 📊 **Prediction Confidence Histogram**
def plot_prediction_confidence(confidences: List[float], save_path: str) -> None:
    """
    Plots and saves a histogram of the prediction confidence scores.

    Args:
        confidences (List[float]): The confidence scores for each prediction (probabilities).
        save_path (str): Path to save the confidence histogram plot.

    Notes:
        The histogram shows the distribution of the confidence levels of the model's predictions.
        Higher confidence scores indicate that the model is more confident about its predictions.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(confidences, bins=20, color="green", alpha=0.7)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Prediction Confidence Distribution")
    plt.savefig(save_path)
    print(f"📊 Saved Confidence Histogram: {save_path}\n")
    plt.close()


# 📊 **Evaluation Pipeline**
def evaluate_and_plot(
    model: Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: List[str],
    run_folder: str,
) -> None:
    """
    Evaluates the model and generates multiple plots: confusion matrix, classification report, and prediction confidence histogram.

    Args:
        model (Module): The trained model to be evaluated.
        data_loader (DataLoader): DataLoader for evaluation.
        loss_fn (torch.nn.Module): Loss function used for evaluation (e.g., CrossEntropyLoss).
        device (torch.device): The device (CPU/GPU) to run the evaluation.
        class_names (List[str]): List of class labels (e.g., ['class_0', 'class_1', 'class_2']).
        run_folder (str): Directory where the evaluation results and plots will be saved.

    Notes:
        - The function generates three plots:
          1. Confusion matrix (saved as `confusion_matrix.png`)
          2. Precision, recall, and F1-score bar chart (saved as `classification_report.png`)
          3. Histogram of prediction confidence scores (saved as `confidence_histogram.png`)
    """
    # ✅ Evaluate Model
    avg_loss, accuracy, y_true, y_pred, confidences = evaluate(
        model, data_loader, loss_fn, device
    )

    # ✅ Create timestamped folder for this training run
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(run_folder, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # ✅ Save Metrics to File
    metrics_path = os.path.join(run_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Validation Loss: {avg_loss:.4f}\n")
        f.write(f"Validation Accuracy: {accuracy:.4f}\n")
    print(f"📄 Saved Metrics: {metrics_path}\n")

    # ✅ Generate & Save Plots
    plot_confusion_matrix(
        y_true, y_pred, class_names, os.path.join(run_dir, "confusion_matrix.png")
    )
    plot_classification_report(
        y_true,
        y_pred,
        class_names,
        os.path.join(run_dir, "classification_report.png"),
    )
    plot_prediction_confidence(
        confidences, os.path.join(run_dir, "confidence_histogram.png")
    )
