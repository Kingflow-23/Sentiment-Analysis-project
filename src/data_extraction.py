import json
import pandas as pd

from config import SENTIMENT_MAPPING, SENTIMENT_MAPPING_3_LABEL_VERSION, LABEL_MAPPING


def load_file_by_type(file_path: str):
    """
    Loads a file based on its type (CSV, JSON, XLSX).

    Args:
        file_path (str): Path to the file.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file format is not supported.
    """
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)  # Convert JSON to DataFrame
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path, engine="openpyxl")  # Load Excel file
        else:
            raise ValueError(
                f"Unsupported file format: {file_path}. Only CSV, JSON, and XLSX are supported."
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {file_path} not found.")


def merge_score_labels(score: int) -> int:
    """
    Merges sentiment score labels into broader categories.
    This function merges low scores (e.g., 0, 1) as 'negative',
    mid scores (e.g., 2, 3) as 'neutral', and high scores (e.g., 4) as 'positive'.

    Args:
        score (int): The original score value.

    Returns:
        int: The merged score category.
    """
    if score <= 1:
        return 0  # Negative
    elif score == 2:
        return 1  # Neutral
    else:
        return 2  # Positive


def load_data(file_path: str, merge_labels: bool = False):
    """
    Loads sentiment analysis dataset from a file and processes it.

    Args:
        file_path (str): Path to the dataset file.
        merge_labels (bool): Whether to merge labels. Defaults to False.

    Returns:
        pd.DataFrame: Processed data with 'text' and 'label'.

    Raises:
        ValueError: If required columns are missing or if labels contain invalid values.
    """
    try:
        df = load_file_by_type(file_path)  # Load file based on type

        # Check if required columns exist
        required_columns = {"content", "score"}
        if not required_columns.issubset(df.columns):
            raise ValueError("Dataset must contain 'content' and 'score' columns.")

        # Keep only relevant columns
        df = df[["content", "score"]].dropna()

        # ✅ Convert `score` values using `LABEL_MAPPING` (1-5 → 0-4)
        if not df["score"].isin(LABEL_MAPPING.keys()).all():
            raise ValueError(
                f"Dataset contains invalid score values. Allowed values: {sorted(LABEL_MAPPING.keys())}"
            )

        df["score"] = df["score"].map(LABEL_MAPPING)

        # If merge_labels is True, merge the labels into broader categories
        if merge_labels:
            df["score"] = df["score"].apply(merge_score_labels)

        if merge_labels:
            df["label"] = df["score"].map(
                lambda x: SENTIMENT_MAPPING_3_LABEL_VERSION[x]
            )
        else:
            df["label"] = df["score"].map(lambda x: SENTIMENT_MAPPING[x + 1])

        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(e)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: File {file_path} is empty.")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")
