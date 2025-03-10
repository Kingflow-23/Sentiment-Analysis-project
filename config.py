import os
import torch

# --------------------------------------------------------------------------
# Define 5-class sentiment mapping
SENTIMENT_MAPPING = {
    1: "Really Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Really Positive",
}

SENTIMENT_MAPPING_3_LABEL_VERSION = {
    1: "Negative",
    2: "Neutral",
    3: "Positive",
}

LABEL_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

# --------------------------------------------------------------------------
# Real Dataset paths
DATASET_PATH = "dataset/real_datasets/dataset.csv"

# --------------------------------------------------------------------------
# Test dataset folder path
TEST_DATA_DIR = "dataset/test_datasets"  # folder containing test data files
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Test dataset paths
NON_EXISTING_DATASET_PATH = "dataset/test_datasets/non_existing.csv"
EMPTY_DATASET_PATH = "dataset/test_datasets/empty.csv"
MISSING_COLUMNS_DATASET_PATH = "dataset/test_datasets/missing_columns.csv"
INVALID_SCORE_DATASET_PATH = "dataset/test_datasets/invalid_score.csv"

CSV_FILE_PATH = "dataset/test_datasets/data.csv"
JSON_FILE_PATH = "dataset/test_datasets/data.json"
XLSX_FILE_PATH = "dataset/test_datasets/data.xlsx"
TXT_FILE_PATH = "dataset/test_datasets/data.txt"
XML_FILE_PATH = "dataset/test_datasets/data.xml"

# --------------------------------------------------------------------------
# Model config
TOKENIZER_NAME = "bert-base-uncased"
MODEL_NAME = "bert-base-uncased"

EPOCHS = 10
N_CLASSES = 5  # 5
DROPOUT = 0.3
MAX_LEN = 128
TEST_SIZE = 0.1
VAL_SIZE = 0.1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
MODEL_TRAINING_OUTPUT_DIR = "output/model_output/training"
MODEL_EVALUATION_OUTPUT_DIR = "output/model_output/evaluation"

# --------------------------------------------------------------------------
# inference config
PRETRAINED_MODEL_5_CLASS_PATH = (
    "output/model_output/training/run_09-03-2025-08-59-06/best_model.pth"
)
PRETRAINED_MODEL_3_CLASS_PATH = (
    "output/model_output/training/run_09-03-2025-20-24-25/best_model.pth"
)

PRETRAINED_MODEL_INVALID_PATH = (
    "output/model_output/training/run_invalid_run/best_model.pth"
)

# --------------------------------------------------------------------------
# Jupyter contants
JUPYTER_MAX_LEN = 160
RANDOM_SEED = 42

JUPYTER_MODEL_NAME = "bert-base-cased"
CLASS_NAME = ["Negative", "Neutral", "Positive"]
HAPPY_COLORS_PALETTE = [
    "#01BEFE",
    "#FFDD00",
    "#FF7D00",
    "#FF006D",
    "#ADFF02",
    "#8F00FF",
]

DATA_ANALYSIS_PATH = "output/data_analysis"

# --------------------------------------------------------------------------
# App config
APP_NAME = "Sentiment Analysis Web App"

# Color Mapping for Sentiment Display
COLOR_MAPPING = {
    "Really Negative": "red",
    "Negative": "red",
    "Neutral": "blue",
    "Positive": "green",
    "Really Positive": "green",
}
