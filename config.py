import os
import torch

# --------------------------------------------------------------------------
# Define 5-class sentiment mapping
SENTIMENT_MAPPING = {
    1: "Really Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Really Positive"
}

LABEL_MAPPING = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4
}

# --------------------------------------------------------------------------
# Real Dataset paths
DATASET_PATH = "dataset/real_datasets/dataset.csv"

# --------------------------------------------------------------------------
# Test dataset folder path
TEST_DATA_DIR = "dataset/test_datasets" # folder containing test data files   
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
MODEL_NAME = "bert-base-uncased"
N_CLASSES = 5
DROPOUT = 0.3
LEARNING_RATE = 2e-5
EPOCHS = 10
BATCH_SIZE = 16
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model 
MODEL_TRAINING_OUTPUT_DIR = "output/model_output/training"
MODEL_EVALUATION_OUTPUT_DIR = "output/model_output/evaluation"
