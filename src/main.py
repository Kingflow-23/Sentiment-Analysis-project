import torch

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from config import *
from src.train import train_model
from src.model import SentimentClassifier
from src.data_extraction import load_data
from src.evaluate import evaluate_and_plot
from src.dataloader import create_data_loader
from src.data_processing import preprocess_data


def main():
    # ✅ Step 1: Load Data & Tokenizer
    print("🔹 Loading dataset and tokenizer...\n")

    # Load raw dataset
    raw_data = load_data(
        DATASET_PATH, merge_labels=True
    )  # merge_labels param = True to merge labels into broader ones

    # Split Test Data BEFORE Preprocessing (to avoid leakage)
    train_val_data, test_data = train_test_split(
        raw_data, test_size=TEST_SIZE, stratify=raw_data["score"], random_state=42
    )

    # Apply preprocessing (removes noise, tokenization, etc.)
    print("🔹 Preprocessing train and validation sets...\n")
    train_data, val_data = preprocess_data(
        train_val_data, test_size=VAL_SIZE, max_length=MAX_LEN
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ✅ Step 2: Create Data Loaders
    print("🔹 Creating data loaders...\n")
    # Create training, validation, and test data loaders
    train_loader = create_data_loader(
        train_data, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
    )
    val_loader = create_data_loader(
        val_data, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
    )
    test_loader = create_data_loader(
        test_data, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
    )

    # ✅ Step 3: Initialize Model
    print("🔹 Initializing model...\n")
    model = SentimentClassifier(n_classes=N_CLASSES).to(DEVICE)

    # ✅ Step 4: Train Model
    print("🔹 Training model...\n")
    trained_model = train_model(model, train_loader, val_loader, DEVICE, epochs=EPOCHS)

    # ✅ Step 5: Evaluate Model
    print("🔹 Evaluating model...\n")

    sentiment_mapper = (
        SENTIMENT_MAPPING if N_CLASSES == 5 else SENTIMENT_MAPPING_3_LABEL_VERSION
    )

    evaluate_and_plot(
        trained_model,
        test_loader,
        torch.nn.CrossEntropyLoss(),
        DEVICE,
        class_names=list(sentiment_mapper.values()),
        run_folder=MODEL_EVALUATION_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
