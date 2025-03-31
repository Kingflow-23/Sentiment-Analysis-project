import os
import torch
import pandas as pd

from datetime import datetime
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from src.inference import load_model
from src.data_extraction import load_data
from src.evaluate import evaluate_and_plot
from src.dataloader import create_data_loader

from config import *
from dataset_config import FAKE_DATASET

# Set random seed for reproducibility
torch.manual_seed(42)

# Set number of classes for both models
MODELS_CONFIG = {
    "merged_labels": {"n_classes": 3, "merge_labels": True},
    "original_labels": {"n_classes": 5, "merge_labels": False},
}

if __name__ == "__main__":
    # Load tokenizer (same one used in training)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Save dataset for consistency
    df = pd.DataFrame(FAKE_DATASET)
    df["score"] = df["score"].apply(lambda x: x + 1)
    df.to_excel(FAKE_DATASET_PATH, index=False)

    # Create timestamped folder for results
    output_dir = MODEL_EVALUATION_OUTPUT_DIR
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    test_run_folder = os.path.join(output_dir, f"test_run_{timestamp}")
    os.makedirs(test_run_folder, exist_ok=True)

    for config_name, config in MODELS_CONFIG.items():
        n_classes = config["n_classes"]
        merge_labels = config["merge_labels"]

        # Load dataset with the appropriate label merging setting
        df_processed = load_data(FAKE_DATASET_PATH, merge_labels=merge_labels)
        print(
            f"📂 Loaded Fake Dataset with {len(df_processed)} samples (merge_labels={merge_labels})\n"
        )

        # Create test data loader
        test_loader = create_data_loader(
            df_processed,
            tokenizer,
            max_len=MAX_LEN,
            batch_size=BATCH_SIZE,
        )

        # Load the trained model
        model_path = (
            PRETRAINED_MODEL_3_CLASS_PATH
            if n_classes == 3
            else PRETRAINED_MODEL_5_CLASS_PATH
        )
        model = load_model(model_path, n_classes=n_classes, device=DEVICE)
        print(f"📂 Loaded Model from {model_path} (merge_labels={merge_labels})\n")
        model.to(DEVICE)

        # Define loss function
        loss_fn = CrossEntropyLoss()

        # Create subfolder for this configuration
        config_folder = os.path.join(test_run_folder, config_name)
        os.makedirs(config_folder, exist_ok=True)

        # Set class names based on the model type
        class_names = (
            [SENTIMENT_MAPPING_3_LABEL_VERSION[i] for i in range(1, 4)]
            if n_classes == 3
            else [SENTIMENT_MAPPING[i] for i in range(1, 6)]
        )

        # Evaluate model and save results
        evaluate_and_plot(
            model, test_loader, loss_fn, DEVICE, class_names, config_folder
        )
        print(
            f"✅ Model evaluation complete for merge_labels={merge_labels}. Results saved in {config_folder}\n"
        )

    print(f"✅ All model test evaluations complete. Results saved in {test_run_folder}")
