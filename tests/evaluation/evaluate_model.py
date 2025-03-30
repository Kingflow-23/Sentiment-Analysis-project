import os
import torch

from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from datetime import datetime

from src.inference import load_model
from src.evaluate import evaluate_and_plot
from src.dataloader import create_data_loader
from config import *
from dataset_config import FAKE_DATASET

# ✅ Set random seed for reproducibility
torch.manual_seed(42)

# ✅ Load tokenizer (same one used in training)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# ✅ Load the fake dataset
print(f"📂 Loaded Fake Dataset with {len(FAKE_DATASET)} samples\n")

# ✅ Create test data loader
test_loader = create_data_loader(
    FAKE_DATASET, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
)

# ✅ Load the trained model
model_path = (
    PRETRAINED_MODEL_3_CLASS_PATH if N_CLASSES == 3 else PRETRAINED_MODEL_5_CLASS_PATH
)
model = load_model(model_path, n_classes=N_CLASSES, device=DEVICE)
print(f"📂 Loaded Model from {model_path}\n")
model.to(DEVICE)

# ✅ Define loss function
loss_fn = CrossEntropyLoss()

# ✅ Create timestamped folder for results
output_dir = MODEL_EVALUATION_OUTPUT_DIR
timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
run_folder = os.path.join(output_dir, f"test_run_{timestamp}")
os.makedirs(run_folder, exist_ok=True)

# ✅ Evaluate and generate all plots at once

# ✅ Create the class_names based on N_CLASSES
if N_CLASSES == 3:
    # Use 3-class sentiment mapping
    class_names = [SENTIMENT_MAPPING_3_LABEL_VERSION[i] for i in range(1, 4)]
else:
    # Use 5-class sentiment mapping
    class_names = [SENTIMENT_MAPPING[i] for i in range(1, 6)]

evaluate_and_plot(model, test_loader, loss_fn, DEVICE, class_names, run_folder)

print(f"✅ Model test evaluation complete. Results saved in {run_folder}")
