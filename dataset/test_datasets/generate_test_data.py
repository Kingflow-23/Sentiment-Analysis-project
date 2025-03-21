import os
import json
import pandas as pd

from dataset_config import FAKE_DATASET

# Test dataset folder path
TEST_DATA_DIR = "dataset/test_datasets"


def generate_test_files():
    """Creates test dataset files automatically."""

    # ✅ Create CSV file (Full dataset)
    csv_path = os.path.join(TEST_DATA_DIR, "data.csv")
    pd.DataFrame(FAKE_DATASET).to_csv(csv_path, index=False)
    print(f"✅ Created: {csv_path}")

    # ✅ Create JSON file
    json_path = os.path.join(TEST_DATA_DIR, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(FAKE_DATASET, f, indent=4)
    print(f"✅ Created: {json_path}")

    # ✅ Create XLSX file
    xlsx_path = os.path.join(TEST_DATA_DIR, "data.xlsx")
    pd.DataFrame(FAKE_DATASET).to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"✅ Created: {xlsx_path}")

    # ✅ Create an empty CSV file
    empty_csv_path = os.path.join(TEST_DATA_DIR, "empty.csv")
    pd.DataFrame().to_csv(empty_csv_path, index=False)
    print(f"✅ Created: {empty_csv_path}")

    # ✅ Create a CSV file with missing columns (only "content" column)
    missing_columns_csv_path = os.path.join(TEST_DATA_DIR, "missing_columns.csv")
    pd.DataFrame({"content": FAKE_DATASET["content"]}).to_csv(
        missing_columns_csv_path, index=False
    )
    print(f"✅ Created: {missing_columns_csv_path}")

    # ✅ Create a CSV file with invalid scores
    invalid_score_data = {
        "content": ["I love this!", "Worst ever!", "Not bad", "Great!", "Terrible"],
        "score": [
            5,
            -1,
            7,
            999,
            0,
        ],  # Invalid scores that should not exist in SENTIMENT_MAPPING
    }
    invalid_score_csv_path = os.path.join(TEST_DATA_DIR, "invalid_score.csv")
    pd.DataFrame(invalid_score_data).to_csv(invalid_score_csv_path, index=False)
    print(f"✅ Created: {invalid_score_csv_path}")

    # ✅ Create a non-existing file (by defining the path but not creating it)
    non_existing_csv_path = os.path.join(TEST_DATA_DIR, "non_existing.csv")
    print(
        f"✅ Non-existing file path set: {non_existing_csv_path} (will not be created)"
    )

    # ✅ Create unsupported file formats (TXT & XML)
    txt_path = os.path.join(TEST_DATA_DIR, "data.txt")
    with open(txt_path, "w") as f:
        f.write("This is a test file for unsupported formats.\n")
    print(f"✅ Created: {txt_path}")

    xml_path = os.path.join(TEST_DATA_DIR, "data.xml")
    with open(xml_path, "w") as f:
        f.write("<root>\n    <test>This is an XML test file</test>\n</root>\n")
    print(f"✅ Created: {xml_path}")

# if __name__ == "__main__":
#     generate_test_files()