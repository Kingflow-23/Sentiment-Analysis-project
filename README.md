# Sentiment Analysis Pipeline

This project implements an end-to-end sentiment analysis pipeline using BERT. Our goal is to provide a fine-grained classification system for sentiment (with both 5-class and 3-class variants) so that social media or app review managers can identify not only positive and negative feedback but also pinpoint critical comments that require immediate attention.

---

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
  - [Approach](#approach)
  - [Challenges & Solutions](#challenges--solutions)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Config](#config)
  - [Data Extraction & Preprocessing](#data-extraction--preprocessing)
  - [Model Training & Evaluation](#model-training--evaluation)
  - [Inference & Deployment](#inference--deployment)
- [Docker & MLOps Pipeline](#docker--mlops-pipeline)
- [Results & Evaluation](#results--evaluation)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Team Members](#team-members)

---

## Introduction

Sentiment analysis is a Natural Language Processing (NLP) task that determines whether a given text expresses positive, negative, or neutral sentiment. In today’s digital age, understanding public opinion is critical for businesses and researchers. This project leverages BERT—a state-of-the-art NLP model—to build an end-to-end pipeline that extracts, preprocesses, and analyzes text data for sentiment classification.

---

## Project Overview

### Approach

- **Objective:** Build a sentiment analysis pipeline that goes beyond simple positive/negative/neutral classification by using a five-class system:
  - **5-class model:** Distinguishes between "Really Negative," "Negative," "Neutral," "Positive," and "Really Positive."
  - **3-class model:** A simpler alternative mapping to "Negative," "Neutral," and "Positive."
  
- **Rationale:** A five-class model provides granular insights. For example, social media managers can prioritize "Really Negative" feedback separately from mildly negative comments, leading to more effective responses.

### Challenges & Solutions 

- **Test Coverage:** Achieved high test coverage using pytest with detailed unit tests and coverage reports. Using the command "pytest “test” -v --cov=”file/folder” --cov-report=term-missing"
- **Overfitting:** Identified early overfitting during training; addressed through regularization, early stopping, and hyperparameter tuning.
- **Evaluation:** Integrated multiple evaluation methods including accuracy metrics, confusion matrices, classification reports, and loss/accuracy plots.
- **Collaboration:** Effective use of Git workflows and CI tools to manage team contributions.

--- 

## Repository Structure


```python
Sentiment-Analysis-project/
├── .github/
│   └── workflows/
│       ├── build.yml                       # Build and Push Docker image to DockerHub
│       ├── evaluate.yml                    # Evaluate the model on Fake dataset at each push on main
│       └── test.yml                        # Workflow which runs unit tests and ensure code linting
├── dataset/                    
│   ├── real_datasets/                      # Real-world dataset files
│   │   └── dataset.csv                     # Sample dataset file
│   └── test_datasets/                      # Automatically generated test files
│       └── generate_test_files.py          # Script to generate test files
├── output/
│   ├── data_analysis/                      # Data analysis results
│   ├──database/
│   │   └── sentiment_logs.db               # Sqlite database to store inferences results
│   └── model_output/
│       ├── training/                       # Training outputs, saved models, history, plots
│       └── evaluation/                     # Evaluation outputs and plots
├── src/
│   ├── api.py                              # FastApi for inference
│   ├── app.py                              # Streamlit app for inference
│   ├── cli.py                              # Command Line Interface for inference
│   ├── data_extraction.py                  # Loads raw data from files
│   ├── data_processing.py                  # Cleans and tokenizes text data, splits dataset
│   ├── dataloader.py                       # Constructs PyTorch DataLoaders
│   ├── db_logger.py                        # Function to init the Sqlite database
│   ├── main.py                             # Main training and evaluation script
│   ├── model.py                            # Defines the SentimentClassifier model
│   ├── train.py                            # Contains training routines and plotting functions
│   ├── evaluate.py                         # Contains evaluation and plotting functions
│   └── inference.py                        # Provides sentiment prediction for new inputs
├── tests/
│   ├── evaluation/
│   │   └── evaluate_model.py               # Function used by github workflow for model evaluation.
│   └── unit/
│       ├── test_api_inference.py           # Unit tests for FastApi inference
│       ├── test_cli_inference.py           # Unit tests or CLI inferences
│       ├── test_data_extraction.py         # Unit tests for data extraction
│       ├── test_data_processing.py         # Unit tests for data processing
│       ├── test_dataloader.py              # Unit tests for data loading
│       ├── test_model.py                   # Unit tests for model initialization
│       ├── test_train.py                   # Unit tests for training and evaluation routines
│       └── test_inference.py               # Unit tests for the inference functionality
├── .gitignore                              # Specifies files to ignore in the Git repository
├── config.py                               # Configuration settings (paths, model parameters, etc.)
├── dataset_config.py                       # Ai-Generated Fake Dataset of review|score
├── Dockerfile                              # Builds the Docker image for the project.
├── docker-compose.yml                      #  Defines services (Streamlit, FastAPI, CLI) and maps volumes.
├── entrypoint.sh                           #  Set the entry point mode for the Docker container.
├── README.md                               # Project documentation (this file)
└── requirements.txt                        # Dependencies required to run the project
```

---

## Installation & Setup

1. **Clone the Repository:**
   
    ```bash
   git clone https://github.com/Kingflow-23/Sentiment-Analysis-project.git
    ```

    
    ```bash
   cd Sentiment-Analysis-project
    ```

2. **Create a Virtual Environment and Activate It:**
   ```bash
   python -m venv .
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**

    Our Python version used for this project is **3.10.11**

    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Generate Test Data (if necessary):**
    - The project includes scripts to automatically generate test datasets in dataset/test_datasets. To generate test data, Uncomment last line in the file and run the following command:

    ```bash
    python dataset/test_datasets/generate_test_data.py
    ```

---

## Usage

### Config 

The configuration file serves as the central location for all settings that control the behavior of the sentiment analysis pipeline. It organizes paths, mappings, and model parameters into logical sections. This makes it easier to customize the project without having to dive deep into the code.

1. **Sentiment Mappings**

    - **5-Class Mapping:**
    ```python
    SENTIMENT_MAPPING = {
    1: "Really Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Really Positive",
    }
    ```

    This mapping is used to interpret numerical sentiment scores into five distinct sentiment categories.

    - **3-Class Mapping:**
    ```python
    SENTIMENT_MAPPING_3_LABEL_VERSION = {
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    }
    ```

    For a more streamlined classification, this mapping consolidates sentiments into three labels.

    - **Label Conversion:**

    ```python
    LABEL_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    ```

    This helps convert the sentiment scores into indices that the model can use for training.

2. **Dataset Paths**

- **Real Dataset:
    ```python
    DATASET_PATH = "dataset/real_datasets/dataset.csv"
    ```
    
    This is the main dataset used for training. Ensure that the file exists in this location or update the path accordingly.

- **Test Dataset Folder & Files:**

    The configuration sets up a folder for test datasets and defines paths for several test cases:

    - Non-existing file, empty file, files with missing columns, or invalid sentiment scores.
    - Additionally, paths for different file formats (CSV, JSON, XLSX, TXT, XML) are specified.

    **Review these if you plan to run tests or want to experiment with different data formats.**

3. **Model Configuration**

    **Model and Tokenizer Names:**

    ```python
    TOKENIZER_NAME = "bert-base-uncased"
    MODEL_NAME = "bert-base-uncased"
    ```

    These determine which pre-trained BERT model and tokenizer are used. You can adjust these if you prefer a different model variant (e.g., bert-base-cased or any other model from Hugging Face).

    **Training Hyperparameters:**

    ```python
    EPOCHS = 10
    N_CLASSES = 3  # 5
    DROPOUT = 0.3
    MAX_LEN = 128
    TEST_SIZE = 0.1
    VAL_SIZE = 0.1
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    ```

    These settings control the training process:

    - **EPOCHS**: Number of training iterations.
    - **N_CLASSES**: Set to 3 for the 3-class model; change to 5 for the 5-class variant.
    - **DROPOUT**: Regularization factor to reduce overfitting.
    - **MAX_LEN**: Maximum token length for input sequences.
    - **TEST_SIZE & VAL_SIZE**: Proportions of data reserved for testing and validation.
    - **BATCH_SIZE & LEARNING_RATE**: Batch size for training and the learning rate for the optimizer.

    You should review these based on your computational resources and the size/nature of your dataset.

    **Device Selection:**

    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

    This setting automatically selects the GPU if available; otherwise, it falls back to the CPU. Confirm that you have the correct device configuration, especially if you run into hardware-related issues.

    **Output Directories:**

    ```python
    MODEL_TRAINING_OUTPUT_DIR = "output/model_output/training"
    MODEL_EVALUATION_OUTPUT_DIR = "output/model_output/evaluation"
    ```

    These directories store training logs, saved models, and evaluation outputs. Ensure these directories exist or adjust the paths as needed for your system setup.

4. **Inference Configuration**

    **Pretrained Model Paths:**

    ```python
    PRETRAINED_MODEL_5_CLASS_PATH = "output/model_output/training/run_10-03-2025-18-46-59/best_model.pth"
    PRETRAINED_MODEL_3_CLASS_PATH = "output/model_output/training/run_11-03-2025-03-30-19/best_model.pth"
    PRETRAINED_MODEL_INVALID_PATH = "output/model_output/training/run_invalid_run/best_model.pth"
    ```

    These paths point to the saved models for inference. Make sure these paths are updated to reflect the actual model locations on your system, especially if you have retrained or relocated models.

5. App Configuration

    **Web Application Settings:**

    ```python
    APP_NAME = "Sentiment Analysis Web App"
    COLOR_MAPPING = {
        "Really Negative": "red",
        "Negative": "red",
        "Neutral": "blue",
        "Positive": "green",
        "Really Positive": "green",
    }
    ```

    These settings configure the display for the Streamlit web app, including the app name and color coding for different sentiment labels.

    - **APP_NAME**: This is the name of the web application that will be displayed in the app's interface.
    - **COLOR_MAPPING**: Colors assigned to each sentiment label (adjust colors according to your preference or UI design).

    You may want to customize these settings to match your preferred UI design or branding guidelines.

---

### Data Extraction & Preprocessing

- **Data Extraction:**

    **src/data_extraction.py** loads raw data from CSV, JSON, or XLSX files and maps sentiment scores using defined mappings. The dataset path has to be specified in the **config.py** file if you would like to change the default path.

    - **Important Note**: Ensure your uploaded dataset is in the correct format and has the required columns ("content" for review & "score" for sentiment score).

    Set the merge_labels argument from load_data function to True to merge labels from 5 sentiment score into 3. 
    
    - **Important Note**: Don't forget to update N_CLASSES accordingly in the config.py file !!

- **Data Preprocessing:**
    
    **src/data_processing.py** cleans the text, tokenizes it with BERT’s tokenizer, and splits the data into training and validation sets.

    **Note**: 
    - If you want to use a different tokenizer, you can replace the BERT tokenizer with your preferred one. Make sure to update the tokenizer in the **config.py** file.

    - Same goes for the **test and val size**, you can change it in the config.py file.

---

### Model Training & Evaluation

- **Pipeline**

    **src/main.py** manages the training process of the BERT-based SentimentClassifier. It saves the best model, training history, and plots under output/model_output/training. It also evaluates the model on the validation set and saves the results under output/model_output/evaluation.

To Train the Model, Run:

```bash
python src/main.py
```

---

## Inference & Deployment

- **Inference**:

Use *src/inference.py* to predict sentiment from new text inputs. This module is used by the CLI (*src/cli.py*), FastAPI (*src/api.py*), and Streamlit (*src/app.py*) interfaces.

- **CLI**
```bash
python src/cli.py "This is an amazing product!" --model 3-class
```
**Demo**

- **API**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
**Demo**

- **Streamlit**:
Launch the web app:
```bash
streamlit run src/app.py --server.fileWatcherType none
```

**Demo**

- **3-Sentiment Classification Demo**:

https://github.com/user-attachments/assets/e283cf05-d137-4d9a-a9bd-d177e33f5b55

- **5-Sentiment Classification Demo**:

https://github.com/user-attachments/assets/ec03dc7a-9c03-4644-8ccd-a9715ce48f05

The confidence scores that we see next to the prediction are quite high as we can see in the result and evaluation in following part. We have to take it into account for further refinement regarding this project.

--- 

## Docker & MLOps Pipeline

### Dockerfile & Entrypoint

- **Dockerfile**:
Builds the Docker image by installing dependencies, copying project files, and setting up the default entrypoint through entrypoint.sh.

- **entrypoint.sh**:
Determines the mode in which the container will run (API, CLI, or Streamlit) based on the ENTRYPOINT_MODE environment variable.

### Docker Compose

- **docker-compose.yml**:

Defines three services:

- **streamlit_app**: Runs the Streamlit interface on port 8501.

- **fastapi_app**: Runs the FastAPI API on port 8000.

- **cli_app**: Executes the CLI tool.

Each service uses shared volumes for datasets, model outputs, and database persistence.

### Running the Docker Containers

1. **Build the Docker Image**:

```bash
docker build --no-cache -t sentiment-analysis .
```

2. **Start Services**:

```bash
docker-compose up -d
```

3. **Test the Services**:

- **Streamlit**: http://localhost:8501

- **FastAPI**: http://localhost:8000/docs

- **CLI**: Check logs with docker logs sentiment-cli

4. **Stop Services**:

```bash
docker-compose down
```

## Results & Evaluation

- ### 5-Class vs. 3-Class Models:

  - #### 5-Class Results
    
    - **Training**
 
      ![accuracy_and_loss_plot](https://github.com/user-attachments/assets/1b48f18c-520c-4b2a-ad92-1150fc87ae36)

    - **Evaluation**

      ![confusion_matrix](https://github.com/user-attachments/assets/536c4bdf-9a0e-4482-afa8-bf29c27794e3)
      ![classification_report](https://github.com/user-attachments/assets/37dda05c-85d1-423f-a88d-6c51fbcc23c3)
      ![confidence_histogram](https://github.com/user-attachments/assets/2c35c54f-c739-43cd-a5e7-f9b8016aec55)

  - #### 3-Class Results

    - **Training**
 
      ![accuracy_and_loss_plot](https://github.com/user-attachments/assets/53824a92-64e4-4fce-b0a6-b838ce760a34)
      
    - **Evaluation**
   
      ![confusion_matrix](https://github.com/user-attachments/assets/5a084b89-89ac-472f-919b-1cf2f0dbb646)
      ![classification_report](https://github.com/user-attachments/assets/fd0fd91b-dd1e-4bab-bbc2-27bb150d307c)
      ![confidence_histogram](https://github.com/user-attachments/assets/fcc4c37a-4c26-4407-bcd6-0f400cbce76f)
    
Experiments show that a 3-class sentiment model performs better overall, achieving about **77% validation accuracy**, compared to a 5-class model, which reaches only **53% validation accuracy**. However, the 5-class model provides more detailed sentiment analysis, allowing us to distinguish between subtle differences, such as "Really Negative" vs. "Negative."

One key challenge is that the model struggles to predict neutral sentiment. This could be due to:

- **The subjective nature of neutrality** – Reviews might lean slightly positive or negative, making it hard to classify them as truly neutral.
- **Imbalanced data distribution** – If there are fewer neutral reviews in the dataset, the model has less exposure to this class, leading to poor predictions.

The distribution plots further highlight these issues. The first plot shows the merged sentiment distribution for the 3-class model, while the second plot presents the original 5-class sentiment distribution. These distributions suggest that some sentiment categories (basically the neutral one) are **underrepresented**, which could impact model performance.

![merged_distribution_of_review_sentiment](https://github.com/user-attachments/assets/5812a3ad-255b-45c5-a21e-807a486992a3)

![original_distribution_of_review_sentiment](https://github.com/user-attachments/assets/2c2ab198-e2bf-4f7d-9bc3-f5179a22c629)

Additionally, **review length may introduce bias** in sentiment prediction. The KDE (Kernel Density Estimation) plot indicates that review length varies across sentiment classes, suggesting that **longer or shorter reviews might be more strongly associated with certain sentiments**. This bias could influence how the model assigns sentiment labels.

![review_length_distribution_with_kde](https://github.com/user-attachments/assets/89fd22c9-8608-4de9-907a-9736f92400ad)

Finally, the performance gap between the 3-class and 5-class models can be explained by the fact that **fewer classes make predictions easier**. With only three options, the model has a higher chance of guessing correctly, even with minimal understanding. In contrast, a 5-class model requires more nuanced differentiation, making accurate predictions more challenging.

These outputs help in understanding the model's performance and guide further improvements.

---

## Future Improvements

- **Data Augmentation**: 

We can think about using larger datasets and augmentation techniques to improve generalization. Implementing techniques like back-translation, paraphrasing, or word substitution. 

We can also think about handling more languages even if we know that it will be hard to implement due primarily to the lack of data.

- **Hyperparameter Tuning**:

Fine-tune learning rate, dropout, and weight decay parameters.

- **Alternative Models**:

Experiment with models like RoBERTa or DistilBERT for potential performance gains.

- **API Deployment**:

Deploy the model as a RESTful API for real-time sentiment analysis.

- **Enhanced UI**:

Improve the Streamlit app with additional interactive features and visualizations.

---

## References
- [Sentiment Analysis](https://medium.com/analytics-vidhya/simple-sentiment-analysis-python-bf9de2d75d0) 
- [Sentiment Analysis with NLP](https://medium.com/analytics-vidhya/nlp-getting-started-with-sentiment-analysis-126fcd61cc4a)
- [GitHub Cheat Sheet: Git Best Practices](https://education.github.com/git-cheat-sheet-education.pdf)
- [Kaggle Notebook: Sentiment Analysis using BERT](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert)
- [Hugging Face Documentation: AutoTokenizer & AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/index)

---

## **Team Members**
- **[Florian HOUNKPATIN](https://www.linkedin.com/in/florian-hounkpatin/)** : Data Scientist 
- **[Ephraim KOSSONOU](https://www.linkedin.com/in/ephraïm-kossonou/)** : Data Analyst | Engineer
