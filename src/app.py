import torch
import streamlit as st

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import *
from src.db_logger import log_prediction
from src.inference import predict_sentiment


# --------------------------- 🚀 Streamlit App ---------------------------
def main():
    st.set_page_config(page_title=APP_NAME, layout="centered")

    st.markdown(
        f"<h1 class='title'>💬 Sentiment Analysis App</h1>", unsafe_allow_html=True
    )

    # Sidebar Model Selection
    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.radio(
        "Select Sentiment Model:",
        options=["3-class model", "5-class model"],
        index=0,
    )

    # Assign n_classes
    n_classes = 3 if model_choice == "3-class model" else 5

    # Assign model path
    model_path = (
        PRETRAINED_MODEL_3_CLASS_PATH
        if model_choice == "3-class model"
        else PRETRAINED_MODEL_5_CLASS_PATH
    )

    # Text Input Box
    user_input = st.text_area("✏️ Enter text for sentiment analysis:", "")

    # Button to analyze sentiment
    if st.button("🔍 Analyze Sentiment"):
        if user_input:

            texts = [t.strip() for t in user_input.split("||") if t.strip()]

            try:
                # Load model and tokenizer
                device = torch.device(DEVICE)
                predictions, confidences = predict_sentiment(
                    text=texts,
                    path_to_model=model_path,
                    device=device,
                    n_classes=n_classes,
                )

                # Select appropriate label set
                sentiment_labels = (
                    SENTIMENT_MAPPING_3_LABEL_VERSION
                    if model_choice == "3-class model"
                    else SENTIMENT_MAPPING
                )

                st.subheader("Results:")
                for idx, (text, pred, conf) in enumerate(
                    zip(texts, predictions, confidences)
                ):
                    sentiment = sentiment_labels.get(pred, "Unknown sentiment")
                    color = COLOR_MAPPING.get(
                        sentiment, "black"
                    )  # Get color, default to black if missing

                    # Log the prediction
                    log_prediction(text, pred, conf, "Streamlit App", n_classes)

                    # Display results with color
                    st.markdown(f"**Text {idx + 1}:** {text}")
                    st.markdown(
                        f"<span style='color:{color}; font-weight:bold;'>Predicted Sentiment: {sentiment} ({conf:.2f}% confidence)</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Please enter some text to analyze!")


if __name__ == "__main__":
    main()

# To run the streamlit app, use:
# streamlit run .\src\app.py --server.fileWatcherType none
