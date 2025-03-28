import torch
import streamlit as st

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import *
from src.inference import predict_sentiment


# ------------------------ 🚀 Streamlit App ------------------------
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
            try:
                # Load model and tokenizer
                device = torch.device(DEVICE)
                prediction, confidence = predict_sentiment(
                    text=user_input,
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

                # Get sentiment label & matching color
                sentiment = sentiment_labels.get(prediction, "Unknown sentiment")
                color = COLOR_MAPPING.get(sentiment, "black")

                # Display result with color formatting
                st.markdown(
                    f"<h3 style='text-align: center; color: {color};'>Predicted Sentiment: {sentiment} at {confidence:.2f}% confidence</h3>",
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Please enter some text to analyze!")


if __name__ == "__main__":
    main()

# To run the streamlit app, use:
# streamlit run .\src\app.py --server.fileWatcherType none
