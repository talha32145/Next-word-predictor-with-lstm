import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model("next_word_predictor (1).keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_resources()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_next_words(seed_text, num_words):

    generated_text = seed_text.lower()

    for _ in range(num_words):

        token_input = tokenizer.texts_to_sequences([generated_text])[0]

        padded_input = pad_sequences(
            [token_input],
            maxlen=11,
            padding="pre"
        )

        pred = model.predict(padded_input, verbose=0)[0]

        # Remove index 0 and 1 if needed
        pred[0] = 0
        pred[1] = 0

        top_k = 5

        indices = np.argsort(pred)[-top_k:]
        probs = pred[indices]
        probs = probs / probs.sum()

        predicted = np.random.choice(indices, p=probs)

        predicted_word = tokenizer.index_word.get(predicted)

        if predicted_word:
            generated_text += " " + predicted_word

    return generated_text

# -----------------------------
# UI
# -----------------------------
st.title("🤖 Next Word Predictor")
st.markdown(
    """
    Generate text using your trained **LSTM Next Word Prediction Model**.
    """
)

seed_text = st.text_area(
    "Enter starting text:",
    placeholder="Example: people"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=20,
    value=5
)

if st.button("🚀 Generate Text"):

    if seed_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Generating..."):
            result = predict_next_words(seed_text, num_words)

        st.success("Generated Successfully!")

        st.subheader("Generated Text")
        st.write(result)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Developed by **Muhammad Talha** | LSTM Next Word Prediction Model"
)