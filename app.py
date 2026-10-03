import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("next_word_lstm.h5")


# Load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model, tokenizer, text):

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])[0]

    # Model's expected input length
    max_length = model.input_shape[1]

    # Keep only the last required words
    sequence = sequence[-max_length:]

    # Pad sequence
    sequence = pad_sequences(
        [sequence],
        maxlen=max_length,
        padding="pre"
    )

    # Predict
    predicted = model.predict(sequence, verbose=0)[0]

    # Get predicted word index
    predicted_word_index = np.argmax(predicted)

    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None


# Streamlit UI
st.title("Next Word Predictor with LSTM")

text = st.text_input(
    "Enter the text",
    "To be or not to"
)


if st.button("Predict Next Word"):

    next_word = predict_next_word(
        model,
        tokenizer,
        text
    )

    st.write("Next Word:", next_word)
