
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_lstm.h5")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model,tokenizer, text,max_sequence_len):
  sequence = tokenizer.texts_to_sequences([text])[0]
  if len(sequence)>=max_sequence_len:
    sequence = sequence[-(max_sequence_len-1):]
  sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
  predicted = model.predict(sequence, verbose=0)
  predicted_word = np.argmax(predicted, axis=1)
  
  for word, index in tokenizer.word.items():
    if index == predicted_word:
      return word
      
    return None


st.title("Next Word Predictor with LSTM")
text = st.text_input("Enter the text", "To be or not to")
if st.button("Predict Next Word"):
  next_word=predict_next_word(model, tokenizer,text,model.input_shape[1]+1)
  st.write("Next Word: ", next_word)
