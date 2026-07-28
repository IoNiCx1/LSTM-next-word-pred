import streamlit as st
import pickle
import numpy as np 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('nextword_model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# FIX 1: Use .items() instead of .item()
reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}

# Match the max_len used during model training (input sequence length)
max_len = 44  # If trained with sequence length = max_len - 1, adjust accordingly

def generate_text(seed_text, num_words=10):
    text = seed_text 
    for _ in range(num_words):
        # FIX 2: Convert text to sequence and pad correctly
        seq = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([seq], maxlen=max_len, padding='pre')
        
        preds = model.predict(padded, verbose=0)
        pos = np.argmax(preds)
        
        next_word = reverse_index.get(pos, "")
        if not next_word:
            break
            
        text += " " + next_word


    return text

st.title("Next Word Pred with Deep learning")
seed = st.text_input('Enter Starting text', 'Hello')
num_words = st.slider('No of words to predict', 1, 15, 10)

if st.button("Generate Text"):
    result = generate_text(seed, num_words)
    st.write(result)