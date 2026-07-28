
import streamlit as st
import pickle

import numpy as np 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('nextword_model.h5')
with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx:word for word,idx in tokenizer.word_index.item()}

max_len = 44

def generate_text(seed_text,num_words=10):
    text = seed_text 
    for _ in range(num_words):
        seq= tokenizer.texts_to_sequences([text])
        padded = pad_sequences([seq],maxlen = max_len,padding = 'pred')
        preds =model.predict(padded,verbose = 0)
        pos= np.argmax(preds)
        next_word = reverse_index.get(pos," ")
        text+=" "+next_word
        return text