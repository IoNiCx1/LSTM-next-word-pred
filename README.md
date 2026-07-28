# 🔤 Next-Word Prediction using LSTM

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lstm-llm.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

An interactive deep learning web application that predicts and generates subsequent words given a seed text prompt. Powered by a Long Short-Term Memory (**LSTM**) neural network trained on headline text data and deployed via **Streamlit Cloud**.

🚀 **Live Demo:** [lstm-llm.streamlit.app](https://lstm-llm.streamlit.app/)

---

## 📌 Project Overview

Next-word prediction is a foundational Natural Language Processing (NLP) task used in predictive keyboards, auto-completion engines, and general language modeling. 

This repository covers the complete end-to-end pipeline:
1. **Data Preprocessing & Tokenization:** Cleaning headline sequences and transforming text tokens into input-target training pairs.
2. **Model Architecture:** Embedding layer followed by stacked LSTM layers to capture sequential dependencies over time.
3. **Sampling Strategy:** Temperature-based sampling to prevent standard `argmax` repetition loops and enable controllable creative generation.
4. **Deployment:** Interactive web UI hosted on Streamlit.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.12
* **Frameworks & Libraries:** 
  * `TensorFlow` / `Keras` (Deep Learning architecture)
  * `Streamlit` (Interactive Web Interface)
  * `NumPy` & `Pandas` (Data manipulation & processing)
  * `scikit-learn` (Evaluation & preprocessing tools)
  * `pickle` (Tokenizer serialization)

---

## ⚙️ Local Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/IoNiCx1/LSTM-next-word-pred.git](https://github.com/IoNiCx1/LSTM-next-word-pred.git)
cd LSTM-next-word-preds