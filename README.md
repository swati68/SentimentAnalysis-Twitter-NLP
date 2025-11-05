# 💬 Twitter Sentiment Analysis using Neural Networks

This project performs **sentiment analysis of tweets** using deep learning techniques in Python.  
Tweets are classified into three categories — **Positive**, **Negative**, and **Neutral** — using multiple **Recurrent Neural Network (RNN)** and **LSTM**-based architectures.
This work explores various neural models to determine the most effective architecture for understanding public sentiment from Twitter data.

---

## 📘 Project Overview

Microblogging platforms like **Twitter** have become a powerful medium for people to express opinions, emotions, and feedback.  
The goal of this project is to automatically identify the **polarity** (positive, negative, or neutral) of tweets using machine learning models that can process **natural language text**.

### Objectives
- Clean and preprocess real-world tweet data.
- Train and evaluate multiple deep learning models (LSTM, CNN, Bi-LSTM, RNN).
- Identify the best-performing model based on accuracy and F1-score.
- Demonstrate potential use cases such as detecting cyberbullying and analyzing public sentiment.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
The dataset was cleaned using **Regular Expressions**, **Tokenization**, and **Stemming**:
- Removed URLs, mentions (`@username`), hashtags, and escape sequences.
- Converted text to lowercase and removed punctuation/emoticons.
- Applied **Snowball Stemmer** and **Stopword Removal**.
- Normalized repeated characters (e.g., “helllooo” → “hello”).

### 2️⃣ Feature Extraction
- Used **Word2Vec (gensim)** for generating dense word embeddings.
- Tokenized and padded sequences to fixed length (200).
- Embedded sequences were passed to neural models for training.

### 3️⃣ Output Encoding
- Used **LabelEncoder** and **One-Hot Encoding** (`positive=2`, `neutral=1`, `negative=0`).

---

## 🧩 Model Architectures

### 🔹 1. LSTM Model
- 52-node LSTM layer  
- Dropout = 0.2, recurrent dropout = 0.2  
- Dense layer with softmax activation  

### 🔹 2. LSTM + CNN Model
- Conv1D → MaxPooling → LSTM → Dense(Softmax)  
- Combines spatial (CNN) and temporal (LSTM) features.  

### 🔹 3. LSTM + Dense Model
- LSTM (52 nodes) → Dense(32, relu) → Dropout(0.5) → Dense(3, softmax)

### 🔹 4. Bidirectional RNN
- Embedding → BiRNN(32) → Dropout → Dense(3, softmax)  
- Achieved **best performance with F1-score: 62.53%**

### 🔹 5. Bidirectional LSTM
- BiLSTM(64) → Dense(64, relu) → Dense(3, softmax)

### 🔹 6. Stacked BiLSTM
- Two BiLSTM layers (64, 32) → Dense(64, relu) → Dropout(0.5) → Output layer

---

## 📊 Dataset

| Dataset | Rows | Columns | Description |
|----------|-------|----------|--------------|
| Train | 21,630 | tweet_id, sentiment, tweet_text | Used for model training |
| Test | 5,399 | tweet_id, tweet_text | Used for sentiment prediction |

**Sentiment Distribution:**
- Positive: 9,064  
- Neutral: 9,014  
- Negative: 3,387  

---

## ⚙️ Dependencies

```bash
numpy
pandas
gensim
keras
tensorflow
sklearn
matplotlib
nltk
```

---

## 📈 Results
| Model |	Accuracy | F1-Score
|--------|-------|----------|
| LSTM | 60.12% | 58.4% |
| LSTM + CNN | 61.78% | 60.9% |
| BiLSTM | 62.14%	| 61.7% |
| Bidirectional RNN	| 63.02% | 62.53% ✅ |

➡️ The Bidirectional RNN architecture performed best among all tested models.


---

## 💡 Future Work

- Implement Stacked Recurrent Networks for better contextual understanding.
- Experiment with Transformers (BERT / RoBERTa) for improved accuracy.
- Develop a Flask web app for real-time tweet sentiment prediction.
- Apply to cyberbullying and hate-speech detection use cases.
