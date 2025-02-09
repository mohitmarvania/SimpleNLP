# 📝 SimpleNLP - Natural language Processing from Scratch

SimpleNLP is an **educational** and **lightweight** natural language processing (NLP) library I have implemented, where key NLP models are implemented using only **NumPy**, **Pandas**, **Scikit-learn** and Python in-built libraries. No external deep learning frameworks like PyTorch and TensorFlow was used! 🚀 This was to understand how NLP processes data from ground level.

This project includes implementations of **tokenization, word embeddings, classifiers, RNNs, and Transformers**, followed by two **real-world NLP tasks**:

1. **Text Classification** 🏷️ (Sentiment Analysis on Twitter Data)
2. **Text Generation** 📝 (Shakespeare-style text generation)

Although the models are **simplified**, they still provide valuable insights into how NLP techniques work under the hood! 💡

---

## 📂 Project Structure

```
SimpleNLP/
│── SimpleNLP/
│   ├── __init__.py
│   ├── tokenization.py      # Tokenizer & Vectorization
│   ├── embeddings.py        # Word2Vec & GloVe
│   ├── classifiers.py       # Logistic Regression, Feedforward NN
│   ├── rnn.py               # Simple RNN implementation
│   ├── transformer.py       # Basic Transformer encoder
│   ├── utils.py             # Helper functions
│── examples/
│   ├── text_classification.py
│   ├── text_generation.py
│── README.md
│── requirements.txt
```

---

## 🚀 Features

✅ **Tokenization & Vectorization** (Word & Char tokenization, NGramTokenizer, Byte-Pair Tokenizer)  
✅ **Word Embeddings** (Word2Vec, TFIDFVectorizer)  
✅ **Machine Learning Classifiers** (Logistic Regression, Feedforward NN)  
✅ **Recurrent Neural Networks (RNN)** (Implemented from scratch)  
✅ **Transformers** (Basic Transformer)  
✅ **Example NLP Tasks** (Sentiment Analysis, Text Generation)

---

## 🏆 Example NLP Tasks

### **📌 1. Text Classification (Twitter Sentiment Analysis)**

- **Dataset**: Twitter Sentiment Analysis (~74,683 training examples, ~1,000 testing examples)
- **Objective**: Classify tweets as positive/negative/neutral
- **Training**: Only trained on **7,000 samples** due to computational constraints
- **Accuracy**: ~66%

📌 **Run Example:**

```bash
python examples/text_classification.py
```

### **📌 2. Text Generation (Shakespeare-Style)**

- **Objective**: Generate Shakespeare-style text using a Transformer-based model
- **Dataset**: Shakespeare corpus
- **Challenges**:
- Transformer was implemented from scratch using NumPy.
- No automatic differentiation or deep learning optimizers used.
- Text quality is not perfect, but it generates somewhat meaningful content.
  📌 Run Example:

```bash
python examples/text_generation.py
```

---

🛠 Installation
1️⃣ Clone the repository

```
git clone https://github.com/mohitmarvania/SimpleNLP.git
cd SimpleNLP
```

2️⃣ Install dependencies

```
pip install -r requirements.txt
```

3️⃣ Run example scripts

```
python examples/text_classification.py
python examples/text_generation.py
```

---

📚 Dependencies
Python 3.x
NumPy
Pandas
Scikit-learn
📌 Install all dependencies using:

```
pip install -r requirements.txt
```

---

🚀 Future Improvements
✅ Implement Attention Mechanism in Transformer
✅ Add LSTM and GRU models
✅ Improve Optimizers for better training
✅ Expand NLP applications (e.g., Named Entity Recognition)

🤝 Contributing
Contributions are welcome! Feel free to open issues and submit pull requests. 🛠️

📜 License
This project is licensed under the MIT License.
