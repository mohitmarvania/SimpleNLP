# Implementing basic text classification
import numpy as np
from SimpleNLP.SimpleNLP.tokenization import Tokenizer, BPETokenizer, NGramTokenizer
from SimpleNLP.SimpleNLP.embeddings import Word2Vec, TFIDFVectorizer
from SimpleNLP.SimpleNLP.classifiers import LogisticRegressionClassifier, FeedForwardNN
from SimpleNLP.SimpleNLP.utils import load_data, preprocess_text

# Load dataset (Example: Sentiment Analysis)
data = [
    ("I love this movie!", "positive"),
    ("This film was terrible.", "negative"),
    ("Absolutely fantastic experience!", "positive"),
    ("Worst thing I have ever watched.", "negative"),
    ("I really enjoyed this!", "positive"),
    ("Not great, quite disappointing.", "negative"),
]

texts, labels = zip(*data)

# Tokenization
tokenizer = Tokenizer(mode="word")
tokenized_texts = [tokenizer.tokenize(text) for text in texts]

# Choose an embedding method (TF-IDF or One-Hot)
vectorizer = TFIDFVectorizer()
# vectorizer = OneHotEncoding()  # Alternative

# Fit vectorizer on the dataset and transform text into vectors
vectorizer.fit(tokenized_texts)
X = np.array([vectorizer.transform(sentence) for sentence in tokenized_texts])

# Encode labels
unique_labels = list(set(labels))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
y = np.array([label_to_idx[label] for label in labels])

# Train classifier
classifier = LogisticRegressionClassifier()
classifier.fit(X, y)

# Predict on some sample texts
sample_texts = ["I'm feeling a bit down today.", "I can't believe how much I enjoyed that book."]
sample_tokens = [tokenizer.tokenize(text) for text in sample_texts]
X_sample = np.array([vectorizer.transform(sentence) for sentence in sample_tokens])

predictions = classifier.predict(X_sample)
predicted_labels = [unique_labels[p] for p in predictions]

# Print results
for text, label in zip(sample_texts, predicted_labels):
    print(f"Text: {text} → Prediction: {label}")

