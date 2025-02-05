import numpy as np
from collections import Counter
import math
import random


# One-Hot Encoding
class OneHotEncoding:
    def __init__(self):
        self.vocab = {}

    def fit(self, corpus):
        words = set(word for sentence in corpus for word in sentence)
        self.vocab = {word: i for i, word in enumerate(words)}

    def transform(self, sentence):
        vector = np.zeros(len(self.vocab))
        for word in sentence:
            if word in self.vocab:
                vector[self.vocab[word]] = 1
        return vector


# Term Frequency - Inverse Document Frequency (TF-IDF)
class TFIDFVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit(self, corpus):
        # create vocabulary
        word_set = set(word for sentence in corpus for word in sentence)
        self.vocab = {word: i for i, word in enumerate(word_set)}

        # compute IDF
        doc_count = {word: 0 for word in self.vocab}
        for sentence in corpus:
            unique_words = set(sentence)
            for word in unique_words:
                doc_count[word] += 1

        total_docs = len(corpus)
        self.idf = {word: math.log(total_docs / (doc_count[word] + 1)) for word in self.vocab}

    def transform(self, sentence):
        tf = Counter(sentence)
        tfidf_vector = np.zeros(len(self.vocab))

        for word, count in tf.items():
            if word in self.vocab:
                tfidf_vector[self.vocab[word]] = count * self.idf[word]

        return tfidf_vector


# Word2Vec (Implementing Skip-Gram Model)
class Word2Vec:
    def __init__(self, vocab_size, embed_size=10, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.W1 = np.random.rand(vocab_size, embed_size)
        self.W2 = np.random.rand(embed_size, vocab_size)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def train(self, pairs, steps=1000):
        for step in range(steps):
            loss = 0
            for center, context in pairs:
                # one-hot encode input
                x = np.zeros(self.vocab_size)
                x[center] = 1

                # Forward pass
                hidden = np.dot(x, self.W1)  # Projection layer
                output = np.dot(hidden, self.W2)  # Output layer
                probs = self.softmax(output)

                # Compute loss
                loss += -np.log(probs[context])

                # Backpropagation
                d_output = probs
                d_output[context] -= 1
                d_hidden = np.dot(self.W2, d_output)

                self.W2 -= self.learning_rate * np.outer(hidden, d_output)
                self.W1 -= self.learning_rate * np.outer(x, d_hidden)

            if steps > 100 and step % 100 == 0:
                print(f"Step {step} | Loss: {loss:.4f}")

    def get_embeddings(self, word_idx):
        return self.W1[word_idx]