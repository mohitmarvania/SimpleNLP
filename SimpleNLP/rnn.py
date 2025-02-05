import numpy as np


# Simple RNN for Text Generation
class SimpleRNN:
    def __init__(self, vocab_size, hidden_size=100, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Initialize Weights
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01  # Hidden to output
        self.bh = np.zeros((hidden_size, 1))  # Bias for hidden state
        self.by = np.zeros((vocab_size, 1))  # Bias for output

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, inputs, h_prev=None):
        xs, ys, ps = {}, {}, {}

        # Ensure h_prev is initialized
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))

        hs = np.zeros((len(inputs) + 1, self.hidden_size, 1))
        hs[0] = h_prev  # Initialize first hidden state

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))  # One-hot encoding
            xs[t][inputs[t]] = 1

            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = self.softmax(ys[t])

        return xs, hs[1:], ys, ps

    def backward(self, xs, hs, ps, targets):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))  # Correct hidden state initialization

        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # Gradient of softmax loss
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, xs[t].T)
            dWhh += np.dot(dh_raw, hs[t - 1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)

        # Clipping gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby

    def update_weights(self, dWxh, dWhh, dWhy, dbh, dby):
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def train(self, data, char_to_idx, idx_to_char, epochs=100, seq_length=25):
        h_prev = np.zeros((self.hidden_size, 1))
        for epoch in range(epochs):
            loss = 0
            for i in range(0, len(data) - seq_length, seq_length):
                inputs = [char_to_idx[ch] for ch in data[i:i + seq_length]]
                targets = [char_to_idx[ch] for ch in data[i + 1:i + seq_length + 1]]

                # Forward and Backward Pass
                xs, hs, ys, ps = self.forward(inputs, h_prev)
                dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ps, targets)

                # Update Weights
                self.update_weights(dWxh, dWhh, dWhy, dbh, dby)

                # Compute Loss
                loss += -sum(np.log(ps[t][targets[t], 0]) for t in range(len(inputs)))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def generate_text(self, seed_char, char_to_idx, idx_to_char, length=100):
        h = np.zeros((self.hidden_size, 1))
        x = np.zeros((self.vocab_size, 1))
        x[char_to_idx[seed_char]] = 1
        generated_text = seed_char

        for _ in range(length):
            _, h, _, ps = self.forward([char_to_idx[generated_text[-1]]], h)
            idx = np.random.choice(range(self.vocab_size), p=ps[len(ps) - 1].ravel())  # Fix index issue
            generated_text += idx_to_char[idx]
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

        return generated_text


# Sample Training Data
text_data = "hello world! this is a simple rnn for text generation."
chars = list(set(text_data))  # Unique characters in the text
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Initialize & Train RNN
rnn = SimpleRNN(vocab_size=len(chars), hidden_size=100, learning_rate=0.01)
rnn.train(text_data, char_to_idx, idx_to_char, epochs=120, seq_length=10)

# Generate Text
generated_text = rnn.generate_text(seed_char="h", char_to_idx=char_to_idx, idx_to_char=idx_to_char, length=100)
print("Generated Text:", generated_text)
