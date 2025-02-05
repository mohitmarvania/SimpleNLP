import numpy as np


# Transformer based model
class Transformer:
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=128, dropout=0.0, learning_rate=1e-3):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.learning_rate = learning_rate

        # Initialize embeddings
        self.token_embedding = np.random.randn(vocab_size, n_embd) * 0.01
        self.position_embedding = np.random.randn(block_size, n_embd) * 0.01

        # Transformer block
        self.layers = [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]

        # Final layer norm and output projection
        self.layer_norm = LayerNorm(n_embd)
        self.lm_head = np.random.randn(n_embd, vocab_size) * 0.01

    def forward(self, idx):
        """
        Forward propagation through the transformer model.
        """
        B, T = idx.shape

        # Token + Positional embeddings
        tok_emb = self.token_embedding[idx]  # (B, T, C)
        pos_emb = self.position_embedding[np.arange(T)]  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer.forward(x)

        # Layer normalization and projection
        x = self.layer_norm.forward(x)
        logits = np.dot(x, self.lm_head)  # (B, T, vocab_size)

        return logits

    def backward(self, idx, target):
        """
                Backpropagation for the transformer model.
                """
        B, T = idx.shape

        # Forward pass to get logits
        logits = self.forward(idx)  # (B, T, vocab_size)

        # Compute softmax probabilities
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)

        # Compute loss (cross-entropy)
        loss = -np.log(probs[np.arange(B)[:, None], np.arange(T), target]).mean()

        # Compute gradient of loss w.r.t logits
        dlogits = probs
        dlogits[np.arange(B)[:, None], np.arange(T), target] -= 1
        dlogits /= B

        # Backpropagate through lm_head
        d_x = np.dot(dlogits, self.lm_head.T)
        d_lm_head = np.dot(d_x.reshape(-1, self.n_embd).T, dlogits.reshape(-1, self.vocab_size))

        # Backpropagate through transformer blocks
        for layer in reversed(self.layers):
            d_x = layer.backward(d_x)

        # Backpropagate through embeddings
        d_token_embedding = np.zeros_like(self.token_embedding)
        np.add.at(d_token_embedding, idx, d_x)

        # Gradient descent updates
        self.lm_head -= self.learning_rate * d_lm_head
        self.token_embedding -= self.learning_rate * d_token_embedding

        return loss

    def generate(self, seed_idx, max_new_tokens, temperature=0.8):
        """
        Generate text using the transformer model.
        """
        idx = np.array(seed_idx).reshape(1, -1)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

            idx_next = np.random.choice(range(self.vocab_size), p=probs.ravel())
            idx = np.hstack((idx, np.array([[idx_next]])))

        return idx


class TransformerBlock:
    """
    A single transformer block with multi-head self-attention and feedforward layers.
    """

    def __init__(self, n_embd, n_head, dropout):
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.mha = MultiHeadAttention(n_head, self.head_size, n_embd, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha.forward(self.ln1.forward(x))
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x


class MultiHeadAttention:
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd

        # Initialize weights
        self.Wq = np.random.randn(n_embd, n_embd) * 0.01
        self.Wk = np.random.randn(n_embd, n_embd) * 0.01
        self.Wv = np.random.randn(n_embd, n_embd) * 0.01
        self.Wo = np.random.randn(n_embd, n_embd) * 0.01

        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V
        Q = np.dot(x, self.Wq)  # (B, T, C)
        K = np.dot(x, self.Wk)  # (B, T, C)
        V = np.dot(x, self.Wv)  # (B, T, C)

        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_size)  # (B, T, T)

        # Casual mask: prevent attending to future tokens
        mask = np.triu(np.ones((T, T)), k=1)  # For upper triangle in matrix
        scores = np.where(mask == 1, -np.inf, scores)

        # Softmax and dropout
        scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        scores = scores / np.sum(scores, axis=-1, keepdims=True)

        # Weighted sum of values
        out = np.matmul(scores, V)  # (B,T,C)
        out = np.dot(out, self.Wo)  # (B,T,C)

        return out


class FeedForward:
    """
    A simple feedforward network with ReLU activation.
    """

    def __init__(self, n_embd, dropout):
        self.W1 = np.random.randn(n_embd, 4 * n_embd) * 0.01
        self.W2 = np.random.randn(4 * n_embd, n_embd) * 0.01
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape  # Extract batch, sequence length and channel

        # Reshape x for maxtrix multiplication
        x = x.reshape(B * T, C)  # Flatten sequence: (B*T, C)
        x = np.dot(x, self.W1)  # (B*T, 256)
        x = np.maximum(0, x)  # ReLU activation
        x = np.dot(x, self.W2)  # (B*T, 64)

        # Reshape back to (B,T,C)
        x = x.reshape(B, T, C)
        return x


class LayerNorm:
    """
    Layer Normalization.
    """

    def __init__(self, n_embd, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((n_embd,))
        self.beta = np.zeros((n_embd,))

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# Example text corpus
text = "This is the text generated with AI and thus it can only be generated by AI."

# Tokenization
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Encode input text
encoded_text = np.array([[stoi[ch] for ch in text[:32]]])

# Initialize model
model = Transformer(vocab_size)

# Generate text
generated_tokens = model.generate(encoded_text, max_new_tokens=100, temperature=0.7)

# Decode output
generated_text = ''.join([itos[i] for i in generated_tokens[0]])
print("Generated Text:\n", generated_text)