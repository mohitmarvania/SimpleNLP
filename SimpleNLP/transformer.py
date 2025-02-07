import numpy as np

# Hyperparameters
block_size = 32
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0


class Layer_Norm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class Attention:
    """Single head attention"""
    def __init__(self, n_embd, head_size, block_size):
        self.key = np.random.randn(n_embd, head_size) * 0.02
        self.query = np.random.randn(n_embd, head_size) * 0.02
        self.value = np.random.randn(n_embd, head_size) * 0.02
        self.tril = np.tril(np.ones((block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = x @ self.key  # (B, T, head_size)
        q = x @ self.query  # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(0, 2, 1) * (C ** -0.5)  # (B, T, T)
        wei = wei * self.tril[:T, :T]  # (B, T, T)
        wei = np.exp(wei) / np.sum(np.exp(wei), axis=-1, keepdims=True)  # softmax

        # weighted aggregation
        v = x @ self.value  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention:
    def __init__(self, n_embd, n_head, block_size):
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        self.heads = [Attention(n_embd, self.head_size, block_size) for _ in range(n_head)]
        self.proj = np.random.randn(n_embd, n_embd) * 0.02

    def forward(self, x):
        out = np.concatenate([h.forward(x) for h in self.heads], axis=-1)
        out = out @ self.proj
        return out


class FeedForward:
    def __init__(self, n_embd):
        self.W1 = np.random.randn(n_embd, 4 * n_embd) * 0.02
        self.W2 = np.random.randn(4 * n_embd, n_embd) * 0.02

    def forward(self, x):
        x = x @ self.W1
        x = np.maximum(0, x)  # ReLU
        x = x @ self.W2
        return x


class TransformerBlock:
    def __init__(self, n_embd, n_head, block_size):
        self.sa = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = Layer_Norm(n_embd)
        self.ln2 = Layer_Norm(n_embd)

    def forward(self, x):
        x = x + self.sa.forward(self.ln1.forward(x))
        x = x + self.ffwd.forward(self.ln2.forward(x))
        return x


class LanguageModel:
    def __init__(self, vocab_size, block_size):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = np.random.randn(vocab_size, n_embd) * 0.02
        self.position_embedding = np.random.randn(block_size, n_embd) * 0.02
        self.blocks = [TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        self.ln_f = Layer_Norm(n_embd)
        self.lm_head = np.random.randn(n_embd, vocab_size) * 0.02

    def forward(self, idx):
        B, T = idx.shape

        # Get embeddings
        tok_emb = self.token_embedding[idx]  # (B,T,C)
        pos_emb = self.position_embedding[:T]  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # Apply transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        x = self.ln_f.forward(x)
        logits = x @ self.lm_head  # (B,T,vocab_size)
        return logits

    def generate(self, context, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # Crop context if needed
            context_cond = context[:, -self.block_size:]
            # Get predictions
            logits = self.forward(context_cond)
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            # Apply softmax
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            # Sample from distribution
            idx_next = np.random.choice(len(probs[0]), p=probs[0])
            # Append to the sequence
            context = np.concatenate([context, np.array([[idx_next]])], axis=1)
        return context


def cross_entropy_loss(logits, targets):
    """
    Compute cross entropy loss
    logits: (B*T, C)
    targets: (B*T,)
    """
    B, T, C = logits.shape
    logits = logits.reshape(-1, C)
    targets = targets.reshape(-1)

    # Compute softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Compute cross entropy
    log_probs = -np.log(probs[np.arange(len(targets)), targets])
    loss = np.mean(log_probs)
    return loss


# Data loading and preprocessing
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode text
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = np.array(encode(text))
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, encode, decode, vocab_size


def get_batch(data, block_size, batch_size):
    ix = np.random.randint(len(data) - block_size, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# Training
if __name__ == "__main__":
    # Load and preprocess data
    train_data, val_data, encode, decode, vocab_size = load_data('../../input.txt')

    # Initialize model
    model = LanguageModel(vocab_size, block_size)

    # Training loop
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            print(f'step {iter}')

        # Get batch
        xb, yb = get_batch(train_data, block_size, batch_size)

        # Forward pass
        logits = model.forward(xb)
        loss = cross_entropy_loss(logits, yb)

        # Compute gradients (simplified)
        B, T, C = logits.shape
        grad_logits = np.zeros_like(logits)
        grad_logits = grad_logits.reshape(-1, C)
        grad_logits[np.arange(B * T), yb.reshape(-1)] = -1.0 / B / T
        grad_logits = grad_logits.reshape(B, T, C)

        # Update lm_head with proper broadcasting
        grad_lm = np.mean(grad_logits.transpose(0, 2, 1) @ model.ln_f.forward(
            model.blocks[-1].forward(model.token_embedding[xb] + model.position_embedding[:T])), axis=0)
        model.lm_head -= learning_rate * grad_lm.T

        if iter % eval_interval == 0:
            print(f'step {iter}: loss {loss:.4f}')

    # Generate text
    context = np.zeros((1, 1), dtype=int)
    generated_text = model.generate(context, max_new_tokens=500, temperature=0.8)
    print(decode(generated_text[0].tolist()))