import numpy as np
from SimpleNLP.SimpleNLP.transformer import TransformerBlock, LanguageModel, load_data, get_batch
from SimpleNLP.SimpleNLP.transformer import cross_entropy_loss

max_iters = 50000
eval_interval = 100
learning_rate = 1e-3
block_size = 32
batch_size = 16


# Training
def train_text_generation_model():
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


train_text_generation_model()