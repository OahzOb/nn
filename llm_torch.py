"""
llm_torch.py — From-scratch LLM with PyTorch on Intel Arc B580
==============================================================

Builds a complete LLM pipeline from scratch using only PyTorch primitives:
  1. CharTokenizer    — character-level tokenizer (no HuggingFace)
  2. RMSNorm          — root-mean-square layer normalization
  3. MultiHeadAttention — causal multi-head self-attention
  4. SwiGLU FFN       — feed-forward network with SwiGLU activation
  5. TransformerDecoderBlock — pre-norm residual block
  6. TransformerLM    — full decoder-only language model
  7. Training loop    — autoregressive next-token prediction
  8. Text generation  — temperature / top-k sampling

Device: auto-selects XPU for Intel Arc, falls back to CPU.
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# Device setup
# ═══════════════════════════════════════════════════════════════

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "xpu":
    print(f"  GPU: {torch.xpu.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════
# 1. Tokenizer — character-level, built from scratch
# ═══════════════════════════════════════════════════════════════

class CharTokenizer:
    """Character-level tokenizer.

    Maps each unique character in the training corpus to an integer id.
    Provides encode() / decode() for text ↔ token-id conversion.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        # Reserve 0 for padding/unknown if needed (not used here but common)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


# ═══════════════════════════════════════════════════════════════
# 2. Model building blocks (from scratch using only nn.Linear etc.)
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Ref: https://arxiv.org/abs/1910.07467
    Faster than LayerNorm: no mean-centering, just RMS scaling.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    """Causal multi-head scaled dot-product attention.

    Splits d_model into n_heads heads of dimension d_head = d_model // n_heads.
    Applies causal masking so each position can only attend to itself and
    earlier positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Linear projections → (B, T, n_heads, d_head) → (B, n_heads, T, d_head)
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, nh, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, nh, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network.

    SwiGLU(x) = (x·W1) ⊙ SiLU(x·W3)  → projected by W2.
    Shown to outperform standard ReLU FFN in PaLM / Llama architectures.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2( SiLU(w1(x)) * w3(x) )
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerDecoderBlock(nn.Module):
    """One decoder block: self-attention + feed-forward with pre-norm residual.

    Architecture: x → RMSNorm → MHA → + → RMSNorm → FFN → +
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Pre-norm residual stream — standard in modern LLMs (GPT-2+, Llama)
        x = x + self.dropout(self.attn(self.attn_norm(x), mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


# ═══════════════════════════════════════════════════════════════
# 3. Full decoder-only language model
# ═══════════════════════════════════════════════════════════════

class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model.

    Architecture:
      token_embed → +pos_embed → [decoder_block × n_layers] → RMSNorm → lm_head

    Weight tying: token_embed.weight == lm_head.weight (reduces params,
    improves convergence — Press & Wolf 2017).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        self.max_seq_len = max_seq_len

        # Pre-computed causal mask: lower-triangular (T, T)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, max_seq_len, max_seq_len)),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max {self.max_seq_len}"
        )

        # Token + learned position embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.token_embedding(x) + self.pos_embedding(positions)

        # Causal mask (trimmed to current length)
        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T)

        for block in self.blocks:
            h = block(h, mask)

        h = self.norm(h)
        logits = self.lm_head(h)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Autoregressive text generation.

        Args:
            tokenizer: CharTokenizer for encode/decode.
            prompt: Input text to start generation.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (>1 more random, <1 more greedy).
            top_k: If set, only sample from the top-k highest probability tokens.

        Returns:
            Generated text including the prompt.
        """
        self.eval()
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], device=next(self.parameters()).device
        )

        for _ in range(max_new_tokens):
            # Crop to max_seq_len (sliding window)
            x = input_ids[:, -self.max_seq_len :]

            logits, _ = self.forward(x)
            logits = logits[:, -1, :] / temperature  # last timestep

            # Top-k filtering
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)

        self.train()
        return tokenizer.decode(input_ids[0].tolist())


# ═══════════════════════════════════════════════════════════════
# 4. Data preparation
# ═══════════════════════════════════════════════════════════════

# A small but structured corpus so the model has learnable patterns
CORPUS = """
Hello world. This is a tiny language model.
It learns from scratch using PyTorch.
The model can generate text after training.
Deep learning is very interesting.
Natural language processing is fun.
Transformers are powerful for text generation.

The quick brown fox jumps over the lazy dog.
She sells sea shells by the sea shore.
How much wood would a woodchuck chuck.

Machine learning models learn from data.
A neural network has many layers.
Each layer transforms the input data.
The model predicts the next word.

Now is the winter of our discontent.
All that glitters is not gold.
To be or not to be, that is the question.
What's in a name? That which we call a rose.
"""

# Build tokenizer from corpus
tokenizer = CharTokenizer(CORPUS)
vocab_size = tokenizer.vocab_size
print(f"\nVocabulary size: {vocab_size}")
print(f"Characters: {repr(''.join(tokenizer.stoi.keys()))}")

# Encode full corpus into a 1D tensor
data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
print(f"Total tokens in corpus: {len(data)}")

# Train / validation split (90/10)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]


# ═══════════════════════════════════════════════════════════════
# 5. Hyperparameters & model instantiation
# ═══════════════════════════════════════════════════════════════

model_cfg = dict(
    vocab_size=vocab_size,
    d_model=64,  # embedding / hidden dimension
    n_heads=4,  # number of attention heads
    n_layers=3,  # number of decoder blocks
    d_ff=256,  # feed-forward inner dimension
    max_seq_len=64,  # maximum context length
    dropout=0.1,
)

train_cfg = dict(
    batch_size=32,
    learning_rate=3e-3,
    epochs=500,
    eval_interval=500,
    grad_clip=1.0,
)

model = TransformerLM(**model_cfg).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print(
    f"Breakdown:\n"
    f"  Embedding:   {sum(p.numel() for p in model.token_embedding.parameters()):>6,}\n"
    f"  Pos Embed:   {sum(p.numel() for p in model.pos_embedding.parameters()):>6,}\n"
    f"  Transformer: {sum(p.numel() for p in model.blocks.parameters()):>6,}\n"
    f"  LM Head:     {sum(p.numel() for p in model.lm_head.parameters()):>6,}"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])


# ═══════════════════════════════════════════════════════════════
# 6. Training loop
# ═══════════════════════════════════════════════════════════════

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of random sequences for next-token prediction."""
    source = train_data if split == "train" else val_data
    max_start = len(source) - model.max_seq_len - 1
    starts = torch.randint(0, max_start, (train_cfg["batch_size"],))

    xs = torch.stack([source[i : i + model.max_seq_len] for i in starts])
    ys = torch.stack(
        [source[i + 1 : i + model.max_seq_len + 1] for i in starts]
    )
    return xs.to(device), ys.to(device)


print(f"\n{'─' * 50}")
print(f"Training {total_params:,}-param model on {device}")
print(f"{'─' * 50}")

train_losses: list[float] = []
val_losses: list[float] = []
val_steps: list[int] = []
start_time = time.time()

for step in range(1, train_cfg["epochs"] + 1):
    model.train()
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
    optimizer.step()

    train_losses.append(loss.item())

    # Evaluation
    if step % train_cfg["eval_interval"] == 0 or step == 1:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch("val")
            _, val_loss = model(xv, yv)

        val_losses.append(val_loss.item())
        val_steps.append(step)
        elapsed = time.time() - start_time

        print(
            f"step {step:5d}/{train_cfg['epochs']:5d}  "
            f"train loss {loss.item():.4f}  "
            f"val loss {val_loss.item():.4f}  "
            f"perplexity {math.exp(val_loss.item()):.2f}  "
            f"time {elapsed:.1f}s"
        )

total_time = time.time() - start_time
print(f"\nTraining complete! {total_time:.2f}s ({total_time/60:.1f} min)")


# ═══════════════════════════════════════════════════════════════
# 7. Text generation — test what the model learned
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═' * 50}")
print("TEXT GENERATION")
print(f"{'═' * 50}")

prompts = [
    "Hello",
    "To be",
    "The model",
    "Deep learning",
    "A neural",
]

for prompt in prompts:
    generated = model.generate(
        tokenizer,
        prompt=prompt,
        max_new_tokens=80,
        temperature=0.8,
        top_k=10,
    )
    print(f"\nPrompt: '{prompt}'")
    print(f"Output: {generated}")


# ═══════════════════════════════════════════════════════════════
# 8. Temperature sweep — show how temperature affects randomness
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═' * 50}")
print("TEMPERATURE SWEEP (prompt: 'The model')")
print(f"{'═' * 50}")

for temp in [0.2, 0.8, 1.5]:
    out = model.generate(
        tokenizer, prompt="The model", max_new_tokens=60,
        temperature=temp, top_k=10,
    )
    print(f"\nt={temp:.1f}: {out}")


# ═══════════════════════════════════════════════════════════════
# 9. Loss plot
# ═══════════════════════════════════════════════════════════════

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train loss", alpha=0.6, linewidth=0.5)
plt.axvline(
    x=split_idx,
    color="gray",
    linestyle="--",
    alpha=0.3,
)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_steps, val_losses, marker="o", markersize=4, label="Val loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Validation Loss & Perplexity")

# Add perplexity axis
ax2 = plt.gca().twinx()
ax2.plot(val_steps, [math.exp(l) for l in val_losses], "r--", alpha=0.4)
ax2.set_ylabel("Perplexity", color="r")
ax2.tick_params(axis="y", labelcolor="r")

plt.tight_layout()
plt.show()

print("\nDone — full LLM pipeline from scratch validated on Intel Arc B580!")
