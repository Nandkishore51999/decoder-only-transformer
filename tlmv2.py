"""
Tiny Language Model

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda"
torch.set_default_device("cuda")

batch_size = 32  # B
block_size = 8  # T
n_embd = 768
n_heads = 8
head_size = int(n_embd/n_heads)
n_layer = 6

dropout = 0.2
learning_rate = 3e-4

max_iters = 1_000
eval_iters = 100
eval_interval = int(0.1*max_iters)


with open(r"C:\Workspace-ML\text_data\BM\ISS.txt", encoding="utf-8") as file:
    text_data = file.read()

vocab = sorted(list(set(text_data)))
vocab = "".join(vocab)
vocab_size = len(vocab)

ch2id = {}
id2ch = {}
for idx, ele in enumerate(vocab):
    ch2id[ele] = idx
    id2ch[idx] = ele


def encode(text):
    return [ch2id[char] for char in text]


def decode(tokens):
    return "".join([id2ch[token] for token in tokens])


text_data_tokens = torch.tensor(encode(text_data), dtype=torch.long)

n = int(0.9*len(text_data_tokens))
train_data = text_data_tokens[:n]
val_data = text_data_tokens[n:]


def get_batch(split):
    """
    torch.randint: return a tensor filled with random integers generated uniformly between low and high
        low: optional, default = 0
        high,
        size

    given a batch_size,
    creates a list of size batch_size: elements are random int between 0 and [len(data) - block_size]

    torch.stack: concatenates a sequence of tensors along a NEW dimension.

    """

    # split: train / val / test
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        """
        tril is not a parameter of module (torch.nn.module), its a buffer (as per pytorch naming convention) 
        so to make it part of model's parameters, we call register_buffer function
        
        self.tril = torch.tril(...) ===> Normal Python attribute. Doesn't move to GPU with model. Not saved.
        register_buffer(...)        ===> Non-trainable, moves with model, saved in .state_dict().
        """
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B, T, C)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x)  # (B,T,C)
        output = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.dropout(self.proj(output))
        return output


class FFNN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, head_size):
        super().__init__()
        self.attention_head = MultiHeadAttention(n_heads, head_size)
        self.ffnn = FFNN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention_head(self.ln1(x))
        x = x + self.ffnn(self.ln2(x))
        return x


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_enc_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, head_size) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embeding_table(idx)
        pos_enc = self.pos_enc_table(torch.arange(T, device=device))
        x = token_emb + pos_enc
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLM().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
        # print(loss.item())

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

a = 5
