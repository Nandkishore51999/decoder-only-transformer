import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import text_preprocessing
import tiktoken

assert torch.cuda.is_available(), 'cuda not found'
device = 'cuda'


class DataloaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')

        # BPE tokeniser has compression ratio of 3:1. given 1000 characters > approx 300 tokens
        # ascii chars each takes 1 bit memory, so file size equals to no of chars in it.
        text = text_preprocessing.read_text_files("C:\\Workspace-ML\\text_data")

        # currently here all the tokens are in CPU, we don't want to waste GPU memory for this
        self.tokens = enc.encode(text)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T) } batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[self.current_position: self.current_position + B * T + 1]).to(device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T+1

        # when we run out of data, then reset and starts new epoch
        if self.current_position + B*T+1 > len(self.tokens):
            self.current_position = 0
        return x, y


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"{config.n_embd % config.n_head}"

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# A clean pathway is desirable from an optimization perspective
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    # attn: a communication operation where all token (i.e. 1024 communicate to each other and exchange information > a aggregation function > a weighted sum function
    # mlp: on individual tokens, no communication between tokens
    # atten is a reduce fn and mlp is map fn

    # first they communicate and then think individually about the information that they gathered
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # first norm then attention
        x = x + self.mlp(self.ln_2(x))  # first norm then mlp aka ffn (feed forward network)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # all of these layers and modules have random initializer inside it by default.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # additional layer norm (not in transformer paper... but in gpt2 paper) Quoted:
            # an additional layer normalization was added after the final self-attention block (after 12th block).
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        # final classifier
        # projects final 768 embds to no of vocab 50k
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  # try with and without this at training (30% reduction in model pramas count)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B,T)
        # each row max length of block size, each element in row reprents a token
        # we have B independent sequences stacked up in batch so that this is efficient
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # shape (B, T, n_embd)  # pos_emb is going to be identical for all sequences
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss


torch.cuda.manual_seed(42)
model = GPT(GPTConfig())
# this is a good practice to put model to eval when we are not going to train it.
# usually model have different behaviour at training and evaluation time i.e. Dropout, Batch-Norm
# In our case, it does not have any difference
model.eval()
model.to(device)



num_return_sequnces = 5
max_length = 30


train_loader = DataloaderLite(B=4, T=32)

# lr = 3e-4 is a good default value for early debugging stages
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(5000):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)  # here we are sending the tokens from CPU to GPU
    # always remember to start with zero gradient
    optimizer.zero_grad()
    logits, loss = model(x, y)

    # add to gradients += operation
    loss.backward()

    # to update the params and to decrease the loss
    optimizer.step()
    print(f"step: {i}\tLoss: {loss.item()}")  # loss.item() will convert the tensor to float, so it lives in CPU.


# at starting point (first loss value), when probability of each vocab in our vocabulary is equal due to random initialization,
# expected loss ~ -ln(1/50257) ~10.8
print(loss)

import sys; sys.exit(0)

# tokens = enc.encode("Hello, who are you?")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequnces, 1)  # # (B, length on tokens after encoding the text)
# x = tokens.to(device)
# print(x)


while x.size(1) < max_length:
    with torch.no_grad():  # telling pytorch that we will not be calling backward on any of below steps so it doesn't have to cache all the intermediate tensors
        logits = model(x) # shpae (B, T, vocab_size)
        print(logits)
        logits = logits[:, -1, :]  # takes only last logits which are the prediction # shpae (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # doing topK sampling, shape (B, 50), (B, 50), Helps in keeping the model on track (HOW???)

        print(topk_probs)
        ix = torch.multinomial(topk_probs, 1)  # select a token from top-k probabilities (B, 1)
        xcol = torch.gather(topk_indices, -1 , ix)  # (B, 1)
        x = torch.cat((x, xcol), dim=1)




for i in range(num_return_sequnces):
    tokens = x[i, :max_length].tolist()
    decode = enc.decode(tokens)
    print("> ", decode)




