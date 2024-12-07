import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import text_preprocessing
import tiktoken

assert torch.cuda.is_available(), 'cuda not found'
device = 'cuda'
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('medium')
USE_FLASH_ATTENTION=1


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
        self.no_epochs = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[self.current_position: self.current_position + B * T + 1]).to(device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T+1

        # when we run out of data, then reset and starts new epoch
        if self.current_position + B*T+1 > len(self.tokens):
            self.current_position = 0
            self.no_epochs += 1
        return x, y, self.no_epochs


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

        # Not helpful for me since Triton is not compatible with Windows and cant install flash-attention library
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
