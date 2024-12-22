import inspect
import math
import os
import traceback
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np

assert torch.cuda.is_available(), 'cuda not found'
device = 'cuda'
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('medium')
USE_FLASH_ATTENTION=1


def read_text_files(folder_path):
    combined_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    combined_text += f.read()
            except:
                traceback.print_exc()
    return combined_text


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class DataloaderLite:
    def __init__(self, B, T, data, data_dir):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')

        # BPE tokeniser has compression ratio of 3~4:1. given 1000 characters > approx 300 tokens
        # ascii chars each takes 1 bit memory, so file size equals to no of chars in it.
        if data == 'train':
            text = read_text_files(data_dir + "/train")
        elif data == 'val':
            text = read_text_files(data_dir + "/val")

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
        # lets say input token count > 1Million
        # for B,T = 4, 1024 > to take entire dataset in single loop > 1M/(1024*4) = 244 batches
        # We ran 2500 training steps > how many times our entire data was repeated for training = 2500/244
        if self.current_position + B*T+1 > len(self.tokens):
            self.current_position = 0
            self.no_epochs += 1
        return x, y, self.no_epochs


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderNumpy:
    def __init__(self, B, T, data, data_dir):
        self.B = B
        self.T = T
        assert data in {'train', 'val'}

        # get the shard filenames
        data_root = data_dir
        shards = os.listdir(data_root)
        shards = [s for s in shards if data in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for data {data}"
        print(f"found {len(shards)} shards for data {data}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            print(f"Next shard reached! {self.current_shard}")
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y, 9999


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

    def config_optimizer(self, weight_decay, learning_rate, device):
        # start with all the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

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
