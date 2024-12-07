import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import text_preprocessing
import tiktoken
from gpt2 import *
import time


assert torch.cuda.is_available(), 'cuda not found'
device = 'cuda'
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('medium')


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


model = GPT(GPTConfig())
# this is a good practice to put model to eval when we are not going to train it.
# usually model have different behaviour at training and evaluation time i.e. Dropout, Batch-Norm
# In our case, it does not have any difference
model.eval()
model.to(device)



num_return_sequnces = 5
max_length = 30


train_loader = DataloaderLite(B=4, T=1024)  # optimal value for RTX 4070ti (12GM VRAM) > 6, 1024 > 4, 1024

# lr = 3e-4 is a good default value for early debugging stages
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
iterations = 50
for i in range(iterations):
    t1 = time.time()
    x, y, no_epochs = train_loader.next_batch()
    x, y = x.to(device), y.to(device)  # here we are sending the tokens from CPU to GPU
    # always remember to start with zero gradient
    optimizer.zero_grad()

    # mix precision
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    # add to gradients += operation
    loss.backward()

    # to update the params and to decrease the loss
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    dt = (t2-t1)*1000
    tokens_per_sec = (train_loader.B * train_loader.T)/dt
    # if i % 10 == 0:
    print(f"step: {i} Loss: {loss.item()}, time: {dt:.2f}, tokens_per_sec: {tokens_per_sec:.2f}")  # loss.item() will convert the tensor to float, so it lives in CPU.
    if i in [10000, 25000]:
        model_save_path = f"gpt2_124M_{i}.pth"
        torch.save(model.state_dict(), model_save_path)

print("total_epochs: ", no_epochs)

# at starting point (first loss value), when probability of each vocab in our vocabulary is equal due to random initialization,
# expected loss ~ -ln(1/50257) ~10.8
print(loss)
model_save_path = f"gpt2_124M_{iterations}.pth"
torch.save(model.state_dict(), model_save_path)

import sys; sys.exit(0)

tokens = enc.encode("Hello, who are you?")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequnces, 1)  # # (B, length on tokens after encoding the text)
x = tokens.to(device)
print(x)


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




