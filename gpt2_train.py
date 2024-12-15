from gpt2 import *
import time
import nanoid

total_time_t0 = time.time()

data_dir = "edu_fineweb10B"
B = 8
T = 1024
GPTConfig.block_size = T
train_loader = DataLoaderNumpy(B=B, T=T, data="train", data_dir=data_dir)  # optimal value for RTX 4070ti (12GM VRAM) > 6, 1024 > 4, 1024
val_loader = DataLoaderNumpy(B=B, T=T, data="val", data_dir=data_dir)

print(f"Data Load Time: {(time.time() - total_time_t0)/60} Min")

max_lr = 6e-5
min_lr = max_lr * 0.1
max_steps = 100
warmup_steps = int(0.03*max_steps)

val_step_unit = int(max_steps/1)
model_ckpt_step_unit = int(max_steps/1)

# create the log directory we will write checkpoints to and log to
log_dir = f"log_{nanoid.generate(size=6)}_{GPTConfig.block_size}_BlockSize_{max_steps}_Steps_{B}_B_{T}_T"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass


assert torch.cuda.is_available(), 'cuda not found'
device = 'cuda'
torch.cuda.empty_cache()
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('medium')
USE_FLASH_ATTENTION=1
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
# torch.backends.cudnn.benchmark = True

model = GPT(GPTConfig())
model.to(device)

# Take some time at starting but reduces the training time so much
# Basically, first it scan entire cod and check what are the operations happening and then optimize those.
# aka Kernel fusion
# as of Dec 2024, does not support on Windows
# model = torch.compile(model)


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# lr = 3e-4 is a good default value for early debugging stages
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.config_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t1 = time.time()

    last_step = (step == max_steps - 1)

    if step % val_step_unit == 0 or last_step:
        model.eval()
        # val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y, _ = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        if step > 0 and (step % model_ckpt_step_unit == 0 or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(log_dir, f"model_{step}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'optimizer': optimizer.state_dict(),
                'step': step,
                'val_loss': val_loss_accum.item(),
                'lr': get_lr(step)
            }
            torch.save(checkpoint, checkpoint_path)

    x, y, no_epochs = train_loader.next_batch()
    x, y = x.to(device), y.to(device)  # here we are sending the tokens from CPU to GPU
    # always remember to start with zero gradient
    optimizer.zero_grad()

    # mix precision
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    # add to gradients += operation
    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # to update the params and to decrease the loss
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    dt = (t2-t1)*1000
    tokens_per_sec = (train_loader.B * train_loader.T)/dt

    print(f"Step: {step} | Loss: {loss.item()} | Lr: {lr:0.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")  # loss.item() will convert the tensor to float, so it lives in CPU.
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss.item():.6f}\n")
        f.write(f"{step} lr {lr:0.4e}\n")

print("total_epochs: ", no_epochs)

# at starting point (first loss value), when probability of each vocab in our vocabulary is equal due to random initialization,
# expected loss ~ -ln(1/50257) ~10.8
print(loss)

model_save_path = os.path.join(log_dir, f"gpt2_124M_Final.pt")
torch.save(model.state_dict(), model_save_path)


total_time_t1 = time.time()
print(f"Total time taken: {(total_time_t1 - total_time_t0) / 60} Min")

import sys; sys.exit(0)

# this is a good practice to put model to eval when we are not going to train it.
# usually model have different behaviour at training and evaluation time i.e. Dropout, Batch-Norm
# model.eval()



