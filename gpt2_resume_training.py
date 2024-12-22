from gpt2 import *
import torch
import time
import argparse
import nanoid
import os

device = 'cuda'

model_t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=8)
parser.add_argument('--T', type=int, default=1024)
parser.add_argument('--max_steps', type=int, default=1_220_704)
args = parser.parse_args()

B = args.B  # Use the passed argument for the batch size
T = args.T  # Use the passed argument for the batch size
max_steps = args.max_steps  # Use the passed argument for the batch size

# train_loader = DataloaderLite(B=4, T=1024, data="train")  # optimal value for RTX 4070ti (12GM VRAM) > 6, 1024 > 4, 1024
# val_loader = DataloaderLite(B=4, T=1024, data="val")
data_dir = "edu_fineweb10B"
train_loader = DataLoaderNumpy(B=B, T=T, data="train", data_dir=data_dir)  # optimal value for RTX 4070ti (12GM VRAM) > 6, 1024 > 4, 1024
val_loader = DataLoaderNumpy(B=B, T=T, data="val", data_dir=data_dir)

model = GPT(GPTConfig())
model.to(device)

# model.load_state_dict(torch.load("C:\Workspace-ML\decoder-only-transformer\log\model_00099.pt", weights_only=True))
# model.eval()
optimizer = model.config_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)


def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    step = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        step = checkpoint['step']
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint['model'].items()}
            model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = 0.1*checkpoint['lr']
        # lr = 0.60006e-05
        print("=> loaded checkpoint '{}' (step {})"
                  .format(filename, checkpoint['step']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, step, lr


model, optimizer, step_done, lr = load_checkpoint(model, optimizer, filename="/home/nand-ml/decoder-only-transformer/log_1024_BlockSize_1220704_Steps_8_B_1024_T_tG2nwc/model_1000000.pt")
model = torch.compile(model)

# max_lr = optimizer.defaults['lr']
# min_lr = max_lr * 0.1
# warmup_steps = 10
max_steps = step_done + 500_000

val_step_unit = 1000
model_ckpt_step_unit = [100_000]

# def get_lr(it):
#     # if it < warmup_steps:
#     #     return max_lr * (it+1) / warmup_steps

#     if it > max_steps:
#         return min_lr

#     decay_ratio = (it-step_done)/max_steps
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # starts at 1 and goes to 0
#     return min_lr + coeff * (max_lr - min_lr)


# create the log directory we will write checkpoints to and log to
log_dir = "log4" #todo
log_dir = f"log_{nanoid.generate(size=6)}"


os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(step_done, max_steps):
    t1 = time.time()

    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % val_step_unit == 0 or step in model_ckpt_step_unit or last_step:
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
        if step > 0 and (step in model_ckpt_step_unit or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(log_dir, f"model_{step}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'optimizer': optimizer.state_dict(),
                'step': step,
                'val_loss': val_loss_accum.item(),
                'lr': lr
            }
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training
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

    # lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # to update the params and to decrease the loss
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    dt = (t2-t1)
    tokens_per_sec = (train_loader.B * train_loader.T)/dt
    # if i % 10 == 0:
    print(f"Step:{step}| Loss: {loss.item():.4f} | Lr: {lr:0.2e} | norm: {norm:.2f} | dt: {dt*1000:.2f}ms | t/s: {tokens_per_sec:.2f}")  # loss.item() will convert the tensor to float, so it lives in CPU.
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss.item():.6f}\n")
        f.write(f"{step} lr {lr:0.4e}\n")


print("total_epochs: ", no_epochs)

# at starting point (first loss value), when probability of each vocab in our vocabulary is equal due to random initialization,
# expected loss ~ -ln(1/50257) ~10.8
# print(loss)

model_save_path = os.path.join(log_dir, f"gpt2_124M_BT_4_1024_{max_steps}.pt")
torch.save(model.state_dict(), model_save_path)


model_t2 = time.time()
print(f"Total time taken: {(model_t2-model_t0)/60}min")

import sys; sys.exit(0)