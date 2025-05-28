import torch.optim as optim
import torch.nn as nn
from utils import WarmupScheduler
import config
import torch
import matplotlib.pyplot as plt

optimizer = optim.Adam(
    [nn.Parameter(torch.empty(4, 4))],
    betas=config.betas,
    eps=config.adam_eps,
)

scheduler = WarmupScheduler(optimizer, config.d_model, 4000)

lrs = []
for step in range(100000 * config.accumulate_grad_batches):
    if (step + 1) % config.accumulate_grad_batches == 0:
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

plt.figure(figsize=(8, 3))
plt.plot(lrs)
plt.ylabel("Learning rate factor")
plt.xlabel("Iterations (in batches)")
plt.grid(True)
plt.title("Inverse Square Root Warm-up Learning Rate Scheduler")
plt.show()