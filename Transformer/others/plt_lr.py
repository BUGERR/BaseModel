import torch
import config

import torch.optim as optim
import torch.nn as nn
from utils import WarmupScheduler
import matplotlib.pyplot as plt

if __name__ == '__main__':

    optimizer = optim.Adam(
        [nn.Parameter(torch.empty(4, 4))], 
        betas=config.betas, 
        eps=config.adam_eps, 
    )

    scheduler = WarmupScheduler(
        optimizer=optimizer, 
        hidden_size=config.d_model, 
        warmup_step=config.warmup_step, 
    )

    lrs = []
    iteration = 100000
    
    for _ in range(iteration):
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