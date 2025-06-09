from data import load_data
import config

import torch
import torch.nn as nn
from modules import ViTForImageClassification

import torch.optim as optim
import config
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch.utils.data import ConcatDataset, DataLoader

dataset = "cifar10"

train_dataloader, valid_dataloader, test_dataloader = load_data(
    dataset, splits=["train", "dev", "test"]
)

base_lr = config.base_lr[dataset]
# we will not use the following setting, see config.py
# total_steps = config.total_steps[dataset]

num_epochs = config.num_epochs
total_steps = num_epochs * len(train_dataloader)

device = torch.device("cuda")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.classifier = nn.Linear(model.classifier.in_features, 10, bias=False)
nn.init.zeros_(model.classifier.weight)
model.to(device)

criterion = nn.CrossEntropyLoss()
initial_sd = {k: v.cpu() for k, v in model.state_dict().items()}


def evaluate(model, dataloader):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating..."):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    return accuracy

def train(model, lr, total_steps, optimizer, scheduler):
    model.train()
    current_step = 0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(total=total_steps, desc=f"Training for lr={lr}")
    while current_step < total_steps:
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            if current_step > total_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / config.accumulate_grad_batches
            loss.backward()

            if (
                batch_idx + 1
            ) % config.accumulate_grad_batches == 0 or batch_idx == total_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            current_step += 1
            pbar.update(1)

    pbar.close()


def configure_optimizer(lr, total_steps):
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return optimizer, scheduler

def grid_search():
    best_lr = 0
    best_acc = 0

    search_ratio = 0.05
    search_steps = int(total_steps * search_ratio)

    for lr in base_lr:
        model.load_state_dict({k: v.to(device) for k, v in initial_sd.items()})
        train(model, lr, search_steps, *configure_optimizer(lr, search_steps))
        acc = evaluate(model, valid_dataloader)
        print(f"Learning rate: {lr}, Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_lr = lr

    print(f"Best learning rate: {best_lr}")

if __name__=='__main__':
    best_lr = 0.03
    train_dataloader = DataLoader(
        ConcatDataset([train_dataloader.dataset, valid_dataloader.dataset]),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.load_state_dict({k: v.to(device) for k, v in initial_sd.items()})
    train(model, best_lr, total_steps, *configure_optimizer(best_lr, total_steps))
    evaluate(model, test_dataloader)