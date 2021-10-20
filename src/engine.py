import torch
from torch._C import dtype
import torch.nn as nn
import config
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

train_fold = pd.read_csv(
    f'{config.path}/{config.csv_name}.csv'
)


def plot_loss(losses):
    plt.plot(losses)
    plt.show()



def train(data_loader, model, optimizer, device=config.device):
    accuracies, losses = [], []
    labels = []
    preds = []

    model.train()

    loop = tqdm(data_loader)
    for idx, (texts, target) in enumerate(loop):
        texts, target = texts.to(device), target.to(device)

        optimizer.zero_grad()
        predictions = model(texts)

        loss = nn.CrossEntropyLoss()(
            predictions.squeeze(),
            target
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(loss.item())
        probs = torch.softmax(predictions, dim=1)
        winners = probs.argmax(dim=1)
        corrects = (winners == target)
        accuracy = corrects.sum().float() / float(target.size(0))
        accuracies.append(accuracy)
        labels += torch.flatten(target).cpu()
        preds += torch.flatten(winners).cpu()
        loop.set_postfix(loss=loss.item())
    avg_train_loss = sum(losses) / len(losses)
    avg_train_acc = sum(accuracies) / len(accuracies)
    report = metrics.classification_report(
        torch.tensor(labels).numpy(),
        torch.tensor(preds).numpy()
    )
    print(report)
    return avg_train_acc, avg_train_loss, losses


def evaluate(data_loader, model, device=config.device):
    preds = []
    labels = []
    test_accuracies = []
    model.eval()
    with torch.no_grad():
        for idx, (texts, target) in enumerate(data_loader):
            texts = texts.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)

            predictions = model(texts)
            probs = torch.softmax(
                predictions, dim=1
            )
            winners = probs.argmax(
                dim=1
            )
            corrects = (winners == target)
            accuracy = corrects.sum().float() / float(target.size(0))
            test_accuracies.append(accuracy)
            labels += torch.flatten(target).cpu()
            preds += torch.flatten(winners).cpu()
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    return avg_test_acc

