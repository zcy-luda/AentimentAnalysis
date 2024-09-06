'''
Author: Chengyu Zheng
Date: 2024-09-06
Description: 
'''
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import BERTClassifier
from data import dataloader

from misc import setup_seed
setup_seed(0)

device = torch.device('cuda:0')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=1, type=int, help='')
    cfg = parser.parse_args()

    # load data
    data_dir = '../data/aclImdb'
    train_dataloader, val_dataloader, test_dataloader = dataloader(data_dir)

    # model
    model = BERTClassifier(num_classes=2).to(device)

    # optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 1.0)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epoch):
        train_loss = train(model, train_dataloader, optimizer, criterion)
        val_loss, val_acc = test(model, val_dataloader, criterion)
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')

    test_loss, test_acc = test(model, test_dataloader, criterion)
    print(f'Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}')

# 定义训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    num_iter = int(len(dataloader))
    loader_iter = dataloader.__iter__()
    train_iterator = tqdm(range(num_iter))
    # for batch in dataloader:
    for iter in train_iterator:
        batch = loader_iter.next()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print('stepped foward')

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 定义测试函数
def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        num_iter = int(len(dataloader))
        loader_iter = dataloader.__iter__()
        train_iterator = tqdm(range(num_iter))
        # for batch in dataloader:
        for iter in train_iterator:
            batch = loader_iter.next()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)


if __name__ == '__main__':
    main()
