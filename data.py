'''
Author: Chengyu Zheng
Date: 2024-09-06 22:34:36
Description: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import re
import os
# import sklearn

from sklearn.model_selection import train_test_split
def read_imdb(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    def read_data(data_dir):
        data, labels = [], []
        for label in ['pos', 'neg']:
            folder_name = os.path.join(data_dir, label)
            for file in os.listdir(folder_name):
                with open(os.path.join(folder_name, file), 'rb') as f:
                    review = f.read().decode('utf-8').replace('\n','')
                    data.append(review)
                    labels.append(1 if label == 'pos' else 0)
        return data, labels
    train_data, train_labels = read_data(train_dir)
    test_data, test_labels = read_data(test_dir)
    return train_data, train_labels, test_data, test_labels

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')

# 定义数据集类
class IMDBDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.y[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'token_type_ids': inputs['token_type_ids'][0],
            'label': torch.tensor(label, dtype=torch.long)
        }

def dataloader(data_dir):
    train_data, train_labels, test_data, test_labels = read_imdb(data_dir)

    # 划分训练集和测试集
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    train_dataset = IMDBDataset(X_train, y_train, tokenizer, max_len=512)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = IMDBDataset(X_val, y_val, tokenizer, max_len=512)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    test_dataset = IMDBDataset(test_data, test_labels, tokenizer, max_len=512)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader