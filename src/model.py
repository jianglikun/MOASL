# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-model.py
@time:2023/4/8 下午2:09
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners, reducers, testers

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = labels
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.data)

class DrugDataset(torch.utils.data.Dataset):
    def __init__(self, data,drugs, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = labels
        self.drugs = drugs
    def __getitem__(self, index):
        return self.data[index],(self.drugs[index], self.labels[index])
    def __len__(self):
        return len(self.data)

class represent_model(nn.Module):
    def __init__(self, hiddensize,gene_num):
        super(represent_model, self).__init__()
        self.hiddenSize = hiddensize
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512,hiddensize),
            nn.Tanh()
        )
    def forward(self, x):
        h0 = self.encoder[0](x)
        h1 = self.encoder[1](h0)
        h2 = self.encoder[2](h1)
        h3 = self.encoder[3](h2)
        h4 = self.encoder[4](h3)
        h5 = self.encoder[5](h4)
        return  h5

class relu_model(nn.Module):
    def __init__(self, hiddensize,gene_num):
        super(relu_model, self).__init__()
        self.hiddenSize = hiddensize
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,hiddensize),
            nn.ReLU()
        )
    def forward(self, x):
        h0 = self.encoder[0](x)
        h1 = F.dropout(self.encoder[1](h0),p=0.1)
        h2 = self.encoder[2](h1)
        h3 = F.dropout(self.encoder[3](h2),p=0.1)
        h4 = self.encoder[4](h3)
        h5 = self.encoder[5](h4)
        return h5


def define_model(device,embedding_size):
    model = represent_model(hiddensize=embedding_size, gene_num=12328).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.01, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.01, distance=distance, type_of_triplets="semihard")

    return model,loss_func,mining_func,optimizer


def define_model_margin(device,embedding_size,margin):
    model = represent_model(hiddensize=embedding_size, gene_num=12328).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets="semihard")

    return model,loss_func,mining_func,optimizer

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

