# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-GPAR.py
@time:2023/3/31 下午4:02
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
from sklearn.preprocessing import LabelEncoder



class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class GPAR_DNN(nn.Module):
    def __init__(self, gene_num,out_dim):
        super(GPAR_DNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 512),
            nn.Linear(512, 256),
            nn.Linear(256,out_dim),
            )
    def forward(self, x):
        h1 = F.dropout(F.relu(self.encoder[0](x)),p=0.1)
        h2 = F.dropout(F.relu(self.encoder[1](h1)),p=0.1)
        h3 = self.encoder[2](h2)
        return h3

def train_test_data(train_file,test_file,sig2drugmoa_file,batch_size):
    sig2drugmoa_dict = np.load(sig2drugmoa_file)

    train_df = pd.read_hdf(train_file)
    test_df = pd.read_hdf(test_file)
    print(train_df.shape,test_df.shape)

    train_moas = [sig2drugmoa_dict[i][1] for i in train_df.index]
    test_moas = [sig2drugmoa_dict[i][1] for i in test_df.index]
    print(train_df.shape,test_df.shape)

    all_label = train_moas + test_moas
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(all_label)
    label_dict = dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))

    train_label = labels[:len(train_moas)]
    test_label = labels[len(train_moas):]
    train_dataset = BasicDataset(train_df.values,train_label)
    test_dataset = BasicDataset(test_df.values,test_label)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=test_df.shape[0]*2,shuffle=False)
    return train_dataloader,test_dataloader,label_dict,train_df.shape[0],test_df.shape[0]


def train(train_dataloader,test_dataloader,label_dict,epoch_num,device,save_file):
    save_df = pd.DataFrame()
    save_train_num = [len((train_dataloader.dataset))]*epoch_num
    save_train_right = []
    save_test_num = [len(test_dataloader.dataset)]*epoch_num
    save_find_num = []

    model = GPAR_DNN(gene_num=12328,out_dim=len(label_dict)).to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimazer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_find = 0
    test_find = 0
    for epoch in range(epoch_num):
            model.train()
            size = len(train_dataloader.dataset)
            num_batches = len(train_dataloader)
            model.eval()
            train_loss, correct = 0, 0
            for train_x,train_y in train_dataloader:
                train_x = train_x.to(device)
                train_y = train_y.to(device)
                train_pred = model(train_x)
                loss = loss_fn(train_pred,train_y)
                correct += (train_pred.argmax(1) == train_y).type(torch.float).sum().item()
                optimazer.zero_grad()
                loss.backward()
                optimazer.step()
            save_train_right.append(correct)
            train_loss /= num_batches
            acc = correct / size
            if correct > train_find:
                train_find = correct
            print(f"Epoch {epoch} Train: find {correct} Accuracy: {(100 * acc):>0.1f}%, Avg loss: {train_loss:>8f}")

            model.eval()
            with torch.no_grad():
                size = len(test_dataloader.dataset)
                num_batches = len(test_dataloader)
                model.eval()
                test_loss, correct = 0, 0
                for test_x,test_y in test_dataloader:
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    pred_test = model(test_x)
                    test_loss += loss_fn(pred_test,test_y).item()
                    correct += (pred_test.argmax(1) == test_y).type(torch.float).sum().item()

                save_find_num.append(correct)
                test_loss /= num_batches
                acc = correct/size
                if correct > test_find:
                    test_find = correct
                print(f"Epoch {epoch} Test: find {correct} Accuracy: {(100 * acc):>0.1f}%, Avg loss: {test_loss:>8f}")
    save_df['Train'] = save_train_num
    save_df['Train_find'] = save_train_right
    save_df['Test'] = save_test_num
    save_df['Test_find'] = save_find_num
    save_df.to_csv(save_file)

    return train_find,test_find

if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)
    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)
    batch_size=256
    epoch_num = 500
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda:1" if train_on_gpu else "cpu")

    save_file = 'result/GPAR.csv'
    f = open(save_file, 'w')
    for fold in [0, 1, 2]:
        train_file = '{}/Train_fold_{}.h5'.format(data_all_dir, fold)
        test_file = '{}/Test_fold_{}.h5'.format(data_all_dir, fold)
        print('#' * 20, 'ALL', '#' * 20)
        cell = 'ALL'
        train_dataloader, test_dateloader, label_dict,train_num,test_num = train_test_data(train_file,test_file,sig2drugmoa_file,batch_size)
        save_file = 'result/GPAR_ALL_fold_{}.csv'.format(fold)
        train_find,test_find = train(train_dataloader,test_dateloader,label_dict,epoch_num,device,save_file)

        f.write('DNN,{},{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, train_find,test_find, round(test_find/test_num,4)))

    cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20',
                      'HCC515', 'HEPG2', 'HA1E', 'NPC', 'VCAP']
    for cell in cell_lines:
        for fold in [0, 1, 2]:
            print('#' * 20, cell, '#' * 20)
            train_file = '{}/{}/Train_fold_{}.h5'.format(data_cell_dir, cell, fold)
            test_file = '{}/{}/Test_fold_{}.h5'.format(data_cell_dir, cell, fold)
            train_dataloader, test_dateloader, label_dict,train_num,test_num = train_test_data(train_file, test_file, sig2drugmoa_file,
                                                                            batch_size)
            save_file = 'result/GPAR_{}_fold_{}.csv'.format(cell,fold)
            train_find,test_find = train(train_dataloader, test_dateloader, label_dict, epoch_num, device, save_file)
            f.write('DNN,{},{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, train_find, test_find,
                                                        round(test_find / test_num, 4)))
