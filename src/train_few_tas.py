# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-train.py
@time:2023/4/8 下午2:12
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

from model import BasicDataset,define_model,get_all_embeddings
from utils import query_moa_function

def get_data(kind,fold,data_info_dir,data_tas_dir,few):
    # load lable
    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)
    sig2drugmoa_dict = np.load(sig2drugmoa_file)
    drug2moa_dict = np.load(drug2moa_file)

    # load train test data
    train_file = '{}/few_data/train_fold_{}_few_{}.h5'.format(data_tas_dir,fold,few)
    test_file = '{}/few_data/test_fold_{}_few.h5'.format(data_tas_dir,fold)
    train_df = pd.read_hdf(train_file)
    test_df = pd.read_hdf(test_file)

    # log
    print('#'*50)
    print('#'*10,kind,few,train_df.shape,test_df.shape)

    # make label
    all_signatures = list(train_df.index) + list(test_df.index)
    moas_label = [sig2drugmoa_dict[i][1] for i in all_signatures]
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(moas_label)
    label_dict = dict(zip(range(len(labelencoder.classes_)), labelencoder.classes_))
    labels_train = labels[:train_df.shape[0]]
    labels_test = labels[train_df.shape[0]:]
    train_dataset = BasicDataset(train_df.values, labels_train)
    test_dataset = BasicDataset(test_df.values, labels_test)
    return train_dataset, test_dataset,label_dict,drug2moa_dict


def train_all(data_dir,fold,batch_size,num_epoch,device,logname,embedding_size,few):
    data_info_dir = '{}/data/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/data/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/data/03_Single_Cell/'.format(data_dir)
    data_tas_dir = '{}/data/06_TAS/'.format(data_dir)

    train_dataset,test_dataset,label_dict,drug2moa_dict = get_data(
        kind='few',
        fold=fold,
        data_info_dir = data_info_dir,
        data_tas_dir = data_tas_dir,
        few=few)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    model, loss_func, mining_func, optimizer = define_model(device,embedding_size)
    test_log_df = pd.DataFrame()
    test_epoch = range(1, num_epoch)
    test_num = []
    test_find = []
    for epoch in range(1, num_epoch):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(data)
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                print(
                    "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                        epoch, batch_idx, loss, mining_func.num_triplets))
        train_embeddings, train_labels = get_all_embeddings(train_dataset, model)
        test_embeddings, test_labels = get_all_embeddings(test_dataset, model)

        train_labels = train_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
        sample_num, find_num = query_moa_function(test_embeddings, train_embeddings, test_labels, train_labels)
        test_num.append(sample_num)
        test_find.append(find_num)
    test_log_df['epoch'] = test_epoch
    test_log_df['sample_num'] = test_num
    test_log_df['find'] = test_find
    test_log_df.to_csv('{}/src/result_few_tas/few_fold{}_{}_{}.csv'.format(data_dir,fold, logname,few))



if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/'
    num_epoch = 300
    batch_size = 256
    logname = '20230507'
    device = torch.device("cuda")

    for fold in [0,1,2]:
        for few in ['ori',1000,500,200,100]:
            train_all(data_dir=data_dir,
                           fold=fold,
                           batch_size=batch_size,
                           num_epoch=num_epoch,
                           device=device,
                           logname=logname,
                           embedding_size=256,
                           few = few)

