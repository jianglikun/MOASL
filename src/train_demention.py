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

from model import BasicDataset,DrugDataset,define_model,get_all_embeddings
from utils import query_moa_high,query_drug_high

def get_data(kind,fold,data_info_dir,data_all_dir,data_cell_dir,cell):
    # load lable
    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)
    sig2drugmoa_dict = np.load(sig2drugmoa_file)
    drug2moa_dict = np.load(drug2moa_file)

    # load train test data
    if kind == 'all':
        train_file = '{}/Train_fold_{}.h5'.format(data_all_dir,fold)
        test_file = '{}/Test_fold_{}.h5'.format(data_all_dir,fold)
        train_df = pd.read_hdf(train_file)
        train_df = train_df.head(100)
        test_df = pd.read_hdf(test_file)
        test_df = test_df.head(100)
    elif kind == 'single':
        train_file = '{}/{}/Train_fold_{}.h5'.format(data_cell_dir,cell,fold)
        test_file = '{}/{}/Test_fold_{}.h5'.format(data_cell_dir,cell,fold)
        train_df = pd.read_hdf(train_file)
        test_df = pd.read_hdf(test_file)

    # log
    print('#'*50)
    print(kind,cell,train_df.shape,test_df.shape)

    # make label
    all_signatures = list(train_df.index) + list(test_df.index)
    moas_label = [sig2drugmoa_dict[i][1] for i in all_signatures]
    drug_label_str = [sig2drugmoa_dict[i][0] for i in all_signatures]
    labelencoder = LabelEncoder()
    drug_labels = labelencoder.fit_transform(drug_label_str)
    drug_label_dict = dict(zip(range(len(labelencoder.classes_)), labelencoder.classes_))

    labelencoder = LabelEncoder()
    moa_labels = labelencoder.fit_transform(moas_label)
    moa_label_dict = dict(zip(labelencoder.classes_,range(len(labelencoder.classes_))))

    # labels_train_moa = labels[:train_df.shape[0]]
    # labels_train_drug = drug_label[:train_df.shape[0]]
    # labels_test_moa = labels[train_df.shape[0]:]
    # labels_test_drug = drug_label[train_df.shape[0]:]

    # labels_train_moa = labels[:train_df.shape[0]]
    labels_train_drug = drug_labels[:train_df.shape[0]]
    # labels_test_moa = labels[train_df.shape[0]:]
    labels_test_drug = drug_labels[train_df.shape[0]:]

    train_dataset = BasicDataset(train_df.values, labels_train_drug)
    test_dataset = BasicDataset(test_df.values, labels_test_drug)
    return train_dataset, test_dataset,drug_label_dict,moa_label_dict,drug2moa_dict


def train_single_cell(data_dir,fold,batch_size,num_epoch,device,cell,logname):
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)

    train_dataset, test_dataset, drug_label_dict,moa_label_dict,drug2moa_dict = get_data(
        kind='single',
        fold=fold,
        data_info_dir=data_info_dir,
        data_all_dir=data_all_dir,
        data_cell_dir=data_cell_dir,
        cell=cell)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    model, loss_func, mining_func, optimizer = define_model(device,embedding_size=256)
    test_log_df = pd.DataFrame()
    test_epoch = range(1,num_epoch)
    test_num = []
    test_find = []
    is_high = 0
    for epoch in range(1, num_epoch):
        model.train()
        for batch_idx, (data,drugs) in enumerate(train_loader):
            labels = [moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in drugs.cpu().tolist()]
            data, labels = data.to(device), torch.tensor(labels).to(device)
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
        train_embeddings,train_labels = get_all_embeddings(train_dataset, model)
        test_embeddings, test_labels = get_all_embeddings(test_dataset, model)

        print(train_embeddings.shape,train_labels.shape)

        train_moas = train_labels.squeeze(1)
        test_moas = test_labels.squeeze(1)

        sample_num,find_num,save_train_df,save_test_df = query_drug_high(test_embeddings, train_embeddings, test_moas,
                                                  train_moas,drug_label_dict,drug2moa_dict,moa_label_dict)
        test_num.append(sample_num)
        test_find.append(find_num)

        # if find_num > is_high:
        #     is_high = find_num
        #     train_save_file = 'result_demention/train_embedding.h5'
        #     test_save_file = 'result_demention/test_embedding.h5'
        #     save_train_df.to_hdf(train_save_file,key = 'dat')
        #     save_test_df.to_hdf(test_save_file,key = 'dat')


    test_log_df['epoch'] = test_epoch
    test_log_df['sample_num'] = test_num
    test_log_df['find'] = test_find
    test_log_df.to_csv('result_demention/{}_fold{}_{}.csv'.format(cell,fold,logname))


if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    num_epoch = 600
    batch_size = 64
    logname = '20230419_Demention'
    device = torch.device("cuda")
    cell = 'MCF7'

    for fold in [2]:
        train_single_cell(data_dir=data_dir,
                              fold=fold,
                              batch_size=batch_size,
                              num_epoch=num_epoch,
                              device=device,
                              cell=cell,
                              logname=logname)