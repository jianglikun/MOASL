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
from utils import query_moa_function,query_moa_cosine,query_drug_cosine

def get_data(moa,data_case_dir):
    # load train test data
    if moa == 'MOA1':
        # load lable
        sig2drugmoa_file = '{}/MOA1/sig2drugmoa.npz'.format(data_case_dir)
        drug2moa_file = '{}/MOA1/drug2moa.npz'.format(data_case_dir)
        drug2name_file = '{}/MOA1/drug2name.npz'.format(data_case_dir)
        drug2name_dict = np.load(drug2name_file)
        sig2drugmoa_dict = np.load(sig2drugmoa_file)
        drug2moa_dict = np.load(drug2moa_file)
        train_file = '{}/MOA1/Train.h5'.format(data_case_dir)
        test_file = '{}/MOA1/Test_all.h5'.format(data_case_dir)
        train_df = pd.read_hdf(train_file)
        test_df = pd.read_hdf(test_file)
        # test_df = test_df.head(1000)
        save_name = 'MOA1'
    elif moa == 'MOA2':
        # load lable
        sig2drugmoa_file = '{}/MOA2/sig2drugmoa.npz'.format(data_case_dir)
        drug2moa_file = '{}/MOA2/drug2moa.npz'.format(data_case_dir)
        drug2name_file = '{}/MOA2/drug2name.npz'.format(data_case_dir)
        drug2name_dict = np.load(drug2name_file)
        sig2drugmoa_dict = np.load(sig2drugmoa_file)
        drug2moa_dict = np.load(drug2moa_file)
        train_file = '{}/MOA2/Train.h5'.format(data_case_dir)
        test_file = '{}/MOA2/Test_all.h5'.format(data_case_dir)
        train_df = pd.read_hdf(train_file)
        test_df = pd.read_hdf(test_file)
        save_name = 'MOA2'
    elif moa == 'MOA3':
        # load lable
        sig2drugmoa_file = '{}/MOA3/sig2drugmoa.npz'.format(data_case_dir)
        drug2moa_file = '{}/MOA3/drug2moa.npz'.format(data_case_dir)
        drug2name_file = '{}/MOA3/drug2name.npz'.format(data_case_dir)
        drug2name_dict = np.load(drug2name_file)
        sig2drugmoa_dict = np.load(sig2drugmoa_file)
        drug2moa_dict = np.load(drug2moa_file)
        train_file = '{}/MOA3/Train.h5'.format(data_case_dir)
        test_file = '{}/MOA3/Test_all.h5'.format(data_case_dir)
        train_df = pd.read_hdf(train_file)
        test_df = pd.read_hdf(test_file)
        save_name = 'MOA3'

    # logq
    print('#'*50)
    print('#'*10,moa,train_df.shape,test_df.shape)

    all_signatures = list(train_df.index) + list(test_df.index)
    moas_label = [sig2drugmoa_dict[i][1] for i in all_signatures]
    drug_label_str = [sig2drugmoa_dict[i][0] for i in all_signatures]
    labelencoder = LabelEncoder()
    drug_labels = labelencoder.fit_transform(drug_label_str)
    drug_label_dict = dict(zip(range(len(labelencoder.classes_)), labelencoder.classes_))

    labelencoder = LabelEncoder()
    moa_labels = labelencoder.fit_transform(moas_label)
    moa_label_dict = dict(zip(labelencoder.classes_,range(len(labelencoder.classes_))))

    labels_train_drug = drug_labels[:train_df.shape[0]]
    labels_test_drug = drug_labels[train_df.shape[0]:]

    train_dataset = BasicDataset(train_df.values, labels_train_drug)
    test_dataset = BasicDataset(test_df.values, labels_test_drug)
    return train_dataset, test_dataset,drug_label_dict,moa_label_dict,drug2moa_dict,save_name,drug2name_dict


def train(moa,data_dir,batch_size,num_epoch,device,logname):
    data_info_dir = '{}/data/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/data/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/data/03_Single_Cell/'.format(data_dir)
    data_case_dir = '{}/data/04_Case_data/'.format(data_dir)



    train_dataset, test_dataset, drug_label_dict, moa_label_dict, drug2moa_dict,save_name,drug2name_dict = get_data(
        moa=moa,data_case_dir=data_case_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    moa_name = {"MOA1":"Dopamine receptor agonist",
                "MOA2":"Glucocorticoid receptor agonist",
                "MOA3":"HSP inhibitor"}
    moa = moa_name[moa]

    model, loss_func, mining_func, optimizer = define_model(device,embedding_size=256)

    is_high = 0
    for epoch in range(1, num_epoch):
        model.train()
        for batch_idx, (data,drugs) in enumerate(train_loader):
            labels = [moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in drugs.cpu().tolist()]
            # print(labels)
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
        train_embeddings, train_labels = get_all_embeddings(train_dataset, model)
        test_embeddings, test_labels = get_all_embeddings(test_dataset, model)

        train_labels = train_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)

        # sample_num, find_num = query_moa_cosine(test_embeddings, train_embeddings, test_labels, train_labels)
        sample_num, find_num,save_df = query_drug_cosine(test_embeddings, train_embeddings, test_labels,
                                                  train_labels,drug_label_dict,drug2moa_dict,moa_label_dict,
                                                         moa,drug2name_dict)

        if find_num > is_high:
            is_high = find_num
            save_df.to_csv('{}/src/result_case/{}_find.csv'.format(data_dir,save_name))



if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/'
    num_epoch = 20
    batch_size = 256
    logname = '20230424'
    device = torch.device("cuda")

    train(data_dir=data_dir,
        moa='MOA1',
        batch_size=batch_size,
        num_epoch=num_epoch,
        device=device,
        logname=logname)
    train(data_dir=data_dir,
          moa='MOA2',
          batch_size=batch_size,
          num_epoch=num_epoch,
          device=device,
          logname=logname)
    train(data_dir=data_dir,
          moa='MOA3',
          batch_size=batch_size,
          num_epoch=num_epoch,
          device=device,
          logname=logname)