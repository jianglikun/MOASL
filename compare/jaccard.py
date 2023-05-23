# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-jaccard.py
@time:2023/4/9 下午1:38
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from multiprocessing import Pool
from tqdm import tqdm
import random
random.seed(2023)

def get_data(train_file,test_file,sig2drugmoa_file):
    sig2drugmoa_dict = np.load(sig2drugmoa_file)
    train_df = pd.read_hdf(train_file)
    test_df = pd.read_hdf(test_file)
    print(train_df.shape, test_df.shape)
    train_moas = [sig2drugmoa_dict[i][1] for i in train_df.index]
    test_moas = [sig2drugmoa_dict[i][1] for i in test_df.index]
    print(train_df.shape, test_df.shape)

    gene_df = '/home/jlk/Project/111_Cmap/MOA/data/01_Info_file/geneinfo_beta.txt'
    gene_df = pd.read_csv(gene_df,sep='\t')
    gene_df = gene_df[gene_df['feature_space']=='landmark']
    landmark_gen = gene_df['gene_id'].tolist()
    landmark_gen = [str(i) for i in landmark_gen]
    train_df = train_df[landmark_gen]
    test_df = test_df[landmark_gen]
    return train_df,test_df,train_moas,test_moas

def distance(query):
    if type == 'cosine':
        distance_metric = cdist(query,train_df,metric='cosine')
        top_where = 0
    elif type == 'euclidean':
        distance_metric = cdist(query,train_df,metric='euclidean')
        top_where = 0
    elif type == 'jaccard':
        distance_metric = cdist(query>2,train_df>2,metric='jaccard')
        top_where = 0
    elif type == 'spearman':
        redefine_df = train_df.T
        redefine_df['query'] = list(query)
        distance_metric_df = spearmanr(redefine_df.values)[0]
        name = list(redefine_df.columns)
        distance_metric_df = pd.DataFrame(distance_metric_df,index=name,columns=name)
        distance_metric = distance_metric_df[train_df.index].loc[['query']]
        distance_metric = distance_metric.values
        top_where = -1

    find_idx = np.argsort(distance_metric[0])[top_where]
    find_moa = train_labels[find_idx]
    return find_moa

def distance_compute(train_df,test_df,train_labels,test_labels,type):
    train_num = len(train_labels)
    test_num  = len(test_labels)
    find_num = 0

    Xte_list = [test_df.iloc[i:i+1,:] for i in range(test_df.shape[0])]
    with Pool(4) as pool:
        results = list(tqdm(pool.imap(distance, Xte_list), total=test_num))
    # print(results)
    for i in range(test_num):
        true_moa = test_labels[i]
        find_moa = results[i]
        if find_moa == true_moa:
            find_num += 1
    print(train_num,test_num,find_num,round(find_num/test_num,4))
    return train_num,test_num,find_num,round(find_num/test_num,4)
    # return 0,1,2,3

if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)
    data_tas_dir = '{}/06_TAS/'.format(data_dir)

    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)


    for type in ['jaccard']:
        save_file = 'result/{}_tas_record.csv'.format(type)
        f = open(save_file, 'w')
        for fold in [0, 1, 2]:
            cell = 'ALL'
            # train_file = '{}/Train_fold_{}.h5'.format(data_all_dir,fold)
            # test_file = '{}/Test_fold_{}.h5'.format(data_all_dir,fold)
            train_file = '{}/Train_fold_{}.h5'.format(data_tas_dir, fold)
            test_file = '{}/Test_fold_{}.h5'.format(data_tas_dir, fold)
            print('#' * 20, 'ALL',type, '#' * 20)
            train_df, test_df, train_labels, test_labels = get_data(train_file, test_file, sig2drugmoa_file)
            train_num, test_num, find_num, score = distance_compute(train_df, test_df, train_labels, test_labels, type)
            f.write('{},{},{},{},{},{},{}\n'.format(type,cell, fold, train_num, test_num, find_num, score))

        # cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20',
        #                   'HCC515', 'HEPG2', 'HA1E', 'NPC', 'VCAP']
        # for cell in cell_lines:
        #     for fold in [0, 1, 2]:
        #         print('#' * 20, cell,type, '#' * 20)
        #         train_file = '{}/{}/Train_fold_{}.h5'.format(data_cell_dir, cell, fold)
        #         test_file = '{}/{}/Test_fold_{}.h5'.format(data_cell_dir, cell, fold)
        #
        #         train_df, test_df, train_labels, test_labels = get_data(train_file, test_file, sig2drugmoa_file)
        #         train_num, test_num, find_num, score = distance_compute(train_df,test_df,train_labels,test_labels,type)
        #         f.write('{},{},{},{},{},{},{}\n'.format(type,cell,fold,train_num,test_num,find_num,score))
        # f.close()
