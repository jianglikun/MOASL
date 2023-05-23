# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-Benchmark.py
@time:2023/4/10 下午7:50
"""
import os
from multiprocessing import Pool
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from itertools import product, chain
import numpy as np
from CMapKS import runKS
from CMapGSEA import runGSEA

class RunMultiProcess(object):
    def __init__(self, methods=''):
        pass
    def myPool(self, func, mylist, processes):
        with Pool(processes) as pool:
            results = list(tqdm(pool.imap(func, mylist), total=len(mylist)))
        return results

def calCosine(Xtr, Xte):
    dat_cor = pd.DataFrame(euclidean_distances(Xte,Xtr))  ###行是Xte, 列是Xtr
    dat_cor.columns = Xtr.index
    dat_cor.index = Xte.index
    return dat_cor

def Pearson(query):
    ref = Xtr
    Nums = 10 if ref.shape[0] >=10 else ref.shape[0]
    query = query.sort_values(by=query.index[0], axis=1)   ####  基因排序
    genes = query.columns[:100].tolist() + query.columns[-100:].tolist()  ### 改变基因的数量
    query = query[genes]; ref = ref[genes]
    dat_cor = calCosine(Xtr = ref, Xte = query)
    for i in dat_cor.index:
        tmp = dat_cor.loc[i,:]
        positive = tmp.sort_values(ascending=True)[:Nums].index.tolist()
        values = tmp.sort_values(ascending=True)[:Nums].values.tolist()
        values = [str(round(i,4)) for i in values]
    return (query.index[0], positive, values)

### 对每个query做循环
def f_Pearson(train_file,test_file,sig2drugmoa_dict,drug2moa_dict):
    global Xtr, method
    method = calCosine
    doMultiProcess = RunMultiProcess()

    Xtr = pd.read_hdf(train_file); Xte = pd.read_hdf(test_file)
    train_num = Xtr.shape[0]
    pert_iname = [sig2drugmoa_dict[i][0] for i in Xtr.index]
    Xtr = Xtr.groupby(pert_iname, axis=0).median()
    tmp = [i for i in Xtr.columns if i in Xte.columns]
    Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
    Xte_list = [Xte.iloc[i:i+1,:] for i in range(Xte.shape[0])]
    results = doMultiProcess.myPool(Pearson, Xte_list, 32)
    test_num = Xte.shape[0]
    find_num =0
    for index, positive, values in results:
        true_moa = sig2drugmoa_dict[index][1]
        find_moa = drug2moa_dict[positive[0]]
        if true_moa == find_moa:
            find_num +=1
    print(train_num, test_num, find_num, round(find_num / test_num, 4))
    return train_num,test_num,find_num,round(find_num/test_num,4)

if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)
    data_tas_dir = '{}/06_TAS/'.format(data_dir)

    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)
    sig2drugmoa_dict = np.load(sig2drugmoa_file)
    drug2moa_dict = np.load(drug2moa_file)

    save_file = 'result/euclidean_tas_record.csv'.format(type)
    f = open(save_file, 'w')

    for fold in [0,1,2]:
        cell = 'ALL'
        # train_file = '{}/Train_fold_{}.h5'.format(data_all_dir,fold)
        # test_file = '{}/Test_fold_{}.h5'.format(data_all_dir,fold)
        train_file = '{}/Train_fold_{}.h5'.format(data_tas_dir, fold)
        test_file = '{}/Test_fold_{}.h5'.format(data_tas_dir, fold)
        print('#' * 20, 'ALL euclidean', '#' * 20)
        train_num, test_num, find_num, score = f_Pearson(train_file, test_file, sig2drugmoa_dict, drug2moa_dict)
        f.write('euclidean,{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, find_num, score))

    # cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20',
    #                       'HCC515', 'HEPG2', 'HA1E', 'NPC', 'VCAP']
    # for cell in cell_lines:
    #     for fold in [0, 1, 2]:
    #         print('#' * 20, cell,fold, '#' * 20)
    #         train_file = '{}/{}/Train_fold_{}.h5'.format(data_cell_dir, cell, fold)
    #         test_file = '{}/{}/Test_fold_{}.h5'.format(data_cell_dir, cell, fold)
    #         file_name = '{}_fold_{}'.format(cell,fold)
    #         train_num, test_num, find_num, score = f_Pearson(train_file,test_file,sig2drugmoa_dict,drug2moa_dict)
    #         f.write('euclidean,{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, find_num, score))
    #
    # f.close()
