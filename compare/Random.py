# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-Random.py
@time:2023/4/9 下午1:14
"""
import pandas as pd
import numpy as np
import random
random.seed(2023)

def get_data(train_file,test_file,sig2drugmoa_file):
    sig2drugmoa_dict = np.load(sig2drugmoa_file)

    train_df = pd.read_hdf(train_file)
    test_df = pd.read_hdf(test_file)
    print(train_df.shape, test_df.shape)

    train_moas = [sig2drugmoa_dict[i][1] for i in train_df.index]
    test_moas = [sig2drugmoa_dict[i][1] for i in test_df.index]


    return train_df,test_df,train_moas,test_moas

def random_select(train_labels,test_labels):
    train_num = len(train_labels)
    test_num  = len(test_labels)
    find_num = 0
    for one in test_labels:
        random_idx = random.randint(0,train_num-1)
        # print(random_idx,train_num)
        random_moa = train_labels[random_idx]
        # print(random_moa)
        if random_moa == one:
            find_num+=1
    print(train_num,test_num,find_num,round(find_num/test_num,4))
    return train_num,test_num,find_num,round(find_num/test_num,4)

if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)
    data_tas_dir = '{}/06_TAS/'.format(data_dir)
    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)

    save_file = 'result/random_tas.csv'
    f = open(save_file, 'w')

    for fold in [0, 1, 2]:
        # train_file = '{}/Train_fold_{}.h5'.format(data_all_dir,fold)
        # test_file = '{}/Test_fold_{}.h5'.format(data_all_dir,fold)
        train_file = '{}/Train_fold_{}.h5'.format(data_tas_dir, fold)
        test_file = '{}/Test_fold_{}.h5'.format(data_tas_dir, fold)
        print('#' * 20, 'ALL', '#' * 20)
        cell = 'ALL'
        train_df,test_df,train_moas,test_moas = get_data(train_file,test_file,sig2drugmoa_file)

        train_num,test_num,find_num,score =random_select(train_moas,test_moas)
        f.write('random,{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, find_num, score))

    # cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20',
    #                   'HCC515', 'HEPG2', 'HA1E', 'NPC', 'VCAP']
    # for cell in cell_lines:
    #     for fold in [0, 1, 2]:
    #         print('#' * 20, cell, '#' * 20)
    #         train_file = '{}/{}/Train_fold_{}.h5'.format(data_cell_dir, cell, fold)
    #         test_file = '{}/{}/Test_fold_{}.h5'.format(data_cell_dir, cell, fold)
    #         train_df, test_df, train_moas, test_moas = get_data(train_file, test_file, sig2drugmoa_file)
    #         train_num, test_num, find_num, score = random_select(train_moas,test_moas)
    #         f.write('random,{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, find_num, score))
    f.close()