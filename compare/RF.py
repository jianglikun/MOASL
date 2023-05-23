# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-SVM.py
@time:2022/7/20 下午8:05
"""

from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def get_data(train_file,test_file,sig2drugmoa_file):
    sig2drugmoa_dict = np.load(sig2drugmoa_file)

    train_df = pd.read_hdf(train_file)
    test_df = pd.read_hdf(test_file)
    print(train_df.shape, test_df.shape)

    train_moas = [sig2drugmoa_dict[i][1] for i in train_df.index]
    test_moas = [sig2drugmoa_dict[i][1] for i in test_df.index]
    print(train_df.shape, test_df.shape)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_df)
    train_df = scaling.transform(train_df)
    test_df = scaling.transform(test_df)

    # all_label = train_moas + test_moas
    # labelencoder = LabelEncoder()
    # labels = labelencoder.fit_transform(all_label)
    # label_dict = dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))

    return train_df,test_df,train_moas,test_moas

def train_test(train_df,test_df,train_labels,test_labels):
    # model = svm.SVC(kernel='linear',max_iter=100000)
    model = RandomForestClassifier()
    model.fit(train_df, train_labels)
    train_score = model.score(train_df, train_labels)
    print("训练集：", train_score)
    test_score = model.score(test_df, test_labels)
    print("测试集：", test_score)
    pred_test = model.predict(test_df)
    # print(pred_test)
    find_num = (pred_test == test_labels).sum().item()
    return train_df.shape[0],test_df.shape[0],find_num,test_score


if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)
    data_tas_dir = '{}/06_TAS/'.format(data_dir)
    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)

    save_file = 'result/random_forest_tas.csv'
    f = open(save_file, 'w')

    for fold in [0, 1, 2]:
        # train_file = '{}/Train_fold_{}.h5'.format(data_all_dir,fold)
        # test_file = '{}/Test_fold_{}.h5'.format(data_all_dir,fold)
        train_file = '{}/Train_fold_{}.h5'.format(data_tas_dir, fold)
        test_file = '{}/Test_fold_{}.h5'.format(data_tas_dir, fold)
        print('#' * 20, 'ALL', '#' * 20)
        cell = 'ALL'
        train_df,test_df,train_moas,test_moas = get_data(train_file,test_file,sig2drugmoa_file)

        train_num,test_num,find_num,score =train_test(train_df,test_df,train_moas,test_moas)
        f.write('RF,{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, find_num, score))


    # cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20',
    #                   'HCC515', 'HEPG2', 'HA1E', 'NPC', 'VCAP']
    # for cell in cell_lines:
    #     for fold in [0, 1, 2]:
    #         print('#' * 20, cell, '#' * 20)
    #         train_file = '{}/{}/Train_fold_{}.h5'.format(data_cell_dir, cell, fold)
    #         test_file = '{}/{}/Test_fold_{}.h5'.format(data_cell_dir, cell, fold)
    #         train_df, test_df, train_moas, test_moas = get_data(train_file, test_file, sig2drugmoa_file)
    #         train_num, test_num, find_num, score = train_test(train_df, test_df, train_moas, test_moas)
    #         f.write('RF,{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, find_num, score))
    f.close()